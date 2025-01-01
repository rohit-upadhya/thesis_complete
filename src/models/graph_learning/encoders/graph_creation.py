import torch # type: ignore
from torch_geometric.data import Data# type: ignore
from rank_bm25 import BM25Okapi # type: ignore
from joblib import Parallel, delayed # type: ignore
import heapq
from sklearn.metrics.pairwise import cosine_similarity

class GraphCreation:
    def __init__(
            self,
            use_topics,
            use_all,
            use_bm25,
            use_cosine,
            use_prev_next_two,
            topic_model
        ) -> None:
        self.use_topics = use_topics
        self.use_all = use_all
        self.use_bm25 = use_bm25
        self.use_cosine = use_cosine
        self.use_prev_next_two = use_prev_next_two
        self.topic_model = topic_model
        
    def build_graph(self, paragraph_encodings, paragraphs):
        """
        Builds a graph with various edge creation strategies
        """
        node_features = paragraph_encodings
        num_paragraphs = len(paragraph_encodings)
        edge_index = []
        topic_embeddings = []
        edge_weights = []
        if self.use_topics:
            probability, topic_embeddings = self.topic_model.obtain_topic_embeddings(embeddings=paragraph_encodings, paragraphs=paragraphs)
            topics_tensor = torch.tensor(topic_embeddings, dtype=paragraph_encodings.dtype, device=paragraph_encodings.device)
            node_features = torch.cat((paragraph_encodings, topics_tensor), dim=0)
            
            for i, topic in enumerate(topic_embeddings):
                topic = torch.tensor(topic).unsqueeze(0)
                for j, paragraph in enumerate(paragraph_encodings):
                    paragraph = paragraph.unsqueeze(0)
                    weight = probability[j][i]
                    edge_index.append([i+len(paragraph_encodings), j])
                    edge_index.append([j, i+len(paragraph_encodings)])
                    edge_weights.extend([weight, weight])
                    
        if self.use_all:
            for i in range(num_paragraphs):
                for j in range(num_paragraphs):
                    if i != j:
                        edge_index.append([i, j])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            return Data(x=node_features, edge_index=edge_index)

        tokenized_corpus = [p.split() for p in paragraphs]
        
        if self.use_bm25:
            bm25 = BM25Okapi(tokenized_corpus)
            def get_top_nodes_bm25(i):
                query = tokenized_corpus[i]
                scores = bm25.get_scores(query)
                indices_and_scores = [(idx, score) for idx, score in enumerate(scores) if idx != i]
                top_nodes = [idx for idx, _ in heapq.nlargest(10, indices_and_scores, key=lambda x: x[1])]
                return i, top_nodes
            bm25_results = Parallel(n_jobs=-1)(delayed(get_top_nodes_bm25)(i) for i in range(num_paragraphs))
        
        if self.use_cosine:
            similarity_matrix = cosine_similarity(paragraph_encodings, paragraph_encodings)
            def get_top_nodes_cosine(i):
                scores = similarity_matrix[i]
                indices_and_scores = [(idx, score) for idx, score in enumerate(scores) if idx != i]
                top_nodes = [idx for idx, _ in heapq.nlargest(10, indices_and_scores, key=lambda x: x[1])]
                return i, top_nodes
            
            cosine_results = Parallel(n_jobs=-1)(delayed(get_top_nodes_cosine)(i) for i in range(num_paragraphs))
        
        if self.use_prev_next_two:
            for i in range(num_paragraphs):
                if i > 0:
                    if [i, i-1] not in edge_index:
                        edge_index.append([i, i-1])
                        weight = cosine_similarity(
                            paragraph_encodings[i].unsqueeze(0),
                            paragraph_encodings[i-1].unsqueeze(0)
                        ).item()
                        edge_weights.append(weight)
                if i > 1:
                    if [i, i-2] not in edge_index:
                        edge_index.append([i, i-2])
                        weight = cosine_similarity(
                            paragraph_encodings[i].unsqueeze(0),
                            paragraph_encodings[i-2].unsqueeze(0)
                        ).item()
                        edge_weights.append(weight)
                if i < num_paragraphs - 1:
                    
                    if [i, i+1] not in edge_index:
                        edge_index.append([i, i+1])
                        weight = cosine_similarity(
                            paragraph_encodings[i].unsqueeze(0),
                            paragraph_encodings[i+1].unsqueeze(0)
                        ).item()
                        edge_weights.append(weight)
                if i < num_paragraphs - 2:
                    if [i, i+2] not in edge_index:
                        edge_index.append([i, i+2])
                        weight = cosine_similarity(
                            paragraph_encodings[i].unsqueeze(0),
                            paragraph_encodings[i+2].unsqueeze(0)
                        ).item()
                        edge_weights.append(weight)

        if self.use_bm25:
            for i, top_nodes in bm25_results:
                for item in top_nodes:
                    if [i, item] not in edge_index:
                        edge_index.append([i, item])

        if self.use_cosine:
            for i, top_nodes in cosine_results:
                for item in top_nodes:
                    if [i, item] not in edge_index:
                        edge_index.append([i, item])
                                    
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).t().contiguous()
        
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        data.num_paragraphs = num_paragraphs
        if self.use_topics:
            data.num_topics = len(topic_embeddings)
        else:
            data.num_topics = 0
        return data