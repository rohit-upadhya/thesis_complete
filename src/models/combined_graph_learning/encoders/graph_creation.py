import torch # type: ignore
from torch_geometric.data import Data# type: ignore
from rank_bm25 import BM25Okapi # type: ignore
from joblib import Parallel, delayed # type: ignore
import heapq

class GraphCreator:
    def __init__(
            self,
        ) -> None:
            pass
        
    def build_graph(self, 
        paragraph_encodings, 
        paragraphs,
        topic_model,
        use_topics=False,
        use_all=False,
        use_bm25=False,
        use_cosine=False,
        use_prev_next_five=True,
        use_threshold=False
    ):
        """
        Builds a graph with various edge creation strategies:
        - Fully connected graph (use_all=True)
        - BM25-based top-k neighbors (use_bm25=True)
        - Cosine similarity-based top-k neighbors (use_cosine=True)
        """
        node_features = paragraph_encodings
        num_paragraphs = len(paragraph_encodings)
        edge_index = []
        topic_embeddings = []
        if use_topics:
            prob, topic_embeddings = topic_model.obtain_topic_embeddings(embeddings=paragraph_encodings, paragraphs=paragraphs)
            topics_tensor = torch.tensor(topic_embeddings, dtype=paragraph_encodings.dtype, device=paragraph_encodings.device)
            node_features = torch.cat((paragraph_encodings, topics_tensor), dim=0)
            for i, topic in enumerate(topic_embeddings):
                for j, paragraph in enumerate(paragraph_encodings):
                    if use_threshold:
                        if prob[j][i] > 0.30: 
                            edge_index.append([i+len(paragraph_encodings), j])
                            edge_index.append([j, i+len(paragraph_encodings)])
                    else:
                        edge_index.append([i+len(paragraph_encodings), j])
                        edge_index.append([j, i+len(paragraph_encodings)])
                    
        if use_all:
            for i in range(num_paragraphs):
                for j in range(num_paragraphs):
                    if i != j:
                        edge_index.append([i, j])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            return Data(x=node_features, edge_index=edge_index)

        
        if use_bm25:
            tokenized_corpus = [p.split() for p in paragraphs]
            bm25 = BM25Okapi(tokenized_corpus)
            def get_top_nodes_bm25(i):
                query = tokenized_corpus[i]
                scores = bm25.get_scores(query)
                indices_and_scores = [(idx, score) for idx, score in enumerate(scores) if idx != i]
                top_nodes = [idx for idx, _ in heapq.nlargest(10, indices_and_scores, key=lambda x: x[1])]
                return i, top_nodes
            bm25_results = Parallel(n_jobs=-1)(delayed(get_top_nodes_bm25)(i) for i in range(num_paragraphs))
        
        if use_cosine:
            encodings_tensor = torch.tensor(paragraph_encodings, device='cuda', dtype=torch.float32)
            encodings_norm = encodings_tensor / encodings_tensor.norm(dim=1, keepdim=True)
            similarity_matrix = torch.matmul(encodings_norm, encodings_norm.T)
            similarity_matrix = similarity_matrix.cpu().numpy()
            
            def get_top_nodes_cosine(i):
                scores = similarity_matrix[i]
                indices_and_scores = [(idx, score) for idx, score in enumerate(scores) if idx != i]
                top_nodes = [idx for idx, _ in heapq.nlargest(10, indices_and_scores, key=lambda x: x[1])]
                return i, top_nodes
            
            cosine_results = Parallel(n_jobs=-1)(delayed(get_top_nodes_cosine)(i) for i in range(len(paragraph_encodings)))
        
        
        if use_prev_next_five:
            for i in range(num_paragraphs):
                if i > 0:
                    if [i, i-1] not in edge_index:
                        edge_index.append([i, i-1])
                if i > 1:
                    if [i, i-2] not in edge_index:
                        edge_index.append([i, i-2])
                if i > 2:
                    if [i, i-3] not in edge_index:
                        edge_index.append([i, i-3])
                if i > 3:
                    if [i, i-4] not in edge_index:
                        edge_index.append([i, i-4])
                if i > 4:
                    if [i, i-5] not in edge_index:
                        edge_index.append([i, i-5])
                if i < num_paragraphs - 1:
                    if [i, i+1] not in edge_index:
                        edge_index.append([i, i+1])
                if i < num_paragraphs - 2:
                    if [i, i+2] not in edge_index:
                        edge_index.append([i, i+2])
                if i < num_paragraphs - 3:
                    if [i, i+3] not in edge_index:
                        edge_index.append([i, i+3])
                if i < num_paragraphs - 4:
                    if [i, i+4] not in edge_index:
                        edge_index.append([i, i+4])
                if i < num_paragraphs - 5:
                    if [i, i+5] not in edge_index:
                        edge_index.append([i, i+5])

        if use_bm25:
            for i, top_nodes in bm25_results:
                for item in top_nodes:
                    if [i, item] not in edge_index:
                        edge_index.append([i, item])

        if use_cosine:
            for i, top_nodes in cosine_results:
                for item in top_nodes:
                    if [i, item] not in edge_index:
                        edge_index.append([i, item])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        data = Data(x=node_features, edge_index=edge_index)
        
        data.num_paragraphs = num_paragraphs
        if use_topics:
            data.num_topics = len(topic_embeddings)
        else:
            data.num_topics = 0
        return data