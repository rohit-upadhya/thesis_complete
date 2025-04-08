from typing import Text, List, Dict, Any, Optional
import numpy as np # type: ignore
import os
import json
import math
from tqdm import tqdm # type: ignore
import warnings
warnings.filterwarnings("ignore")
import torch # type: ignore
from torch_geometric.data import Data, Batch # type: ignore
from rank_bm25 import BM25Okapi # type: ignore
from joblib import Parallel, delayed # type: ignore
import heapq
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from time import sleep

from src.models.vector_db.commons.input_loader import InputLoader
from src.models.vector_db.inference.encoder import Encoder
from src.models.vector_db.inference.val_encoder import ValEncoder
from src.models.vector_db.inference.faiss_vector_db import FaissVectorDB
# from src.models.graph_learning.inference.inference_paragraph_gat import ParagraphGAT
# from src.models.graph_learning.inference.new_inference_gat import ParagraphGATInference
from src.models.graph_learning.encoders.paragraph_gat import ParagraphGAT as ParagraphGATInference
# from src.models.graph_learning.inference.inference_topic_modeling import TopicModeling
from src.models.graph_learning.train.new_topic_modeling import TopicModeling
from src.models.graph_learning.encoders.graph_creation import GraphCreation

class Inference:
    def __init__(
            self, 
            inference_folder: Optional[Text] = 'input/train_infer/english/new_split/test', 
            inference_datapoints: Optional[List[Dict[str, Any]]] = None, 
            bulk_inference: bool = False,
            use_translations: bool = False,
            device: str = "cpu",
            language: str = "english",
            question_model_name_or_path: str = 'bert-base-multilingual-cased',
            ctx_model_name_or_path: str = 'bert-base-multilingual-cased',
            save_recall: bool = True,
            run_val: bool = False,
            tokenizer: Optional[str] = None,
            model: Optional[str] = None,
            device_: Optional[str] = None,
            graph_model: str = '/srv/upadro/models/graph/2024-12-13___all_gat___training/_final_model/graph_model.pt',
            comment: str = 'topic',
            use_bm25: bool = False,
            use_topics: bool = False,
            use_cosine: bool = False,
            use_prev_next_two: bool = True,
            use_threshold: bool = False,
        ):
        self.use_topics = use_topics
        self.use_bm25 = use_bm25
        self.use_all = False
        self.input_loader = InputLoader()
        self.file_base_name = language
        self.use_translations = use_translations
        self.run_val = run_val
        self.recall_file_name = f"recall_{graph_model.replace('/', '_')}_{'translated' if use_translations else 'not_translated'}_{inference_folder.split('/')[-1]}.json" # type: ignore
        self.language = language
        self.save_recall = save_recall
        self.use_threshold = use_threshold
        if bulk_inference:
            inference_datapoints = self._load_all_input_from_dir(inference_folder)
            self.file_base_name = inference_folder.split("/")[-3] # type: ignore
        self._format_input(inference_datapoints)
        self.folder_name = inference_folder.split('/')[-1] # type: ignore
        if run_val:
            self.encoder = ValEncoder(question_model=model, ctx_model=model, question_tokenizer=tokenizer, ctx_tokenizer=tokenizer, device=device_)
        else:
            self.encoder = Encoder(device=device, question_model_name_or_path=question_model_name_or_path, ctx_model_name_or_path=ctx_model_name_or_path, use_dpr=False, use_roberta=False)
        self.faiss = FaissVectorDB()
        self.comment = comment
        self.graph_model = graph_model
        self.all_paragraph_encodings = []
        self.all_unique_keys = []
        self.query_paragraph_mapping = {}
        self.model_trained_language = question_model_name_or_path.split("_")[-4] if "training" in question_model_name_or_path else "base"
        self.topic_model = TopicModeling()
        self.use_cosine = use_cosine
        self.use_prev_next_two = use_prev_next_two
        self.graph_creation = GraphCreation(use_topics=self.use_topics,
            use_all=False,
            use_bm25 = self.use_bm25,
            use_cosine = self.use_cosine,
            use_prev_next_two = self.use_prev_next_two,
            topic_model = self.topic_model)
        print("self.use_topics", self.use_topics)
        print("self.use_cosine", self.use_cosine)
        print("self.use_prev_next_two", self.use_prev_next_two)
    
    def _build_graph(self, paragraph_encodings, paragraphs):
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
        if self.use_topics:
            prob, topic_embeddings = self.topic_model.obtain_topic_embeddings(embeddings=paragraph_encodings, paragraphs=paragraphs)
            topics_tensor = torch.tensor(topic_embeddings, dtype=paragraph_encodings.dtype, device=paragraph_encodings.device)
            node_features = torch.cat((paragraph_encodings, topics_tensor), dim=0)
            for i, topic in enumerate(topic_embeddings):
                for j, paragraph in enumerate(paragraph_encodings):
                    if self.use_threshold:
                        # if self.print:
                        #     print("use_threshold",self.use_threshold)
                        #     self.print = not self.print
                        if prob[j][i] > 0.30: 
                            edge_index.append([i+len(paragraph_encodings), j])
                            edge_index.append([j, i+len(paragraph_encodings)])
                    else:
                        edge_index.append([i+len(paragraph_encodings), j])
                        edge_index.append([j, i+len(paragraph_encodings)])
                    
        if self.use_all:
            for i in range(num_paragraphs):
                for j in range(num_paragraphs):
                    if i != j:
                        edge_index.append([i, j])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            return Data(x=node_features, edge_index=edge_index)

        
        if self.use_bm25:
            tokenized_corpus = [p.split() for p in paragraphs]
            bm25 = BM25Okapi(tokenized_corpus)
            def get_top_nodes_bm25(i):
                query = tokenized_corpus[i]
                scores = bm25.get_scores(query)
                indices_and_scores = [(idx, score) for idx, score in enumerate(scores) if idx != i]
                top_nodes = [idx for idx, _ in heapq.nlargest(10, indices_and_scores, key=lambda x: x[1])]
                return i, top_nodes
            bm25_results = Parallel(n_jobs=-1)(delayed(get_top_nodes_bm25)(i) for i in range(num_paragraphs))
        
        if self.use_cosine:
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
        
        # if self.use_cosine:
        #     with torch.no_grad():
        #         encodings_tensor = torch.tensor(paragraph_encodings, device='cuda', dtype=torch.float32)
        #         encodings_norm = encodings_tensor / encodings_tensor.norm(dim=1, keepdim=True)

        #         similarity_matrix = torch.matmul(encodings_norm, encodings_norm.T)

        #         threshold_mask = similarity_matrix > self.cosine_threshold
        #         threshold_mask.fill_diagonal_(False)
                
        #         matching_indices = torch.nonzero(threshold_mask, as_tuple=False)

        #         from collections import defaultdict
        #         cosine_results = defaultdict(list)
        #         for row, col in matching_indices:
        #             cosine_results[row.item()].append(col.item())

        #         cosine_results = [(i, matches) for i, matches in cosine_results.items()]


        
        if self.use_prev_next_two:
            
            for i in range(num_paragraphs):
                if i > 0:
                    if [i, i-1] not in edge_index:
                        edge_index.append([i, i-1])
                if i > 1:
                    if [i, i-2] not in edge_index:
                        edge_index.append([i, i-2])
                if i < num_paragraphs - 1:
                    
                    if [i, i+1] not in edge_index:
                        edge_index.append([i, i+1])
                if i < num_paragraphs - 2:
                    if [i, i+2] not in edge_index:
                        edge_index.append([i, i+2])

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
        
        # if self.use_cosine:
        #     for i, above_threshold_nodes in cosine_results:
        #         for item in above_threshold_nodes:
        #             if [i, item] not in edge_index:
        #                 edge_index.append([i, item])
                                    
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        data = Data(x=node_features, edge_index=edge_index)
        
        data.num_paragraphs = num_paragraphs
        if self.use_topics:
            data.num_topics = len(topic_embeddings)
        else:
            data.num_topics = 0
        return data
    
    def visualize_graph(self, data, file_name="src/models/graph_learning/train/graph.png"):
        """
        Convert a PyTorch Geometric Data object to a NetworkX graph and visualize it,
        ensuring correct edge weight mapping.
        """
        import matplotlib.pyplot as plt  # type: ignore
        import networkx as nx # type: ignore
        from torch_geometric.utils import to_networkx # type: ignore

        # Convert the graph to NetworkX
        G = to_networkx(data, to_undirected=True)

        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(G, seed=42)  # Layout for positioning nodes

        # Draw nodes
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color='skyblue',
            edge_color='gray',
            node_size=800,
            font_size=8,
            font_color='black',
            alpha=0.9
        )

        # Map edge weights to edges explicitly
        edge_weights = data.edge_attr.cpu().numpy()  # Ensure edge_attr is on CPU
        edge_index = data.edge_index.cpu().numpy().T  # Convert edge_index to numpy for mapping
        edge_labels = {}

        for idx, (src, dst) in enumerate(edge_index):
            if (src, dst) in G.edges():
                edge_labels[(src, dst)] = f"{edge_weights[idx]:.2f}"
            if (dst, src) in G.edges(): 
                edge_labels[(dst, src)] = f"{edge_weights[idx]:.2f}"

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.title("Graph Visualization with Edge Weights")
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.show()
        
    def _get_graph_encodings(self, gnn_model, encodings, all_paragraphs):
        graph_data = self._build_graph(encodings, paragraphs=all_paragraphs)
        updated_data = gnn_model(graph_data)
        updated_paragraph_vectors = updated_data.x[:graph_data.num_paragraphs]
        return updated_paragraph_vectors.detach()
    
    def _load_all_input_from_dir(self, input_data_path):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(input_data_path):
            for filename in filenames:
                if "json" in filename:
                    files.append(os.path.join(dirpath, filename))
        total_inference_datapoints = []
        for file in files:
            individual_datapoints = self.input_loader.load_data(data_file=file)
            total_inference_datapoints.extend(individual_datapoints) # type: ignore
        print("self.run_val", self.run_val)
        if self.run_val:
            return total_inference_datapoints
        return total_inference_datapoints


    def _format_input(self, inference_datapoints):
        self.data_points = []
        
        translatons_query = {}
        if self.language != 'english' and self.use_translations:
            translatons_query = self.input_loader.load_data(data_file=f"output/translation_outputs/query_translations_{self.language}.json")
        for idx, data in enumerate(inference_datapoints):
            key = data["link"]
            final_query = []
            if self.use_translations and self.language != 'english':
                for query_item in data["query"]:
                    for trdata_point in translatons_query: # type: ignore
                        if trdata_point["original"] == query_item:
                            final_query.append(trdata_point["translation"])
            else:
                final_query = data["query"]
            
            unique_key = f"datapoint_{key}"
            self.data_points.append(
                {
                    "query": ", ".join(final_query),
                    "all_paragraphs": ["\n".join(paras) for paras in data["all_paragraphs"]],
                    "unique_keys": [f"{unique_key}_para_{i+1}" for i in range(len(data["all_paragraphs"]))],
                    "paragraph_numbers": data.get("paragraph_numbers", []),
                    "link": key,
                    "length_of_all_paragraphs": len(data["all_paragraphs"])
                }
            )
        print(self.data_points[0]["query"])
        print(inference_datapoints[0]["query"])
    
    
    def _encode_all_paragraphs(self, gnn_model, batch_size=192, ):
        index_counter = 0
        paragraph_set = set()
        total_paragraphs = sum(len(points["all_paragraphs"]) for points in self.data_points if points["link"] not in paragraph_set)
        with tqdm(total=total_paragraphs, desc="Encoding paragraphs", unit="paragraph") as pbar:
            for idx, points in enumerate(self.data_points):
                if points["link"] not in paragraph_set:
                    all_paragraphs = points["all_paragraphs"]
                    num_paragraphs = len(all_paragraphs)
                    encoded_paragraphs = []

                    for start_idx in range(0, num_paragraphs, batch_size):
                        end_idx = min(start_idx + batch_size, num_paragraphs)
                        batch_paragraphs = all_paragraphs[start_idx:end_idx]
                        
                        encoded_batch = self.encoder.encode_ctx(batch_paragraphs).cpu().numpy()
                        encoded_paragraphs.append(encoded_batch)
                        pbar.update(end_idx - start_idx)

                    encoded_paragraphs = np.vstack(encoded_paragraphs)
                    encoded_paragraphs = torch.from_numpy(encoded_paragraphs)
                    encoded_paragraphs = self._get_graph_encodings(gnn_model=gnn_model, encodings=encoded_paragraphs, all_paragraphs=points["all_paragraphs"])
                    
                    self.all_paragraph_encodings.append(encoded_paragraphs)
                    self.all_unique_keys.extend(points["unique_keys"])
                    self.query_paragraph_mapping[idx] = list(range(index_counter, index_counter + len(all_paragraphs)))
                    index_counter += len(all_paragraphs)
                    paragraph_set.add(points["link"])
                    
        all_encodings = np.vstack(self.all_paragraph_encodings)
        self.faiss.build_index(all_encodings, self.all_unique_keys)

    def _encode_query(self, query: Text):
        return self.encoder.encode_question([query]).cpu().numpy()
    
    
    def recall_at_k(self, actual, predicted, k):
        relevance = [1 if x in actual else 0 for x in predicted]
        r = np.asarray(relevance)[:k]
        return np.sum(r) / len(actual) if len(actual) > 0 else 0

    def calculate_recall(self, results):
        percentages = [2, 5, 10]
        recall = {}
        
        for percentage in percentages:
            r_at_percentage = []
            for idx, result in enumerate(results):
                query_data = self.data_points[idx]
                actual_relevant = set(query_data.get('paragraph_numbers', []))
                
                predicted_relevant = result[f'relevant_paragraphs']
                k = result[f"no_{percentage}"]
                recall_score = self.recall_at_k(actual_relevant, predicted_relevant, k)
                r_at_percentage.append(recall_score)
            
            recall[f"mean_recall_at_{percentage}_percentage"] = np.mean(np.asarray(r_at_percentage))
        
        return recall

    def _load_model(self, model_path, hidden_dim, graph_model_type="gat"):
        """
        Load the trained graph model from a saved state dict.
        """
        if graph_model_type == "gat":
            model = ParagraphGATInference(hidden_dim=hidden_dim)
        else:
            raise NotImplementedError("Load the appropriate model architecture here.")

        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
        
    def main(self):
        gnn_model = self._load_model(model_path=self.graph_model, hidden_dim=768)
        self._encode_all_paragraphs(gnn_model=gnn_model, batch_size=512)
        results = []
        print(self.graph_model)
        for idx, points in enumerate(self.data_points):
            query_encodings = self._encode_query(points['query'])
            number_of_relevant_paragraph_2 = max(int(math.ceil(points["length_of_all_paragraphs"] * 0.02)), 1)
            number_of_relevant_paragraph_5 = max(int(math.ceil(points["length_of_all_paragraphs"] * 0.05)), 1)
            number_of_relevant_paragraph_10 = max(int(math.ceil(points["length_of_all_paragraphs"] * 0.1)), 1)            
            relevant_paragraph_keys = self.faiss.perform_search(
                query=query_encodings,
                datapoint = points,
            )
            results.append({
                "query": points["query"],
                "link": points["link"],
                "relevant_paragraphs": [int(paragraph.split("_")[-1]) for paragraph in relevant_paragraph_keys],
                "no_2": number_of_relevant_paragraph_2,
                "no_5": number_of_relevant_paragraph_5,
                "no_10": number_of_relevant_paragraph_10
            })
            
        recall = self.calculate_recall(results)
        print(f"Recall: {recall}")
        
        recall_data = {}
        if self.save_recall:
            recall_path = os.path.join('output/inference_outputs/new_splits/new_experiments/graph/',self.comment, self.model_trained_language, self.folder_name, self.recall_file_name)
            recall_dir = os.path.dirname(recall_path)
            os.makedirs(recall_dir, exist_ok=True)
            if os.path.exists(recall_path):
                with open(recall_path, 'r') as json_file:
                    recall_data = json.load(json_file)

            recall_data[self.file_base_name] = recall
            recall_data["model"] = self.graph_model
            with open(recall_path, 'w+') as json_file:
                json.dump(recall_data, json_file, indent=4, ensure_ascii=False)
        return recall

if __name__ == "__main__":
    # languages = ["english", "french", "italian", "romanian", "russian", "turkish", "ukrainian"]
    # languages = ["russian"]
    # file = "val"
    
    # translations  = [True, False]
    # file = "val"
    # models = [
    #     ['bert-base-multilingual-cased', 'bert-base-multilingual-cased'],
    #     ['castorini/mdpr-tied-pft-msmarco', 'castorini/mdpr-tied-pft-msmarco'],
    #     ['castorini/mdpr-tied-pft-msmarco-ft-all', 'castorini/mdpr-tied-pft-msmarco-ft-all'],
    #     ['bert-base-uncased', 'bert-base-uncased']
    # ]

    
    # models = [
    #     ["facebook/dpr-question_encoder-single-nq-base",
    #     # "facebook/dpr-question_encoder-single-nq-base"]
         
    #      "facebook/dpr-ctx_encoder-single-nq-base"]
    # ]
    
    files = [
            'unique_query', 
            'test'
        ]
    # files = ['test']
    models = [
            [
                "castorini/mdpr-tied-pft-msmarco",
                "castorini/mdpr-tied-pft-msmarco",
                "/srv/upadro/models/new_expt/graph/2025-01-17___all_gat_final_new_graph_expts_no_norm_base_prev_next_2_training/checkpoints/epoch_20/graph_model.pt",
                "final_new_graph_expts_no_norm_base_prev_next_2",
                {
                    "use_topics": False,
                    "use_cosine": False,
                    "use_prev_next_two": True,
                    "use_topic_threshold": False,
                }
            ],
]


    print(models[0][4])
    for model in models:
        for file in files:
            translations  = [
                    # True, 
                    False
                ]
            for use_translations in translations:
                # languages = ["english", "russian", "french", "italian", "romanian", "turkish", "ukrainian"]
                languages = ["english", "french", "italian", "romanian", "russian", "turkish", "ukrainian"]
                # languages = ["french", "italian", "romanian", "russian", "turkish", "ukrainian"]
                # languages = ["italian"]
                for language in tqdm(languages, desc="Processing Languages"):
                    print(f"Processing language: {language}")
                    print(f"translations : {use_translations}")
                    
                    inference_folder = f"input/train_infer/{language}/new_split/{file}"
                    bulk_inference = True
                    # use_translations = False
                    print(model[3])
                    inference = Inference(
                        inference_folder=inference_folder, 
                        bulk_inference=bulk_inference,
                        use_translations=use_translations,
                        device='cuda:2',
                        language=language,
                        question_model_name_or_path = model[0],
                        ctx_model_name_or_path = model[1],
                        graph_model = model[2],
                        comment=model[3],
                        use_bm25=False,
                        use_topics=model[4]["use_topics"],
                        use_cosine=model[4]["use_cosine"],
                        use_prev_next_two=model[4]["use_prev_next_two"],
                        use_threshold = model[4]["use_topic_threshold"],
                    )
                    inference.main()
                    
                    print(f"Completed processing for language: {language}")
                    print("*" * 40)
                    
                print(f"Inference process completed for all languages for {'translated' if use_translations else 'not translated'} for model {model} for {file} datapoints")
    # sleep(180)