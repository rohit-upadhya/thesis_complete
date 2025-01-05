from dataclasses import dataclass
from typing import Optional, Text
import torch # type: ignore
import os
import logging
from tqdm import tqdm # type: ignore
import numpy as np # type: ignore
from torch_geometric.data import Data, Batch # type: ignore
import torch.optim as optim # type: ignore
import torch.nn as nn # type: ignore
import random
import pickle
import gc
from rank_bm25 import BM25Okapi # type: ignore
from joblib import Parallel, delayed # type: ignore
import heapq
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import torch.nn.functional as F # type: ignore

from sentence_transformers import SentenceTransformer
from src.models.single_datapoints.common.utils import current_date
from src.models.single_datapoints.common.data_loader import InputLoader
from src.models.graph_learning.encoders.paragraph_gcn import ParagraphGNN
from src.models.graph_learning.encoders.paragraph_gat import ParagraphGAT
from src.models.graph_learning.encoders.graph_encoder import GraphEncoder as Encoder
from src.models.graph_learning.train.newer_topic_modeling import TopicModeling

class GraphTrainer:
    def __init__(
        self,
        use_dpr=False,
        use_roberta=False,
        train_data_folder=None,
        val_data_folder=None,
        config_file=None,
        batch_size=4,
        val_batch_size=8,
        individual_datapoints=7,
        epochs=1,
        device_str='cpu',
        dual_encoders=True,
        language='english',
        lr=2e-5,
        save_checkpoints=False,
        step_validation=False,
        model_name_or_path="bert-base-multilingual-cased",
        query_model_name_or_path="bert-base-multilingual-cased",
        ctx_model_name_or_path="bert-base-multilingual-cased",
        use_translations=False,
        num_negatives=7,
        graph_model: Text = "gcn",
        comments = "",
        use_bm25 = False,
        use_all=False,
        use_cosine = False,
        use_topics = False,
        use_prev_next_two = True,
        sentence_model: Text ="all-mpnet-base-v2",
        use_threshold: bool = False,
    ):
        self.use_dpr = use_dpr
        self.use_topics = use_topics
        self.use_roberta = use_roberta
        self.train_data_folder = train_data_folder
        self.val_data_folder = val_data_folder
        self.config_file = config_file
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.individual_datapoints = individual_datapoints
        self.epochs = epochs
        self.device_str = device_str
        self.dual_encoders = dual_encoders
        self.language = language
        self.lr = lr
        self.save_checkpoints = save_checkpoints
        self.step_validation = step_validation
        self.model_name_or_path = model_name_or_path
        self.query_model_name_or_path = query_model_name_or_path
        self.ctx_model_name_or_path = ctx_model_name_or_path
        self.use_translations = use_translations
        self.num_negatives = num_negatives
        self.graph_model = graph_model
        self.save_model_name = self.query_model_name_or_path.replace('/','_')
        self.current_date = current_date()
        self.input_loader = InputLoader()
        self.comments = comments
        self.use_bm25 = use_bm25
        self.use_all = use_all
        self.use_cosine = use_cosine
        self.use_prev_next_two = use_prev_next_two
        self.use_threshold = use_threshold
        if self.config_file == "":
            raise ValueError("config file empty, please contact admin")
        self.topic_encoder_model = SentenceTransformer(sentence_model)
        self.topic_model = TopicModeling(embedding_model=self.topic_encoder_model)
        self.config = self.input_loader.load_config(self.config_file)  # type: ignore
        self.config = self.config["dual_encoder"] if self.dual_encoders else self.config["single_encoder"] # type: ignore

        self.device = torch.device(
            self.device_str
            if torch.cuda.is_available() and 'cuda' in self.device_str
            else 'cpu'
        )
        self.encoding_type = "dual" if self.dual_encoders else "single"
        self.recall_save = self.query_model_name_or_path.replace('/', '_')
        self.save_model_name = self.query_model_name_or_path.replace('/', '_')

        self.encoder = Encoder(
            device=self.device_str,
            question_model_name_or_path=self.query_model_name_or_path,
            ctx_model_name_or_path=self.ctx_model_name_or_path,
            use_dpr=self.use_dpr,
            use_roberta=self.use_roberta,
        )

        log_file_name = os.path.join(
            "output/model_logs",
            f"training_graph_{self.language}_{self.encoding_type}_{self.recall_save}_{self.current_date}.log",
        )
        self.setup_logging(log_file_name)
        print("Initialization completed.")
    def setup_logging(self, log_file_name: str):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file_name)
        logging.info(f"Logging setup completed for {log_file_name}")
        
    def _load_all_input_from_dir(self, input_data_path):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(input_data_path):
            for filename in filenames:
                if "json" in filename:
                    files.append(os.path.join(dirpath, filename))
        total_datapoints = []
        for file in files:
            print(file)
            individual_datapoints = self.input_loader.load_data(data_file=file)
            
            total_datapoints.extend(individual_datapoints) # type: ignore
        
        
        if self.use_translations:
            languages = ["english", "french", "italian", "romanian", "russian", "turkish", "ukrainian"]
            whole_translations = []
            for language in languages:
                translatons_query = self.input_loader.load_data(data_file=f"output/translation_outputs/query_translations_{language}.json")
                whole_translations.extend(translatons_query) # type: ignore
            for point in total_datapoints:
                final_query = []
                for query_item in point["query"]:
                    for trdata_point in whole_translations: # type: ignore
                        if trdata_point["original"] == query_item:
                            final_query.append(trdata_point["translation"])
                            break
                point["query"] = final_query
        return total_datapoints
    
    def _format_input(self, training_datapoints):
        training_data_points = []
        
        if self.language != 'english' and self.use_translations:
            languages = ["english", "french", "italian", "romanian", "russian", "turkish", "ukrainian"]
            whole_translations = []
            for language in languages:
                translatons_query = self.input_loader.load_data(data_file=f"output/translation_outputs/query_translations_{language}.json")
                whole_translations.extend(translatons_query) # type: ignore
        for idx, data in enumerate(training_datapoints):
            key = data["link"]
            final_query = []
            if self.use_translations and self.language != 'english':
                for query_item in data["query"]:
                    for trdata_point in whole_translations: # type: ignore
                        if trdata_point["original"] == query_item:
                            final_query.append(trdata_point["translation"])
            else:
                final_query = data["query"]
            
            unique_key = f"datapoint_{key}"
            query = ", ".join(final_query)
            training_data_points.append(
                {
                    "query": query,
                    "all_paragraphs": ["\n".join(paras) for paras in data["all_paragraphs"]],
                    "unique_keys": [f"{unique_key}_para_{i+1}" for i in range(len(data["all_paragraphs"]))],
                    "paragraph_numbers": data.get("paragraph_numbers", []),
                    "key": f"{query}_{key}",
                    "length_of_all_paragraphs": len(data["all_paragraphs"])
                }
            )
        return training_data_points

    def _encode_all_paragraphs(self, training_datapoints, batch_size=256):
        index_counter = 0
        total_paragraphs = sum(len(points["all_paragraphs"]) for points in training_datapoints)
        with tqdm(total=total_paragraphs, desc="Encoding paragraphs", unit="paragraph") as pbar:
            for idx, points in enumerate(training_datapoints):
                    all_paragraphs = points["all_paragraphs"]
                    num_paragraphs = len(all_paragraphs)
                    encoded_paragraphs = []

                    for start_idx in range(0, num_paragraphs, batch_size):
                        end_idx = min(start_idx + batch_size, num_paragraphs)
                        batch_paragraphs = all_paragraphs[start_idx:end_idx]
                        
                        with torch.no_grad():
                            encoded_batch = self.encoder.encode_ctx(batch_paragraphs).cpu().numpy()
                        encoded_paragraphs.append(encoded_batch)
                        pbar.update(end_idx - start_idx)

                    encoded_paragraphs = np.vstack(encoded_paragraphs)
                    encoded_paragraphs = torch.from_numpy(encoded_paragraphs)
                    with torch.no_grad():
                        encoded_query = self.encoder.encode_question([points["query"]]).squeeze().detach().cpu()
                    points["encoded_query"] = encoded_query
                    encoded_paragraphs = encoded_paragraphs.detach().cpu()
                    points["encoded_paragraphs"] = encoded_paragraphs
                    index_counter += len(all_paragraphs)
        
        return training_datapoints
    
    def _obtain_pre_graph_datapoints(self, training_datapoints):
        
        final_training_datapoints = []
        print("Building Graph...")
        for item in tqdm(training_datapoints):
            query = item["query"]
            all_paragraphs = item["all_paragraphs"]
            encoded_paragraphs = item["encoded_paragraphs"].detach().cpu()
            
            if "encoded_query" not in item.keys():
                with torch.no_grad():
                    encoded_query = self.encoder.encode_question([item["query"]]).squeeze().detach().cpu()
            else:
                encoded_query = item["encoded_query"]
            graph = self._build_graph(encoded_paragraphs, all_paragraphs)
            final_training_datapoints.append(
                {
                    "query": query,
                    "paragraph_numbers": item.get("paragraph_numbers", []),
                    "key": item["key"],
                    "encoded_paragraphs": encoded_paragraphs,
                    "encoded_query": item.get("encoded_query", encoded_query).detach().cpu(),
                    "all_paragraphs": all_paragraphs,
                    "graph": graph,
                }
            )
        return final_training_datapoints
    
    def _build_single_datapoint(self, final_training_datapoints):
        batches = []
        
        for i in range(0, len(final_training_datapoints), self.batch_size):
            batch = final_training_datapoints[i:i + self.batch_size]
            batches.append(batch)
        
        return batches

    def _bm25_nodes(self, i, paragraphs):
        tokenized_corpus = [p.split() for p in paragraphs]
        bm25 = BM25Okapi(tokenized_corpus)
        
        query = paragraphs[i].split()
        scores = bm25.get_scores(query)
        
        indices_and_scores = [(idx, score) for idx, score in enumerate(scores) if idx != i]
        indices_and_scores.sort(key=lambda x: x[1], reverse=True)
        
        min_range = min(10, len(paragraphs))
        top_5 = [idx for idx, _ in indices_and_scores[:min_range]]
        
        return top_5
    
    
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
            prob, topic_embeddings = self.topic_model.obtain_topic_embeddings(paragraphs=paragraphs)
            topics_tensor = torch.tensor(topic_embeddings, dtype=paragraph_encodings.dtype, device=paragraph_encodings.device)
            node_features = torch.cat((paragraph_encodings, topics_tensor), dim=0)
            for i, topic in enumerate(topic_embeddings):
                for j, paragraph in enumerate(paragraph_encodings):
                    if self.use_threshold:
                        if prob[j][i] > 0.25: 
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
        Convert a PyTorch Geometric Data object to a NetworkX graph and visualize it with better separation.
        """
        import matplotlib.pyplot as plt # type: ignore
        import networkx as nx # type: ignore
        from torch_geometric.utils import to_networkx # type: ignore

        G = to_networkx(data, to_undirected=True)
        
        pos = nx.spring_layout(G, seed=42)
        
        plt.figure(figsize=(12, 12))
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
        plt.title("Graph Visualization", fontsize=15)
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.show()

        
    def _get_new_encodings_with_graphs(self, points, gnn_model):
        batched_graphs = []
        datapoint_map = []

        batch_size = len(points)
        for idx, datapoint in enumerate(points):
            graph = datapoint["graph"]
            batched_graphs.append(graph)
            
            datapoint_map.append(idx)
            if len(batched_graphs) == batch_size or idx == len(points) - 1:
                batch = Batch.from_data_list(batched_graphs).to(self.device)
                batched_graphs = []
                
                updated_batch = gnn_model(batch)
                start_idx = 0
                for graph_idx, original_graph in enumerate(batch.to_data_list()):
                    num_nodes = original_graph.x.shape[0]
                    end_idx = start_idx + num_nodes
                    updated_encodings = updated_batch.x[start_idx:end_idx]
                    num_paragraphs = original_graph.num_paragraphs
                    paragraph_encodings = updated_encodings[:num_paragraphs]
                    start_idx = end_idx

                    datapoint_index = datapoint_map[graph_idx]
                    points[datapoint_index]["updated_paragraph_encodings"] = paragraph_encodings
                    
                    del updated_encodings
                    torch.cuda.empty_cache()
                datapoint_map = []
        return points
    
    def _create_individual_datapoints(self, batch):
        
        
        final_datapoints = []
        
        for item in batch:
            pos = []
            neg = []
            query_embeddings = item["encoded_query"]
            for idx, para in enumerate(item["updated_paragraph_encodings"]):
                if (idx + 1) in item["paragraph_numbers"]:
                    pos.append(para)
                else:
                    neg.append(para)
            if len(neg) < 7:
                neg = neg * (7 // len(neg)) + random.sample(neg, 7 % len(neg))
                
            for para  in pos:
                # print("para", para.shape)
                final_datapoints.append(
                    {
                        "query": query_embeddings,
                        "pos": para,
                        "neg": random.sample(neg, 7)
                    }
                )
        return final_datapoints
    
    def save_model(self, model, topic_model, path, epoch = None):
        """Save the model state dictionary to a file.
        
        Args:
            model: The PyTorch model to save.
            epoch: If provided, this is used to create a filename specific to that epoch.
        """
        

        if epoch is not None:
            graph_path = os.path.join(path, f"epoch_{epoch}", "graph_model.pt")
            # topic_path = os.path.join(path, f"epoch_{epoch}", "topic_model") 
        else:
            graph_path = os.path.join(path, "graph_model.pt")
            # topic_path = os.path.join(path, "topic_model")
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        # os.makedirs(os.path.dirname(topic_path), exist_ok=True)
        torch.save(model.state_dict(), graph_path)
        # topic_model.save(topic_path)
        # self.topic_model.embedding_model.save(topic_path)
        print(f"Model saved to {path}")
        
    def train(self, use_saved_data=False, processed_data_path="/srv/upadro/embeddings"):
        use_topics = self.use_topics
        use_cosine = self.use_cosine
        use_prev_next_two = self.use_prev_next_two
        
        print(self.use_topics, self.use_cosine, self.use_prev_next_two)
        path = os.path.join(processed_data_path, self.language, f"processed_encoded_training_data_{self.save_model_name}.pkl")
        
        if os.path.exists(path):
            use_saved_data = True
        print(use_saved_data)
        if use_saved_data:
            self.training_datapoints = self._load_processed_data(processed_data_path, file_name=f"processed_encoded_training_data_{self.save_model_name}.pkl")
        
        else:
            self.training_datapoints = self._load_all_input_from_dir(self.train_data_folder)
            self.training_datapoints = self._format_input(self.training_datapoints)
            self.training_datapoints = self._encode_all_paragraphs(training_datapoints=self.training_datapoints)
            self._save_processed_data(self.training_datapoints, save_path=processed_data_path, file_name=f"processed_encoded_training_data_{self.save_model_name}.pkl")
        
        
        
        self.use_topics, self.use_cosine, self.use_prev_next_two = use_topics, use_cosine, use_prev_next_two
        print(self.use_topics, self.use_cosine, self.use_prev_next_two)
        
        
        self.final_training_datapoints = self._obtain_pre_graph_datapoints(self.training_datapoints)
        random.shuffle(self.final_training_datapoints)
        
        self.batches = self._build_single_datapoint(self.final_training_datapoints,)
        del self.final_training_datapoints
        del self.training_datapoints
        torch.cuda.empty_cache()
        num_batches = len(self.batches)
        self._remove_encoder()
        hidden_dim = self.batches[0][0]["encoded_paragraphs"].shape[1]
        graph = self.batches[0][0]["graph"]
        self.visualize_graph(data=graph)
        gnn_model = ParagraphGNN(hidden_dim=hidden_dim, num_layers=3) if self.graph_model == "gcn" else ParagraphGAT(hidden_dim=hidden_dim)
        gnn_model.to(self.device)
        
        optimizer = optim.AdamW(gnn_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in tqdm(range(self.epochs)):
            random.shuffle(self.batches)
            gnn_model.train()
            total_loss = 0.0
            
            with tqdm(total=num_batches, desc=f"Training Epoch {epoch+1}", unit="batch") as progress_bar:
                for i, batch in enumerate(self.batches):
                    batch = self._get_new_encodings_with_graphs(points=batch, gnn_model=gnn_model)
                    batch = self._create_individual_datapoints(batch=batch)
                    
                    
                    query_tensors = []
                    positive_tensors = []
                    negative_tensors = []
                    
                    for item in batch:
                        query_tensors.append(item["query"].unsqueeze(0))
                        positive_tensors.append(item["pos"].unsqueeze(0))
                        negative_tensors.extend([neg.unsqueeze(0) for neg in item["neg"]])
                    
                    query_tensor = torch.cat(query_tensors, dim=0).to(self.device)
                    positive_tensor = torch.cat(positive_tensors, dim=0).to(self.device)
                    negatives_tensor = torch.stack(negative_tensors).view(len(batch), -1, query_tensors[0].shape[-1]).to(self.device)
                    all_paragraphs = torch.cat([positive_tensor.unsqueeze(1), negatives_tensor], dim=1)
                    
                    similarity_scores = torch.nn.functional.cosine_similarity(
                        query_tensor.unsqueeze(1), 
                        all_paragraphs, 
                        dim=-1
                    )
                    
                    logits = similarity_scores
                    labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
                    optimizer.zero_grad()
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    
                    loss_val = loss.item()
                    total_loss += loss_val
                    avg_loss = total_loss / (i + 1)
                    
                    progress_bar.set_postfix(loss=avg_loss)
                    progress_bar.update(1)
            model_save_dir = f"/srv/upadro/models/graph/new_gat/{self.current_date}___{self.language}_{self.graph_model}_{self.comments}_training/checkpoints"
            topic_model = self.topic_model.embedding_model
            self.save_model(gnn_model, topic_model=topic_model, path = model_save_dir, epoch=epoch)
            average_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{self.epochs}, Average Loss: {average_loss:.4f}")
        model_save_dir = f"/srv/upadro/models/graph/new_gat/{self.current_date}___{self.language}_{self.graph_model}_{self.comments}_training/_final_model"
        topic_model = self.topic_model.embedding_model
        self.save_model(gnn_model, topic_model=topic_model, path = model_save_dir)
        print("Training complete.")
    
    def _save_processed_data(self, data, save_path="/srv/upadro/embeddings", file_name="processed_training_data.pkl"):
        save_path = os.path.join(save_path, self.language, file_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Processed training data saved to {save_path}")
        
    def _load_processed_data(self, save_path="/srv/upadro/embeddings", file_name="processed_training_data.pkl"):
        save_path = os.path.join(save_path, self.language, file_name)
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"No processed training data found at {save_path}")
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        print(f"Processed training data loaded from {save_path}")
        return data

    def _remove_encoder(self,):
        del self.encoder
        torch.cuda.empty_cache()
        print(f"model removed")
        
if __name__ == "__main__":
    # languages = ["russian", "english", "french", "italian", "romanian", "turkish", "ukrainian"]
    # languages = ["russian", "french", "italian", "romanian", "turkish", "ukrainian"]
    languages = ["all"]
    for language in languages:
        # dual_encoders = [False, True]
        dual_encoders = [True]
        models = [
            # [
            #     "castorini/mdpr-tied-pft-msmarco",
            #     "castorini/mdpr-tied-pft-msmarco",
            #     "sentence_transformer_threshold_no_weight_0_shot_use_topics_use_prev_next_two",
            #     {
            #         "use_topics": True,
            #         "use_cosine": False,
            #         "use_prev_next_two": True
            #     }
            # ],
            # [
            #     "castorini/mdpr-tied-pft-msmarco",
            #     "castorini/mdpr-tied-pft-msmarco",
            #     "no_weight_0_shot_use_cosine_use_prev_next_two",
            #     {
            #         "use_topics": False,
            #         "use_cosine": True,
            #         "use_prev_next_two": True
            #     }
            # ],
            [
                "castorini/mdpr-tied-pft-msmarco",
                "castorini/mdpr-tied-pft-msmarco",
                "sentence_transformer_threshold_no_weight_0_shot_use_topic_use_cosine_use_prev_next_two",
                {
                    "use_topics": True,
                    "use_cosine": True,
                    "use_prev_next_two": True
                }
            ],
            
        ]
        train_data_folder = f'input/train_infer/{language}/new_split/train_test_val'
        val_data_folder = f'input/train_infer/{language}/new_split/val'
        
        print(train_data_folder)
        for dual_encoder in dual_encoders:
            for idx, model in enumerate(models):
                # try:
                    if model[0] != 'bert-base-uncased' and "all" in language:
                        # translations = [True, False]
                        translations = [False]
                    else:
                        # translations = [True]
                        translations = [False]
                    for use_translation in translations:
                        # try:
                            config_file = "src/models/vector_db/commons/config.yaml"
                            language = train_data_folder.split('/')[-3] 
                            trainer = GraphTrainer(
                                use_dpr=False,
                                use_roberta=False,
                                train_data_folder=train_data_folder,
                                val_data_folder=val_data_folder,
                                config_file=config_file,
                                device_str='cuda:2',
                                dual_encoders=dual_encoder,
                                language=language,
                                batch_size=4,
                                epochs=40,
                                lr=2e-5, #2e-5 or 1e-5 TODO
                                save_checkpoints=True,
                                step_validation=False,
                                query_model_name_or_path=model[0],
                                ctx_model_name_or_path=model[1],
                                use_translations=use_translation,
                                graph_model="gat",
                                comments = model[2],
                                use_threshold = True,
                                # use_bm25=True, 
                                # use_all=True,
                                
                                use_prev_next_two=model[3]["use_prev_next_two"],
                                use_cosine=model[3]["use_cosine"],
                                use_topics=model[3]["use_topics"],
                            )
                            trainer.train()
        print("done")