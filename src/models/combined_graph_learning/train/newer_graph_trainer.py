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

from sentence_transformers import SentenceTransformer # type: ignore
from src.models.single_datapoints.common.utils import current_date
from src.models.combined_graph_learning.encoders.paragraph_gat import ParagraphGAT
from src.models.combined_graph_learning.encoders.paragraph_gat_mixing import ParagraphGATAllMixing
from src.models.combined_graph_learning.encoders.paragraph_gat_gated import ParagraphGATGated
from src.models.combined_graph_learning.encoders.paragraph_gat_gated_all import ParagraphGATAllGating
from src.models.combined_graph_learning.encoders.graph_creation import GraphCreator
from src.models.graph_learning.encoders.graph_encoder import GraphEncoder as Encoder
# from src.models.graph_learning.train.newer_topic_modeling import TopicModeling
from src.models.graph_learning.train.new_topic_modeling import TopicModeling


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
        gat_type:str = "normal",
    ):
        self.use_dpr = use_dpr
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
        self.comments = comments
        self.gat_type = gat_type
        if self.config_file == "":
            raise ValueError("config file empty, please contact admin")
        self.topic_model = TopicModeling()

        self.print = True
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
        self.graph_creator = GraphCreator()

        print("Initialization completed.")
    def _build_graph(self, paragraph_encodings, paragraphs):
        graph1 = self.graph_creator.build_graph(
            paragraph_encodings=paragraph_encodings,
            paragraphs=paragraphs,
            topic_model=None,
            use_prev_next_five=True,
        )
        graph2 = self.graph_creator.build_graph(
            paragraph_encodings=paragraph_encodings,
            paragraphs=paragraphs,
            topic_model=self.topic_model,
            use_prev_next_five=False,
            use_topics=True,
            use_threshold=True,
        )
        
        return (graph1, graph2)
    
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
    
    def _get_new_encodings_with_graphs(self, points, gnn_model):
        """
        Processes points with a ParagraphGAT model for paired graphs and updates paragraph encodings.

        Args:
            points: List of datapoints, each containing two graphs and their data.
            gnn_model: ParagraphGAT model to process paired graphs.

        Returns:
            Updated points with new paragraph encodings.
        """
        batched_graphs_1 = []
        batched_graphs_2 = []
        datapoint_map = []

        batch_size = len(points)
        for idx, datapoint in enumerate(points):
            graph1, graph2 = datapoint["graph"]
            batched_graphs_1.append(graph1)
            batched_graphs_2.append(graph2)
            datapoint_map.append(idx)

            if len(batched_graphs_1) == batch_size or idx == len(points) - 1:
                batch_1 = Batch.from_data_list(batched_graphs_1).to(self.device)
                batch_2 = Batch.from_data_list(batched_graphs_2).to(self.device)

                updated_batch = gnn_model((batch_1, batch_2))
                batched_graphs_1 = []
                batched_graphs_2 = []

                start_idx = 0
                for graph_idx, (original_graph_1, original_graph_2) in enumerate(zip(batch_1.to_data_list(), batch_2.to_data_list())):
                    num_paragraphs_1 = original_graph_1.num_paragraphs
                    num_paragraphs_2 = original_graph_2.num_paragraphs

                    num_nodes_1 = original_graph_1.x.shape[0]
                    num_nodes_2 = original_graph_2.x.shape[0]

                    end_idx = start_idx + num_nodes_1
                    updated_encodings = updated_batch[start_idx:end_idx]
                    paragraph_encodings = updated_encodings[:num_paragraphs_1]
                    start_idx = end_idx

                    datapoint_index = datapoint_map[graph_idx]
                    points[datapoint_index]["updated_paragraph_encodings"] = paragraph_encodings

                    del updated_encodings, paragraph_encodings
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
    
    def train(self, processed_data_path="/srv/upadro/embeddings"):
        self.training_datapoints = self._load_processed_data(processed_data_path, file_name=f"processed_encoded_training_data_{self.save_model_name}.pkl")
        self.training_datapoints = self.training_datapoints
        self.final_training_datapoints = self._obtain_pre_graph_datapoints(self.training_datapoints)
        random.shuffle(self.final_training_datapoints)
        
        self.batches = self._build_single_datapoint(self.final_training_datapoints,)
        del self.final_training_datapoints
        del self.training_datapoints
        torch.cuda.empty_cache()
        num_batches = len(self.batches)
        hidden_dim = self.batches[0][0]["encoded_paragraphs"].shape[1]
        graph = self.batches[0][0]["graph"]
        self.visualize_graph(data=graph[0], file_name="src/models/combined_graph_learning/train/graph_1.png")
        self.visualize_graph(data=graph[1], file_name="src/models/combined_graph_learning/train/graph_2.png")
        
        if self.gat_type == "all_mixing":
            gnn_model = ParagraphGATAllMixing(hidden_dim=hidden_dim)
        elif self.gat_type == "last_gated":
            gnn_model = ParagraphGATGated(hidden_dim=hidden_dim)
        elif self.gat_type == 'all_gated':
            gnn_model = ParagraphGATAllGating(hidden_dim=hidden_dim)
        else:
            gnn_model = ParagraphGAT(hidden_dim=hidden_dim)
        
        gnn_model.to(self.device)
        print(gnn_model)
        optimizer = optim.AdamW(gnn_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        print("Number of unusable datapoints", self.topic_model.count)
        
        for epoch in tqdm(range(self.epochs)):
            random.shuffle(self.batches)
            gnn_model.train()
            total_loss = 0.0
            gnn_model.unusable = 0
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
            model_save_dir = f"/srv/upadro/models/new_expt/ensemble/language/{self.current_date}___{self.language}_{self.graph_model}_{self.comments}_training/checkpoints"
            self.save_model(gnn_model, path = model_save_dir, epoch=epoch)
            average_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{self.epochs}, Average Loss: {average_loss:.4f}")
            print(f"unusable datapoints for epoch : {gnn_model.unusable}")
        
        model_save_dir = f"/srv/upadro/models/new_expt/ensemble/language/{self.current_date}___{self.language}_{self.graph_model}_{self.comments}_training/_final_model"
        self.save_model(gnn_model, path = model_save_dir)
        print("Training complete.")
        pass
    
    def _load_processed_data(self, save_path="/srv/upadro/embeddings", file_name="processed_training_data.pkl"):
        save_path = os.path.join(save_path, self.language, file_name)
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"No processed training data found at {save_path}")
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        print(f"Processed training data loaded from {save_path}")
        return data

    def save_model(self, model, path, epoch = None):
        """Save the model state dictionary to a file.
        
        Args:
            model: The PyTorch model to save.
            epoch: If provided, this is used to create a filename specific to that epoch.
        """
        

        if epoch is not None:
            graph_path = os.path.join(path, f"epoch_{epoch}", "graph_model.pt")
            topic_path = os.path.join(path, f"epoch_{epoch}", "topic_model") 
        else:
            graph_path = os.path.join(path, "graph_model.pt")
            topic_path = os.path.join(path, "topic_model")
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        os.makedirs(os.path.dirname(topic_path), exist_ok=True)
        torch.save(model.state_dict(), graph_path)
        print(f"Model saved to {path}")

if __name__ == "__main__":
    languages = ["russian", "ukrainian", "romanian", "french", "turkish", "italian", "english"]
    # languages = ["russian", "english", "french", "italian"]
    # languages = ["romanian", "turkish", "ukrainian"]
    # languages = ["russian", "french", "italian", "romanian", "turkish", "ukrainian"]
    # languages = ["all"]
    for language in languages:
        # dual_encoders = [False, True]
        dual_encoders = [True]
        models = [
            [
                "castorini/mdpr-tied-pft-msmarco",
                "castorini/mdpr-tied-pft-msmarco",
                "new_gat_ablation_last_mixing_next_5_topic_threshold",
                "normal"
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
                                device_str='cuda:3',
                                dual_encoders=dual_encoder,
                                language=language,
                                batch_size=1,
                                epochs=40,
                                lr=2e-5, #2e-5 or 1e-5 TODO
                                save_checkpoints=True,
                                step_validation=False,
                                query_model_name_or_path=model[0],
                                ctx_model_name_or_path=model[1],
                                use_translations=use_translation,
                                graph_model="gat",
                                comments = model[2],
                                gat_type= model[3]
                            )
                            trainer.train()
        print("done")