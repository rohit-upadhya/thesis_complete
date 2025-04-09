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

from src.models.single_datapoints.common.utils import current_date
from src.models.single_datapoints.common.data_loader import InputLoader
# from src.models.graph_learning.encoders.new_paragraph_gcn import ParagraphGNN
from src.models.graph_learning.encoders.paragraph_gcn import ParagraphGNN
from src.models.graph_learning.encoders.paragraph_gat import ParagraphGAT
from src.models.graph_learning.encoders.graph_encoder import GraphEncoder as Encoder
# torch.autograd.set_detect_anomaly(True)

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
        graph_model: Text = "gcn"
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
        self.input_loader = InputLoader()

        if self.config_file == "":
            raise ValueError("config file empty, please contact admin")

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
        print(training_data_points[0]["query"])
        return training_data_points

    def _encode_all_paragraphs(self, training_datapoints, batch_size=512):
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
                    points["encoded_paragraphs"] = encoded_paragraphs.detach().cpu()
                    index_counter += len(all_paragraphs)
        
        return training_datapoints
    
    def _obtain_pre_graph_datapoints(self, training_datapoints):
        
        final_training_datapoints = []
        key_set = set()
        for item in training_datapoints:
            query = item["query"]
            for idx, paras in enumerate(item["all_paragraphs"]):
                final_training_datapoints.append(
                    {
                        "para_no": idx+1,
                        "para": paras,
                        "query": query,
                        "all_paragraphs": item["all_paragraphs"],
                        "paragraph_numbers": item.get("paragraph_numbers", []),
                        "key": item["key"],
                        "positive": True if (idx+1) in item.get("paragraph_numbers", []) else False,
                        "encoded_paragraphs": item["encoded_paragraphs"].detach().cpu(),
                        "encoded_query": item["encoded_query"].detach().cpu()
                    }
                )
            key_set.add(item["key"])
        return final_training_datapoints, list(key_set)

    # def _build_single_datapoint(self,training_datapoints, key_list):
    #     temp_batches = []
    #     for key in key_list:
    #         positive = []
    #         negative = []
    #         for item in training_datapoints:
    #             if item["key"] == key:
    #                 if item["positive"]:
    #                     positive.append(item)
    #                 else:
    #                     negative.append(item)
            
    #         # print(len(positive))
    #         for item in positive:
    #             batch = []
    #             batch.append(item)
    #             batch.extend(random.sample(negative, 7))
    #             temp_batches.append(batch)
        
    #     batches = []
        
    #     return batches
    def _build_single_datapoint(self, training_datapoints, key_list):
        temp_batches = []
        for key in key_list:
            positive = []
            negative = []
            for item in training_datapoints:
                if item["key"] == key:
                    if item["positive"]:
                        positive.append(item)
                    else:
                        negative.append(item)

            for item in positive:
                batch = []
                batch.append(item)
                batch.extend(random.sample(negative, 7)) 
                temp_batches.append(batch)

        final_batches = [temp_batches[i:i + self.batch_size] for i in range(0, len(temp_batches), self.batch_size)]

        return final_batches
    
    # def _build_graph(self, paragraph_encodings):
        
    #     num_paragraphs = len(paragraph_encodings)
    #     edge_index = []
        
    #     for i in range(num_paragraphs):
    #         if i > 0:
    #             edge_index.append([i, i-1])
    #         if i > 1:
    #             edge_index.append([i, i-2])
    #         if i < num_paragraphs - 1:
    #             edge_index.append([i, i+1])
    #         if i < num_paragraphs - 2:
    #             edge_index.append([i, i+2])
        
    #     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    #     node_features = paragraph_encodings.clone().detach()
    #     edge_index = edge_index.clone().detach()
    #     data = Data(x=node_features, edge_index=edge_index)
    #     return data
    
    def _build_graph(self, paragraph_encodings):
        num_paragraphs = len(paragraph_encodings)
        edge_index = []
        
        for i in range(num_paragraphs):
            if i > 0:
                edge_index.append([i, i-1])
            if i > 1:
                edge_index.append([i, i-2])
            if i < num_paragraphs - 1:
                edge_index.append([i, i+1])
            if i < num_paragraphs - 2:
                edge_index.append([i, i+2])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        node_features = paragraph_encodings
        return Data(x=node_features, edge_index=edge_index)
    
    def _get_new_encodings_with_graphs(self, points, gnn_model):
        """
        Generate updated encodings for all datapoints using GNN, preserving gradients for backpropagation.
        Each datapoint will store updated encodings for all its paragraphs.
        """
        batched_graphs = []
        datapoint_map = []

        flattened_points = [item for sublist in points for item in sublist]
        batch_size = self.batch_size * 8
        # print(batch_size)
        for idx, datapoint in enumerate(flattened_points):
            paragraph_encodings = datapoint["encoded_paragraphs"].to(self.device)
            graph = self._build_graph(paragraph_encodings)
            batched_graphs.append(graph)
            
            datapoint_map.append(idx)
            if len(batched_graphs) == batch_size or idx == len(flattened_points) - 1:
                batch = Batch.from_data_list(batched_graphs).to(self.device)
                batched_graphs = []
                # num_nodes = batch.x.size(0)
                # num_edges = batch.edge_index.size(1)
                # print(f"Batched graph: {num_nodes} nodes, {num_edges} edges")
                updated_batch = gnn_model(batch)

                start_idx = 0
                for graph_idx, original_graph in enumerate(batch.to_data_list()):
                    num_nodes = original_graph.x.shape[0]
                    end_idx = start_idx + num_nodes

                    updated_encodings = updated_batch.x[start_idx:end_idx]

                    start_idx = end_idx

                    datapoint_index = datapoint_map[graph_idx]
                    flattened_points[datapoint_index]["updated_paragraph_encodings"] = updated_encodings
                    del updated_encodings
                    # gc.collect()
                    torch.cuda.empty_cache()
                del updated_batch
                gc.collect()
                torch.cuda.empty_cache()
                datapoint_map = []

        reshaped_points = [flattened_points[i:i + 8] for i in range(0, len(flattened_points), 8)]

        return reshaped_points

    
    def _create_individual_datapoints(self, batch):
        batch_points = []
        
        for datapoint_set in batch:
            pos = datapoint_set[0]
            query = pos["encoded_query"]
            
            neg = datapoint_set[1:]
            negative_encodings = []
            
            for item in neg:
                negative_encodings.append(item["updated_paragraph_encodings"][item["para_no"] - 1])
            
            batch_points.append(
                {
                    "query": query, 
                    "positive": pos["updated_paragraph_encodings"][pos["para_no"] - 1],
                    "negatives": negative_encodings
                }
            )
        
        return batch_points

    
    
    def save_model(self, model, path, epoch = None):
        """Save the model state dictionary to a file.
        
        Args:
            model: The PyTorch model to save.
            epoch: If provided, this is used to create a filename specific to that epoch.
        """
        

        if epoch is not None:
            path = os.path.join(path, f"epoch_{epoch}", "graph_model.pt")
        else:
            path = os.path.join(path, "graph_model.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")
        
    def train(self, use_saved_data=False, use_saved_batch=False, processed_data_path="/srv/upadro/embeddings"):
        batch_path = os.path.join(processed_data_path, self.language, "batch_data.pkl")
        if os.path.exists(batch_path):
            use_saved_batch = True
        if use_saved_batch:
            self.batches = self._load_processed_data(save_path=processed_data_path, file_name="batch_data.pkl")
        else:
            path = os.path.join(processed_data_path, self.language, "processed_training_data.pkl")
            if os.path.exists(path):
                use_saved_data = True
            if use_saved_data:
                self.training_datapoints = self._load_processed_data(processed_data_path)
                # self.batches = self._load_processed_data(save_path=processed_data_path, file_name="batch_data.pkl")
            else:
                self.training_datapoints = self._load_all_input_from_dir(self.train_data_folder)
                self.training_datapoints = self._format_input(self.training_datapoints)
                self.training_datapoints = self._encode_all_paragraphs(training_datapoints=self.training_datapoints)
                self._save_processed_data(self.training_datapoints, save_path=processed_data_path)
                # self._save_processed_data(data=self.batches,save_path=processed_data_path, file_name="batch_data.pkl")
                # self.final_training_datapoints, self.key_list = self._obtain_pre_graph_datapoints(self.training_datapoints)
                # random.shuffle(self.final_training_datapoints)
                # self.batches = self._build_single_datapoint(self.final_training_datapoints, self.key_list)
                # random.shuffle(self.batches)
            self.final_training_datapoints, self.key_list = self._obtain_pre_graph_datapoints(self.training_datapoints)
            random.shuffle(self.final_training_datapoints)
            self.batches = self._build_single_datapoint(self.final_training_datapoints, self.key_list)
            self._save_processed_data(data=self.batches,save_path=processed_data_path, file_name="batch_data.pkl")
        # random.shuffle(self.batches)
        # num_batches = int(len(self.batches)/2)
        # self.batches = self.batches[:num_batches] 
        num_batches = int(len(self.batches))
        
        # with open("src/models/graph_learning/testing.txt", "w+") as file:
        #     file.write(f'{self.training_datapoints[0]["encoded_paragraphs"]}')
        # print( self.batches[0][0][0]["encoded_paragraphs"].shape)
        hidden_dim = self.batches[0][0][0]["encoded_paragraphs"].shape[1]
        
        
        gnn_model = ParagraphGNN(hidden_dim=hidden_dim, num_layers=3) if self.graph_model == "gcn" else ParagraphGAT(hidden_dim=hidden_dim)
        gnn_model.to(self.device)
        
        optimizer = optim.AdamW(gnn_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.epochs):
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
                        positive_tensors.append(item["positive"].unsqueeze(0))
                        negative_tensors.extend([neg.unsqueeze(0) for neg in item["negatives"]])
                    
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
                    
                    del query_tensor
                    del positive_tensor
                    del negatives_tensor
                    del all_paragraphs
                    del similarity_scores
                    del logits
                    del labels
                    del batch
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                if epoch%10 == 0:
                    model_save_dir = f"/srv/upadro/models/graph/{self.current_date}__{self.save_model_name}_{self.language}_training/checkpoints"
                    self.save_model(gnn_model, path = model_save_dir, epoch=epoch)
                average_loss = total_loss / num_batches
            # print("*"*20)
            torch.cuda.empty_cache()
            
            print(f"Epoch {epoch + 1}/{self.epochs}, Average Loss: {average_loss:.4f}")
        model_save_dir = f"/srv/upadro/models/graph/{self.current_date}__{self.save_model_name}_training/_final_model"
        self.save_model(gnn_model, path = model_save_dir)
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

    
if __name__ == "__main__":
    # languages = ["russian", "english", "french", "italian", "romanian", "turkish", "ukrainian"]
    # languages = ["russian", "french", "italian", "romanian", "turkish", "ukrainian"]
    languages = ["english"]
    for language in languages:
        # dual_encoders = [False, True]
        dual_encoders = [False]
        models = [
            # ["joelniklaus/legal-xlm-roberta-base", "joelniklaus/legal-xlm-roberta-base"]
            # ['castorini/mdpr-tied-pft-msmarco-ft-all', 'castorini/mdpr-tied-pft-msmarco-ft-all'],
            # ["facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-ctx_encoder-single-nq-base"]
            ['/srv/upadro/models/all/dual/2024-10-29__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco-ft-all_training/_final_model/query_model', '/srv/upadro/models/all/dual/2024-10-29__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco-ft-all_training/_final_model/ctx_model']
            # ['bert-base-multilingual-cased', 'bert-base-multilingual-cased']
        ]
        train_data_folder = f'input/train_infer/{language}/new_split/train_test_val'
        val_data_folder = f'input/train_infer/{language}/new_split/val'
        
        print(train_data_folder)
        # train_data_folder = 'input/test_train/new_split'
        # val_data_folder = 'input/test_val/new_split'
        for dual_encoder in dual_encoders:
            for model in models:
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
                                batch_size=4,
                                epochs=5,
                                lr=1e-6, 
                                save_checkpoints=True,
                                step_validation=False,
                                query_model_name_or_path=model[0],
                                ctx_model_name_or_path=model[1],
                                use_translations=use_translation,
                                graph_model="gat"
                            )
                            trainer.train()
                #         except Exception as e:
                #             with open("output/model_logs/error.txt", "a+") as file:
                #                 file.write(f"error occured in encoder : {'dual' if dual_encoder else 'single'} and {model} and {'trasnlation' if use_translation else 'no translation'} \n")
                # except Exception as e:
                #     with open("output/model_logs/error.txt", "a+") as file:
                #         file.write(f"error occured in encoder : {'dual' if dual_encoder else 'single'} and {model}\n")
        print("done")