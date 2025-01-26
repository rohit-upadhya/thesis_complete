from typing import Text
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

from src.models.single_datapoints.common.utils import current_date
from src.models.single_datapoints.common.data_loader import InputLoader
from src.models.attention.encoders.attention_encoder import AttentionEncoder as Encoder
from src.models.attention.encoders.attention_block import AttentionBlock

class AttentionTrainer:
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
        comments=None,
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
        self.save_model_name = self.query_model_name_or_path.replace('/','_')
        self.current_date = current_date()
        self.input_loader = InputLoader()
        self.comments = comments
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
            f"training_attention_{self.language}_{self.encoding_type}_{self.recall_save}_{self.current_date}.log",
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
    
    def _build_single_datapoint(self, final_training_datapoints):
        batches = []
        
        for i in range(0, len(final_training_datapoints), self.batch_size):
            batch = final_training_datapoints[i:i + self.batch_size]
            batches.append(batch)
        
        return batches
    
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
            random.shuffle(neg)
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
    
    def save_model(self, model, path, epoch = None):
        if epoch is not None:
            attention_path = os.path.join(path, f"epoch_{epoch}", "attention_model.pt")
        else:
            attention_path = os.path.join(path, "attention_model.pt")
        os.makedirs(os.path.dirname(attention_path), exist_ok=True)
        torch.save(model.state_dict(), attention_path)
        print(f"Model saved to {path}")
    
    def _get_new_encodings_with_attention_block(self, batch, attention_model):
        max_paragraphs = max(item["encoded_paragraphs"].shape[0] for item in batch)

        batch_tensors = []
        lengths = []

        for item in batch:
            parag = item["encoded_paragraphs"]
            num_parag = parag.size(0)
            lengths.append(num_parag)

            if num_parag < max_paragraphs:
                pad_size = max_paragraphs - num_parag
                pad_zeros = parag.new_zeros((pad_size, parag.shape[1]))
                parag = torch.cat([parag, pad_zeros], dim=0)

            batch_tensors.append(parag.unsqueeze(0))

        paragraphs_batch = torch.cat(batch_tensors, dim=0).to(self.device)

        updated_embeddings, attn_weights1 = attention_model(paragraphs_batch)

        for i, item in enumerate(batch):
            length = lengths[i]
            item["updated_paragraph_encodings"] = updated_embeddings[i, :length]

        return batch

    def train(self, use_saved_data=False, processed_data_path="/srv/upadro/embeddings"):
        
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
        
        
        random.shuffle(self.training_datapoints)
        
        self.batches = self._build_single_datapoint(self.training_datapoints,)
        del self.training_datapoints
        torch.cuda.empty_cache()
        num_batches = len(self.batches)
        self._remove_encoder()
        hidden_dim = self.batches[0][0]["encoded_paragraphs"].shape[1]
        # graph = self.batches[0][0]["graph"]
        # self.visualize_graph(data=graph)
        attention_model = AttentionBlock(embedding_dim=hidden_dim)
        attention_model.to(self.device)
        print(attention_model)
        optimizer = optim.AdamW(attention_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in tqdm(range(self.epochs)):
            random.shuffle(self.batches)
            attention_model.train()
            total_loss = 0.0
            
            with tqdm(total=num_batches, desc=f"Training Epoch {epoch+1}", unit="batch") as progress_bar:
                for i, batch in enumerate(self.batches):
                    batch = self._get_new_encodings_with_attention_block(batch=batch, attention_model=attention_model)
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
            model_save_dir = f"/srv/upadro/models/new_expt/attention/{self.current_date}___{self.language}_{self.comments}_training/checkpoints"
            self.save_model(attention_model, path = model_save_dir, epoch=epoch)
            average_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{self.epochs}, Average Loss: {average_loss:.4f}")
        model_save_dir = f"/srv/upadro/models/new_expt/attention/{self.current_date}___{self.language}_{self.comments}_training/_final_model"
        self.save_model(attention_model, path = model_save_dir)
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
            [
                "castorini/mdpr-tied-pft-msmarco",
                "castorini/mdpr-tied-pft-msmarco",
                "new_attention_expts_pos_just_1_ff_base",
            ],
            [
                "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/query_model",
                "/srv/upadro/models/all/dual/2024-10-28__dual__all__not_translated__castorini_mdpr-tied-pft-msmarco_training/_final_model/ctx_model",
                "new_attention_expts_pos_just_1_ff_ours",
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
                            trainer = AttentionTrainer(
                                use_dpr=False,
                                use_roberta=False,
                                train_data_folder=train_data_folder,
                                val_data_folder=val_data_folder,
                                config_file=config_file,
                                device_str='cuda:0',
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
                                comments = model[2],
                            )
                            trainer.train()
        print("done")