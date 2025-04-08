import torch
import logging
from dataclasses import dataclass
from transformers import BertTokenizer, BertModel, BertTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import torch.optim as optim
from sklearn.model_selection import train_test_split
from typing import Text, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import random
import numpy as np
import torch.nn.functional as F

import time 

from src.models.single_datapoints.common.data_loader import InputLoader
from src.models.single_datapoints.multi_models.common.dual_encoder import DualEncoderModel
from src.models.single_datapoints.multi_models.common.single_encoder_new import SingleEncoderModel
from src.models.single_datapoints.multi_models.common.paragraph_dataset import DualEncoderDataset
from src.models.single_datapoints.common.utils import current_date

@dataclass
class ContrastiveTrainer:
    use_dpr: bool = False
    train_data_folder: Optional[Text] = None
    val_data_folder: Optional[Text] = None
    config_file: Optional[Text] = None
    batch_size: int = 6
    val_batch_size: int = 6
    individual_datapoints: int = 8
    epochs: int = 1
    device: str = 'cpu'
    dual_encoders: bool = True
    language: str = 'english'
    lr: float = 2e-5
    save_checkpoints: bool = False
    step_validation: bool = False
    
    def __post_init__(self):
        self.input_loader = InputLoader()
        if self.config_file == "":
            raise ValueError("config file empty, please contact admin")
        self.config = self.input_loader.load_config(self.config_file) # type: ignore
        self.config = self.config["dual_encoder"] if self.dual_encoders else self.config["single_encoder"] # type: ignore
        self.device = torch.device('cuda' if torch.cuda.is_available() and 'cuda' in self.device else 'cpu')
        self.encoding_type = "dual" if self.dual_encoders else "single" 
        log_file_name = os.path.join("output/model_logs",f"training_{self.language}_{self.encoding_type}_{current_date()}.log")
        self.setup_logging(log_file_name)
    
    def setup_logging(self, log_file_name):
        logging.basicConfig(
            filename=log_file_name,
            filemode='w',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging is set up. Log file: {log_file_name}")
    
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
        return total_inference_datapoints
    
    def _load_data(self, data_file):
        data = self._load_all_input_from_dir(data_file)
        # data = self.input_loader.load_data(self.data_file) # type: ignore
        # counts = 0
        examples = []
        for item in data: # type: ignore
            combined_query = ', '.join(item['query'])
            case_name = item['case_name']
            case_link = item['link']
            for i, paragraph in enumerate(item['all_paragraphs']):
                label = 1 if (i + 1) in item['paragraph_numbers'] else 0
                examples.append((combined_query, paragraph, label, case_name, case_link))
            # counts += len(item["paragraph_numbers"])
        
        return examples#, counts
    
    def _load_tokenizer(self, model_name_or_path):
        
        if self.use_dpr:
            return (
                DPRQuestionEncoderTokenizer.from_pretrained(model_name_or_path, use_fast=False)
            )
        return (
            BertTokenizer.from_pretrained(model_name_or_path)
        )
    
    def _load_model(self,):
        if self.dual_encoders:
            query_model = BertModel.from_pretrained(self.config['query_model_name_or_path'])
            paragraph_model = BertModel.from_pretrained(self.config['doc_model_name_or_path'])

            query_model.to(self.device)
            paragraph_model.to(self.device)

            model = DualEncoderModel(query_model, paragraph_model)
            model.to(self.device)
        else:
            single_model = BertModel.from_pretrained(self.config['model_name_or_path'])
            single_model.to(self.device)
            model = SingleEncoderModel(single_model)
            model.to(self.device)

    
        self.logger.info(f"Model initialized: {model}")
        return model
    
    def collate_fn(self, batch):
        """
        Custom collator function to prepare a batch where each item is a list of examples:
        [pos, neg1, neg2, neg3, neg4, neg5, neg6, neg7]
        Each example is a dictionary with keys:
        'query_input_ids', 'query_attention_mask', 'paragraph_input_ids', 'paragraph_attention_mask', 'labels'
        """

        batch_query_input_ids = []
        batch_query_attention_mask = []
        batch_paragraph_input_ids = []
        batch_paragraph_attention_mask = []
        batch_labels = []

        for item in batch:
            query_input_ids = item[0]['query_input_ids']
            query_attention_mask = item[0]['query_attention_mask']

            paragraph_input_ids_list = []
            paragraph_attention_mask_list = []
            labels_list = []

            for example in item:
                paragraph_input_ids_list.append(example['paragraph_input_ids'])
                paragraph_attention_mask_list.append(example['paragraph_attention_mask'])
                labels_list.append(example['labels'])

            paragraph_input_ids = torch.stack(paragraph_input_ids_list)
            paragraph_attention_mask = torch.stack(paragraph_attention_mask_list)
            labels = torch.tensor(labels_list)

            batch_query_input_ids.append(query_input_ids)
            batch_query_attention_mask.append(query_attention_mask)
            batch_paragraph_input_ids.append(paragraph_input_ids)
            batch_paragraph_attention_mask.append(paragraph_attention_mask)
            batch_labels.append(labels)

        batch_query_input_ids = torch.stack(batch_query_input_ids)
        batch_query_attention_mask = torch.stack(batch_query_attention_mask)
        batch_paragraph_input_ids = torch.stack(batch_paragraph_input_ids)
        batch_paragraph_attention_mask = torch.stack(batch_paragraph_attention_mask)
        batch_labels = torch.stack(batch_labels)

        return {
            'query_input_ids': batch_query_input_ids,
            'query_attention_mask': batch_query_attention_mask,
            'paragraph_input_ids': batch_paragraph_input_ids,
            'paragraph_attention_mask': batch_paragraph_attention_mask,
            'labels': batch_labels
        }







    
    def _create_contrastive_datapoints(self, dataset, batch_size):
        query_pos_indices = {}
        query_neg_indices = {}
        
        for idx, element in enumerate(dataset):
            query_id = element["qid"]
            if query_id not in query_pos_indices:
                query_pos_indices[query_id] = []
                query_neg_indices[query_id] = []

            if element["labels"] == 1:
                query_pos_indices[query_id].append(element)
            else:
                query_neg_indices[query_id].append(element)

        unique_queries = list(query_pos_indices.keys())
        
        final_dataset = []
        for query in unique_queries:
            pos_samples = query_pos_indices.get(query, [])
            neg_samples = query_neg_indices.get(query, [])
            random.shuffle(neg_samples)
            if not pos_samples or not neg_samples:
                continue
            
            for pos in pos_samples:
                if len(neg_samples) >= batch_size- 1:
                    batch = [pos] + neg_samples[:batch_size - 1]
                else:
                    batch = [pos] + neg_samples
                    batch += neg_samples[:batch_size-1-len(batch)]
                batch = batch[:batch_size]
                
                final_dataset.append(batch)
            
        return final_dataset
    
    def train(self):
        self.logger.info(f"Using device: {self.device}")

        train_examples = self._load_data(self.train_data_folder)
        val_examples = self._load_data(self.val_data_folder)
        
        if self.dual_encoders:
            query_tokenizer = self._load_tokenizer(self.config['query_model_name_or_path'])
            paragraph_tokenizer = self._load_tokenizer(self.config['doc_model_name_or_path'])
            train_dataset = DualEncoderDataset(train_examples, query_tokenizer, paragraph_tokenizer)
            val_dataset = DualEncoderDataset(val_examples, query_tokenizer, paragraph_tokenizer)
        else:
            tokenizer = self._load_tokenizer(self.config['model_name_or_path'])
            train_dataset = DualEncoderDataset(train_examples, tokenizer, tokenizer)
            val_dataset = DualEncoderDataset(val_examples, tokenizer, tokenizer)
        
        train_dataset = self._create_contrastive_datapoints(train_dataset, self.individual_datapoints)
        val_dataset = self._create_contrastive_datapoints(val_dataset, self.individual_datapoints)
        
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        model = self._load_model()
        
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-5)
        w_positive = 5.0
        w_negative = 1.0
        class_weights = torch.tensor([w_positive, w_negative]).to(self.device)
        class_weights = torch.full((self.individual_datapoints,), w_negative).to(self.device)
        class_weights[0] = w_positive
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        val_steps = self.config.get('val_steps', 1500)
        global_step = 0
        
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            model.train()
            total_loss = 0
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training Batches")):
                optimizer.zero_grad()

                query_input_ids = batch['query_input_ids'].to(self.device)
                query_attention_mask = batch['query_attention_mask'].to(self.device)
                paragraph_input_ids = batch['paragraph_input_ids'].to(self.device)
                paragraph_attention_mask = batch['paragraph_attention_mask'].to(self.device)
                
                logits = model(
                    query_input_ids,
                    query_attention_mask,
                    paragraph_input_ids,
                    paragraph_attention_mask
                )
                logits = logits - logits.max(dim=1, keepdim=True)[0]
                probabilities = F.softmax(logits, dim=1)
                targets = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)

                loss = criterion(logits, targets)

                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                global_step += 1

                if global_step % val_steps == 0 and self.step_validation:
                    self.validate(model, val_dataloader, criterion, is_step=True)

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")
            self.logger.info(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

            self.validate(model, val_dataloader, criterion, is_step=False)

            if self.save_checkpoints:
                model_dir = self.config['save_path'].format(
                    date_of_training=current_date(),
                    mode=self.encoding_type,
                    language=self.language
                )
                if self.dual_encoders:
                    self.save_dual_model(
                        model_dir=os.path.join(model_dir, "checkpoints", f"epoch_{epoch + 1}"),
                        query_model=model.query_model,
                        paragraph_model=model.paragraph_model,
                        query_tokenizer=query_tokenizer,
                        paragraph_tokenizer=paragraph_tokenizer
                    )
                else:
                    self.save_single_model(
                        model_dir=os.path.join(model_dir, "checkpoints", f"epoch_{epoch + 1}"),
                        model=model.model,
                        tokenizer=tokenizer
                    )

        model_dir = self.config["save_path"].format(
            date_of_training=current_date(),
            mode=self.encoding_type,
            language=self.language
        )
        if self.dual_encoders:
            self.save_dual_model(
                model_dir=os.path.join(model_dir, "_adapter"),
                query_model=model.query_model,
                paragraph_model=model.paragraph_model,
                query_tokenizer=query_tokenizer,
                paragraph_tokenizer=paragraph_tokenizer
            )
        else:
            self.save_single_model(
                model_dir=os.path.join(model_dir, "_adapter"),
                model=model.model,
                tokenizer=tokenizer
            )

    def validate(self, model, val_dataloader, criterion, is_step):
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation Batches"):
                query_input_ids = batch['query_input_ids'].to(self.device)
                query_attention_mask = batch['query_attention_mask'].to(self.device)
                paragraph_input_ids = batch['paragraph_input_ids'].to(self.device)
                paragraph_attention_mask = batch['paragraph_attention_mask'].to(self.device)

                logits = model(
                    query_input_ids,
                    query_attention_mask,
                    paragraph_input_ids,
                    paragraph_attention_mask
                )
                targets = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)

                loss = criterion(logits, targets)
                total_val_loss += loss.item()

                predictions = logits.argmax(dim=1)
                correct_predictions += (predictions == targets).sum().item()
                total_predictions += targets.size(0)

        avg_val_loss = total_val_loss / len(val_dataloader)
        accuracy = correct_predictions / total_predictions

        if is_step:
            print(f"Step Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
            self.logger.info(f"Step Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
        else:
            print(f"Epoch Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
            self.logger.info(f"Epoch Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")


    
    
    def save_dual_model(self, model_dir, query_model, paragraph_model, query_tokenizer, paragraph_tokenizer):
        os.makedirs(model_dir, exist_ok=True)
        query_model.save_pretrained(f'{model_dir}/query')
        paragraph_model.save_pretrained(f'{model_dir}/paragraph')
        
        query_tokenizer.save_pretrained(f'{model_dir}/query')
        paragraph_tokenizer.save_pretrained(f'{model_dir}/paragraph')
        
        self.logger.info(f"Model and Tokenizer saved in {model_dir}")
    
    def save_single_model(self, model_dir, model, tokenizer):
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(f'{model_dir}')
        tokenizer.save_pretrained(f'{model_dir}')
        self.logger.info(f"Model and Tokenizer saved in {model_dir}")
        

if __name__=="__main__":
    train_data_folder = 'input/test_train'
    val_data_folder = 'input/test_val'
    config_file = "src/models/vector_db/commons/config.yaml"
    language = train_data_folder.split('/')[-2]
    trainer = ContrastiveTrainer(
        train_data_folder=train_data_folder,
        val_data_folder=val_data_folder,
        config_file=config_file,
        device='cuda:0',
        dual_encoders=False,
        language=language,
        epochs=5,
        lr=3e-4,
        save_checkpoints=False,
        step_validation=False
    )
    trainer.train()
    pass