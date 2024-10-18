import torch
import logging
from dataclasses import dataclass
from transformers import BertTokenizer, BertModel, BertTokenizer
import torch.optim as optim
from sklearn.model_selection import train_test_split
from typing import Text, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import random

import time 

from src.models.single_datapoints.multi_models.common.dual_encoder import DualEncoderModel
from src.models.single_datapoints.multi_models.common.single_encoder import SingleEncoderModel
from src.models.single_datapoints.multi_models.common.paragraph_dataset import DualEncoderDataset
from src.models.single_datapoints.common.data_loader import InputLoader
from src.models.single_datapoints.common.utils import current_date
from src.models.vector_db.training.sampler import BatchSampler
from src.models.vector_db.inference.inference import Inference
@dataclass
class ContrastiveTrainer:
    train_data_folder: Optional[Text] = None
    val_data_folder: Optional[Text] = None
    config_file: Optional[Text] = None
    batch_size: int = 8
    val_batch_size: int = 8
    epochs: int = 1
    device: str = 'cpu'
    dual_encoders: bool = True
    language: str = 'english'
    lr: float = 2e-5
    save_checkpoints: bool = False
    step_validation: bool = False
    gradient_accumulation_steps: int = 4
    def __post_init__(self):
        
        if self.train_data_folder is None:
            raise ValueError("data file is not present. Please add proper data file.")
        if self.config_file is None:
            raise ValueError("config file is not present. Please add proper config file.")
        
        self.input_loader:InputLoader = InputLoader()
        self.config = self.input_loader.load_config(self.config_file)
        self.config = self.config["dual_encoder"] if self.dual_encoders else self.config["single_encoder"]# type: ignore
        self.encoding_type = "dual" if self.dual_encoders else "single"
        self.train_batch_counts = 0
        self.val_batch_counts = 0
        log_file_name = os.path.join("output/model_logs",f"training_{self.language}_{self.encoding_type}_{current_date()}.log")
        self.setup_logging(log_file_name)
    
    def setup_logging(self, log_file_name):
        # Configure the logger
        logging.basicConfig(
            filename=log_file_name,
            filemode='w',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging is set up. Log file: {log_file_name}")
        
    def tokenizer(self, model_name_or_path):
        tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        return tokenizer
    
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
    
    def load_data(self, data_file):
        data = self._load_all_input_from_dir(data_file)
        # data = self.input_loader.load_data(self.data_file) # type: ignore
        counts = 0
        examples = []
        for item in data: # type: ignore
            combined_query = ', '.join(item['query'])
            case_name = item['case_name']
            case_link = item['link']
            for i, paragraph in enumerate(item['all_paragraphs']):
                label = 1 if (i + 1) in item['paragraph_numbers'] else 0
                examples.append((combined_query, paragraph, label, case_name, case_link))
            counts += len(item["paragraph_numbers"])
        
        return examples, counts
    
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
    
    def main(self):
        if 'cuda' in self.device:
            device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        self.logger.info(f"Using device: {device}")

        train_examples, train_batch_counts = self.load_data(self.train_data_folder)
        val_examples, val_batch_counts = self.load_data(self.val_data_folder)
        
        with open("src/models/vector_db/training/text.txt", "a+") as file:
                for item in train_examples:
                    file.write(f"{item}\n")
        # random.shuffle(train_examples)
        # random.shuffle(val_examples)

        # Initialize tokenizers and datasets
        if self.dual_encoders:
            query_tokenizer = self.tokenizer(self.config['query_model_name_or_path'])
            paragraph_tokenizer = self.tokenizer(self.config['doc_model_name_or_path'])
            train_dataset = DualEncoderDataset(train_examples, query_tokenizer, paragraph_tokenizer)
            val_dataset = DualEncoderDataset(val_examples, query_tokenizer, paragraph_tokenizer)
        else:
            tokenizer = self.tokenizer(self.config['model_name_or_path'])
            train_dataset = DualEncoderDataset(train_examples, tokenizer, tokenizer)
            val_dataset = DualEncoderDataset(val_examples, tokenizer, tokenizer)

        train_sampler = BatchSampler(dataset=train_dataset, batch_size=self.batch_size, batch_counts=train_batch_counts)
        val_sampler = BatchSampler(dataset=val_dataset, batch_size=self.batch_size, batch_counts=val_batch_counts)

        train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
        val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler)
        
        # Initialize the model (single or dual encoder)
        if self.dual_encoders:
            query_model = BertModel.from_pretrained(self.config['query_model_name_or_path'])
            paragraph_model = BertModel.from_pretrained(self.config['doc_model_name_or_path'])

            query_model.to(device)
            paragraph_model.to(device)

            model = DualEncoderModel(query_model, paragraph_model)
            model.to(device)
        else:
            single_model = BertModel.from_pretrained(self.config['model_name_or_path'])
            single_model.to(device)
            model = SingleEncoderModel(single_model)
            model.to(device)

        self.logger.info(f"Model initialized: {model}")

        # Replace NLLLoss with BCEWithLogitsLoss
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0)
        # criterion = torch.nn.BCEWithLogitsLoss()  # <-- Updated loss function
        criterion = torch.nn.NLLLoss()
        log_softmax = torch.nn.LogSoftmax(dim=1)
        val_steps = self.config.get('val_steps', 20)
        global_step = 0
        accumulation_steps = self.gradient_accumulation_steps
        
        for epoch in tqdm(range(self.epochs)):
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                # print(f"Batch size (query_input_ids): {batch['query_input_ids'].shape[0]}")
                # optimizer.zero_grad()

                # Move input tensors to the correct device
                query_input_ids = batch['query_input_ids'].to(device)
                query_attention_mask = batch['query_attention_mask'].to(device)
                paragraph_input_ids = batch['paragraph_input_ids'].to(device)
                paragraph_attention_mask = batch['paragraph_attention_mask'].to(device)
                
                # Only the first example in the batch is correct (positive), others are negative
                batch_size = query_input_ids.size(0)
                labels = torch.zeros(batch_size, dtype=torch.float32).to(device)
                labels[0] = 1  # The first item in the batch is the positive example
                
                # Forward pass: compute cosine similarity
                outputs = model(query_input_ids, query_attention_mask, paragraph_input_ids, paragraph_attention_mask)
                log_probs = log_softmax(outputs.squeeze(-1).unsqueeze(0))
                # outputs = model(**batch)
                print(log_probs)
                # Use BCEWithLogitsLoss directly on the cosine similarity outputs
                # loss = criterion(outputs, labels)
                # loss = criterion(log_probs, torch.zeros(self.batch_size, dtype=torch.long).to(self.device)) 
                loss = criterion(log_probs, torch.zeros(1, dtype=torch.long).to(self.device)) 
                total_loss += loss.item()
                print(loss.item())
                # Backpropagation
                loss.backward()
                optimizer.step()

                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()  # Perform a gradient update
                    optimizer.zero_grad()
                    
                global_step += 1

                if global_step % val_steps == 0 and self.step_validation:
                    self.validate(model, val_dataloader, device, criterion, is_step=True)

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")
            self.logger.info(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

            # Validation step at the end of each epoch
            self.validate(model, val_dataloader, device, criterion, is_step=False)

            # Save model checkpoints if required
            if self.save_checkpoints:
                model_dir = f"{self.config['save_path'].format(date_of_training=current_date(), mode=self.encoding_type, language=self.language)}"
                if self.dual_encoders:
                    self.save_dual_model(model_dir=os.path.join(model_dir, "checkpoints", f"epoch_{epoch + 1}"), query_model=model.query_model, paragraph_model=model.paragraph_model, query_tokenizer=query_tokenizer, paragraph_tokenizer=paragraph_tokenizer)
                else:
                    self.save_single_model(model_dir=os.path.join(model_dir, "checkpoints", f"epoch_{epoch + 1}"), model=single_model, tokenizer=tokenizer)

        # Final model saving
        model_dir = self.config["save_path"].format(date_of_training=current_date(), mode=self.encoding_type, language=self.language)  # type: ignore
        if self.dual_encoders:
            self.save_dual_model(model_dir=os.path.join(model_dir, "_adapter"), query_model=model.query_model, paragraph_model=model.paragraph_model, query_tokenizer=query_tokenizer, paragraph_tokenizer=paragraph_tokenizer)
        else:
            self.save_single_model(model_dir=os.path.join(model_dir, "_adapter"), model=single_model, tokenizer=tokenizer)

    def validate(self, model, val_dataloader, device, criterion, is_step):
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        log_softmax = torch.nn.LogSoftmax(dim=1)

        with torch.no_grad():  # No need to calculate gradients during validation
            for batch in tqdm(val_dataloader):
                # Move inputs to the appropriate device
                query_input_ids = batch['query_input_ids'].to(device)
                query_attention_mask = batch['query_attention_mask'].to(device)
                paragraph_input_ids = batch['paragraph_input_ids'].to(device)
                paragraph_attention_mask = batch['paragraph_attention_mask'].to(device)
                labels = batch['labels'].to(device).float()

                # Forward pass: compute the cosine similarity
                outputs = model(query_input_ids, query_attention_mask, paragraph_input_ids, paragraph_attention_mask)
                
                # Calculate loss using BCEWithLogitsLoss directly (just like in training)
                # loss = criterion(outputs, labels)
                loss = criterion(log_softmax(outputs.squeeze(-1).unsqueeze(0)), torch.zeros(1,dtype=int).to(device)) # type: ignore
                total_val_loss += loss.item()

                # Convert logits to probabilities using sigmoid and compute predictions
                predictions = torch.sigmoid(outputs) > 0.5
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        # Calculate average validation loss and accuracy
        avg_val_loss = total_val_loss / len(val_dataloader)
        accuracy = correct_predictions / total_predictions

        if is_step:
            print(f"Step Validation Loss: {avg_val_loss}, Accuracy: {accuracy}")
            self.logger.info(f"Step Validation Loss: {avg_val_loss}, Accuracy: {accuracy}")
        else:
            print(f"Epoch Validation Loss: {avg_val_loss}, Accuracy: {accuracy}")
            self.logger.info(f"Epoch Validation Loss: {avg_val_loss}, Accuracy: {accuracy}")

    
if __name__ == "__main__":
    train_data_folder = 'input/inference_input/russian/train_test_val'
    val_data_folder = 'input/inference_input/russian/val'
    
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
        epochs=2,
        lr=0.0005,
        save_checkpoints=False,
        step_validation=False,
    )
    trainer.main()