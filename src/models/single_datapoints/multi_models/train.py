# train_model.py
import json
import torch
from dataclasses import dataclass, field
from transformers import BertTokenizer, BertModel, BertTokenizer
import torch.optim as optim
from sklearn.model_selection import train_test_split
from typing import Text
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss

from src.models.single_datapoints.multi_models.common.dual_encoder import DualEncoderModel
from src.models.single_datapoints.multi_models.common.paragraph_dataset import DualEncoderDataset
from src.models.single_datapoints.common.data_loader import InputLoader
from src.models.single_datapoints.common.utils import current_date

@dataclass
class RetreivalTrainer:
    data_file: Text = None
    config_file: Text = None
    
    def __post_init__(self):
        
        if self.data_file is None:
            raise ValueError("data file is not present. Please add proper data file.")
        if self.config_file is None:
            raise ValueError("data file is not present. Please add proper data file.")
        
        self.input_loader:InputLoader = InputLoader()
        self.config = self.input_loader.load_config(self.config_file)
        self.config = self.config["dual_encoder"]
        
    def tokenizer(self, model_name_or_path):
        tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        return tokenizer
    
    def load_data(self):
        data = self.input_loader.load_data(self.data_file)
        
        examples = []
        for item in data:
            combined_query = ' '.join(item['query'])
            
            for i, paragraph in enumerate(item['all_paragraphs']):
                label = 1 if (i + 1) in item['paragraph_numbers'] else 0
                examples.append((combined_query, paragraph, label))
        
        return examples
    
    def save_model(self, model_dir, query_model, paragraph_model, query_tokenizer, paragraph_tokenizer):
        query_model.save_pretrained(f'{model_dir}/query_model')
        paragraph_model.save_pretrained(f'{model_dir}/paragraph_model')
        
        query_tokenizer.save_pretrained(f'{model_dir}/query_tokenizer')
        paragraph_tokenizer.save_pretrained(f'{model_dir}/paragraph_tokenizer')
        
    def main(self):
        device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        examples = self.load_data()
        query_tokenizer = self.tokenizer(self.config['query_model_name_or_path'])
        paragraph_tokenizer = self.tokenizer(self.config['doc_model_name_or_path'])

        train_examples, val_examples = train_test_split(examples, test_size=0.1, random_state=42)
        
        train_dataset = DualEncoderDataset(train_examples, query_tokenizer, paragraph_tokenizer)
        val_dataset = DualEncoderDataset(val_examples, query_tokenizer, paragraph_tokenizer)

        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        query_model = BertModel.from_pretrained(self.config['query_model_name_or_path'])
        paragraph_model = BertModel.from_pretrained(self.config['doc_model_name_or_path'])

        query_model.to(device)
        paragraph_model.to(device)

        model = DualEncoderModel(query_model, paragraph_model)
        model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        criterion = BCEWithLogitsLoss()

        epochs = 1
        for epoch in tqdm(range(epochs)):
            model.train()
            total_loss = 0

            for batch in tqdm(train_dataloader):
                optimizer.zero_grad()
                
                query_input_ids = batch['query_input_ids'].to(device)
                query_attention_mask = batch['query_attention_mask'].to(device)
                paragraph_input_ids = batch['paragraph_input_ids'].to(device)
                paragraph_attention_mask = batch['paragraph_attention_mask'].to(device)
                labels = batch['labels'].to(device).float() 
                
                outputs = model(query_input_ids, query_attention_mask, paragraph_input_ids, paragraph_attention_mask)
                
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

            model.eval()
            total_val_loss = 0
            correct_predictions = 0
            total_predictions = 0

            with torch.no_grad():
                for batch in tqdm(val_dataloader):
                    query_input_ids = batch['query_input_ids'].to(device)
                    query_attention_mask = batch['query_attention_mask'].to(device)
                    paragraph_input_ids = batch['paragraph_input_ids'].to(device)
                    paragraph_attention_mask = batch['paragraph_attention_mask'].to(device)
                    labels = batch['labels'].to(device).float()

                    outputs = model(query_input_ids, query_attention_mask, paragraph_input_ids, paragraph_attention_mask)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()

                    predictions = torch.sigmoid(outputs) > 0.5
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.size(0)

            avg_val_loss = total_val_loss / len(val_dataloader)
            accuracy = correct_predictions / total_predictions

            print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {accuracy}")

        
        model_dir = self.config["save_path"].format(date_of_training=current_date())
        print(model_dir)
        self.save_model(model_dir=model_dir, query_model=model.query_model, paragraph_model=model.paragraph_model, query_tokenizer=query_tokenizer, paragraph_tokenizer=paragraph_tokenizer)
        # model_dir, query_model, paragraph_model, query_tokenizer, paragraph_tokenizer
        # model.query_model.save_pretrained(f'{model_dir}/query_model')
        # model.paragraph_model.save_pretrained(f'{model_dir}/paragraph_model')
        
        # query_tokenizer.save_pretrained(f'{model_dir}/query_tokenizer')
        # paragraph_tokenizer.save_pretrained(f'{model_dir}/paragraph_tokenizer')
        
if __name__ == "__main__":
    data_file = 'src/bert/common/test.json'
    config_file = "src/bert/common/config.yaml"
    trainer = RetreivalTrainer(
        data_file=data_file,
        config_file=config_file
    )
    trainer.main()