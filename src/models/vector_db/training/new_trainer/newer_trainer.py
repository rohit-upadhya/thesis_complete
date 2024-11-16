import torch
import logging
from dataclasses import dataclass
from transformers import BertTokenizer, BertTokenizer
from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import torch.optim as optim
from typing import Text, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import json
import time 

from src.models.single_datapoints.common.data_loader import InputLoader
from src.models.vector_db.training.new_trainer.dual_contrastive_encoder import DualContrastiveModel as DualEncoder
from src.models.vector_db.training.new_trainer.contrastive_encoder import ContrastiveModel as SingleEncoder
from src.models.vector_db.inference.inference import Inference
from src.models.single_datapoints.common.utils import current_date

@dataclass
class ContrastiveTrainer:
    use_dpr: bool = False
    train_data_folder: Optional[Text] = None
    val_data_folder: Optional[Text] = None
    config_file: Optional[Text] = None
    batch_size: int = 8
    val_batch_size: int = 8
    individual_datapoints: int = 7
    epochs: int = 1
    device_str: str = 'cpu'
    dual_encoders: bool = True
    language: str = 'english'
    lr: float = 2e-5
    save_checkpoints: bool = False
    step_validation: bool = False
    model_name_or_path: str = "bert-base-multilingual-cased"
    query_model_name_or_path: str = "bert-base-multilingual-cased"
    ctx_model_name_or_path: str = "bert-base-multilingual-cased"
    use_translations: bool = False
    
    def __post_init__(self):
        self.current_date = current_date()
        self.input_loader = InputLoader()
        if self.config_file == "":
            raise ValueError("config file empty, please contact admin")
        self.config = self.input_loader.load_config(self.config_file) # type: ignore
        self.config = self.config["dual_encoder"] if self.dual_encoders else self.config["single_encoder"] # type: ignore
        self.device = torch.device(self.device_str if torch.cuda.is_available() and 'cuda' in self.device_str else 'cpu')
        self.encoding_type = "dual" if self.dual_encoders else "single" 
        self.recall_save = self.query_model_name_or_path.replace('/','_')
        self.save_model_name = self.query_model_name_or_path.replace('/','_')
        log_file_name = os.path.join("output/model_logs",f"training_{self.language}_{self.encoding_type}_{self.recall_save}_{current_date()}.log")
        self.setup_logging(log_file_name)
        print("Post Init completed.")
    
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
        total_datapoints = []
        for file in files:
            individual_datapoints = self.input_loader.load_data(data_file=file)
            total_datapoints.extend(individual_datapoints) # type: ignore
        
        
        print(total_datapoints[0]["query"])
        if self.use_translations:
            languages = ["english", "french", "italian", "romanian", "russian", "turkish", "ukrainian"]
            whole_translations = []
            for language in languages:
                translatons_query = self.input_loader.load_data(data_file=f"output/translation_outputs/query_translations_{language}.json")
                whole_translations.extend(translatons_query)
            for point in total_datapoints:
                final_query = []
                for query_item in point["query"]:
                    for trdata_point in whole_translations: # type: ignore
                        if trdata_point["original"] == query_item:
                            final_query.append(trdata_point["translation"])
                            
                            break
                point["query"] = final_query
        print(total_datapoints[0]["query"])
        return total_datapoints
    
    def _load_data(self, data_file, num_negatives=7):
        data = self._load_all_input_from_dir(data_file)
        data_points = []

        for item in data:
            combined_query = ', '.join(item['query'])
            positive_indices = set(item['paragraph_numbers'])
            all_paragraphs = item['all_paragraphs']
            # print(item['query'])
            # print("Positive",positive_indices)
            # print(len(all_paragraphs))
            # print(item["link"])
            # print(item["file"])
            # positive_paragraphs = []
            # counts = 0
            # for i in positive_indices:
            #     if len(all_paragraphs) > (i - 1):
            #         positive_paragraphs.append(all_paragraphs[i - 1])
            #     else:
            #         print(item["link"])
            #         print(item["file"])
            #         counts += 1
            try:
                positive_paragraphs = [all_paragraphs[i - 1] for i in positive_indices]
            except:
                print(item["link"])
                print(item["file"])
            negative_paragraphs = [p for idx, p in enumerate(all_paragraphs, 1) if idx not in positive_indices]

            if len(negative_paragraphs) < num_negatives:
                negative_paragraphs = (negative_paragraphs * ((num_negatives // len(negative_paragraphs)) + 1))[:num_negatives]
            else:
                random.shuffle(negative_paragraphs)
                negative_paragraphs = negative_paragraphs[:num_negatives]

            for pos_paragraph in positive_paragraphs:
                data_point = {
                    'query': combined_query,
                    'positive': pos_paragraph,
                    'negatives': negative_paragraphs
                }
                data_points.append(data_point)
        print("done")
        return data_points
    
    def _load_inference(self, model, data_folder, epoch):
        inference = Inference(
                    inference_folder=data_folder, 
                    bulk_inference=True,
                    use_translations=False,
                    device=self.device_str,
                    # device='cuda:2',
                    language=self.language,
                    question_model_name_or_path = self.model_name_or_path,
                    ctx_model_name_or_path = self.model_name_or_path,
                    dpr = False,
                    save_recall = False,
                    model=model.bert,
                    tokenizer=self.tokenizer,
                    device_=self.device,
                    run_val=True
                )
        recall = inference.main()
        recall["epoch"] = epoch
        return recall
    
    def _load_tokenizer(self, query_model_name_or_path, ctx_model_name_or_path):
        print(query_model_name_or_path, ctx_model_name_or_path)
        if self.use_dpr:
                query = DPRQuestionEncoderTokenizer.from_pretrained(query_model_name_or_path, use_fast=False)
                print("query loaded")
                ctx = DPRContextEncoderTokenizer.from_pretrained(ctx_model_name_or_path, use_fast=False)
                print("ctx loaded")
        else :
            query = BertTokenizer.from_pretrained(query_model_name_or_path)
            print("query loaded")
            ctx = BertTokenizer.from_pretrained(ctx_model_name_or_path)
            print("ctx loaded")
        return query, ctx

    def tokenize_data_point(self, query_tokenizer, ctx_tokenizer, data_point, max_length=512):
        query = data_point['query']
        pos_paragraph = data_point['positive']
        neg_paragraphs = data_point['negatives']

        if isinstance(pos_paragraph, list):
            pos_paragraph = ' '.join(pos_paragraph)
        
        query_inputs = query_tokenizer(
            query,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        pos_inputs = ctx_tokenizer(
            pos_paragraph,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        flat_neg_paragraphs = [' '.join(paragraph) if isinstance(paragraph, list) else paragraph for paragraph in neg_paragraphs]
        neg_inputs = ctx_tokenizer(
            flat_neg_paragraphs,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        return {
            'query_inputs': query_inputs,
            'pos_inputs': pos_inputs,
            'neg_inputs': neg_inputs 
        }


    def data_tokenizer(self, data_points, query_tokenizer, ctx_tokenizer, max_length=512):
        tokenized_dataset = []

        for data_point in data_points:
            tokenized_data = self.tokenize_data_point(
                query_tokenizer=query_tokenizer,
                ctx_tokenizer=ctx_tokenizer,
                data_point=data_point,
                max_length=max_length
            )
            tokenized_dataset.append(tokenized_data)

        return tokenized_dataset

    def collate_fn(self, batch):
        batch_query_input_ids = []
        batch_query_attention_mask = []

        batch_pos_input_ids = []
        batch_pos_attention_mask = []

        batch_neg_input_ids = []
        batch_neg_attention_mask = []

        for idx, data_point in enumerate(batch):
            query_inputs = data_point['query_inputs']
            pos_inputs = data_point['pos_inputs']
            neg_inputs = data_point['neg_inputs']

            query_input_ids = query_inputs['input_ids'].squeeze(0)
            query_attention_mask = query_inputs['attention_mask'].squeeze(0)

            pos_input_ids = pos_inputs['input_ids'].squeeze(0)
            pos_attention_mask = pos_inputs['attention_mask'].squeeze(0)

            neg_input_ids = neg_inputs['input_ids']
            neg_attention_mask = neg_inputs['attention_mask']

            batch_query_input_ids.append(query_input_ids)
            batch_query_attention_mask.append(query_attention_mask)
            batch_pos_input_ids.append(pos_input_ids)
            batch_pos_attention_mask.append(pos_attention_mask)
            batch_neg_input_ids.append(neg_input_ids)
            batch_neg_attention_mask.append(neg_attention_mask)

        batch_query_input_ids = torch.stack(batch_query_input_ids)
        batch_query_attention_mask = torch.stack(batch_query_attention_mask)
        batch_pos_input_ids = torch.stack(batch_pos_input_ids)
        batch_pos_attention_mask = torch.stack(batch_pos_attention_mask)

        batch_neg_input_ids = torch.stack(batch_neg_input_ids)
        batch_neg_attention_mask = torch.stack(batch_neg_attention_mask)

        return {
            'query_input_ids': batch_query_input_ids,
            'query_attention_mask': batch_query_attention_mask,
            'pos_input_ids': batch_pos_input_ids,
            'pos_attention_mask': batch_pos_attention_mask,
            'neg_input_ids': batch_neg_input_ids,
            'neg_attention_mask': batch_neg_attention_mask
        }
    
    def train(
        self
    ):
        self.logger.info(f"Using device: {self.device}")

        print("Pretraining steps started.")
        train_data_points = self._load_data(self.train_data_folder, num_negatives=self.individual_datapoints)
        print(f"Number of training data points: {len(train_data_points)}")
        if self.dual_encoders:
            # print(self.query_model_name_or_path, self.ctx_model_name_or_path)
            self.query_tokenizer, self.ctx_tokenizer  = self._load_tokenizer(self.query_model_name_or_path, self.ctx_model_name_or_path)
            # self.ctx_tokenizer = self._load_tokenizer(self.ctx_model_name_or_path)
        else:
            self.query_tokenizer, self.ctx_tokenizer = self._load_tokenizer(self.query_model_name_or_path, self.query_model_name_or_path)
        tokenized_data = self.data_tokenizer(
            data_points=train_data_points, 
            query_tokenizer=self.query_tokenizer,
            ctx_tokenizer=self.ctx_tokenizer,
            max_length=512)
        
        random.shuffle(tokenized_data)
        
        train_loader = DataLoader(
            tokenized_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        print("before model")
        if self.dual_encoders:
            model = model = DualEncoder(
                query_model_name_or_path=self.query_model_name_or_path,
                ctx_model_name_or_path=self.ctx_model_name_or_path,
                use_dpr=self.use_dpr,
                device=self.device_str
            ).to(self.device)
        else:
            print("before sigle")
            model = SingleEncoder(self.query_model_name_or_path).to(self.device)
        print(model)
        self.logger.info(f"model : \n  {model}")
        print(f"model : \n  {model}")
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        print("Pretraining steps completed.")
        model.train()
        self.logger.info("Training Started.")
        print("Training Started.")
        print(f"Total number of batches: {len(train_loader)}")
        print("self.epochs", self.epochs)
        for epoch in range(self.epochs):
            print(f"Starting epoch {epoch+1}/{self.epochs}")
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}")
            total_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            for batch_idx, batch in enumerate(train_loader):
                # print(f"Processing batch {batch_idx+1} in epoch {epoch+1}")
                try:
                    optimizer.zero_grad()

                    query_inputs = {
                        'input_ids': batch['query_input_ids'].to(self.device),
                        'attention_mask': batch['query_attention_mask'].to(self.device)
                    }
                    pos_inputs = {
                        'input_ids': batch['pos_input_ids'].to(self.device),
                        'attention_mask': batch['pos_attention_mask'].to(self.device)
                    }
                    neg_inputs = {
                        'input_ids': batch['neg_input_ids'].to(self.device),
                        'attention_mask': batch['neg_attention_mask'].to(self.device)
                    }

                    logits = model(query_inputs, pos_inputs, neg_inputs)
                    # print(f"Logits for batch {batch_idx+1}: {logits}")

                    labels = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)

                    loss = criterion(logits, labels)
                    total_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                    progress_bar.set_postfix(loss=loss.item())
                    progress_bar.update(1)
                except Exception as e:
                    print(f"Error in processing batch {batch_idx+1} in epoch {epoch+1}: {e}")
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            self.logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            if self.step_validation:
                recall_data = {}
                recall = self._load_inference(model=model, data_folder=self.val_data_folder, epoch=epoch)
                recall_path = "output/model_logs/val_recall/{date_of_training}_{model}_{language}_recall.json".format(date_of_training=self.current_date, model=self.recall_save, language=language)
                if os.path.exists(recall_path):
                    with open(recall_path, 'r') as json_file:
                        recall_data = json.load(json_file)
                recall_data["model"] = self.model_name_or_path
                recall_data[f"epoch_{epoch+1}"] = recall
                with open(recall_path, "w+") as file:
                    json.dump(recall_data, file, ensure_ascii=False, indent=4)
                
            if self.save_checkpoints:
                epoch_save_dir = "/srv/upadro/models/language/{mode}/turkish_{date_of_training}__{mode}__{language}__{translated}__{model_name}_training/checkpoint/epoch_{epoch}".format(
                    date_of_training=current_date(),
                    mode=self.encoding_type,
                    language=self.language,
                    epoch=epoch+1,
                    translated="translated" if self.use_translations else "not_translated",
                    model_name=f"{self.save_model_name}"
                )
                # self.save_model_and_tokenizer(model, self.tokenizer, epoch_save_dir)
                self.save_model_and_tokenizer(model, self.query_tokenizer, self.ctx_tokenizer, epoch_save_dir, is_checkpoint=True, epoch=epoch+1)
                
        # if self.step_validation:
        #     with open("output/model_logs/val_recall/{date_of_training}_{model}_recall.json".format(date_of_training=current_date(), model=self.recall_save), "w+") as file:
        #         json.dump(recall_per_epoch, file, ensure_ascii=False, indent=4)
        
        # model_save_dir = "model_output/{date_of_training}_{mode}_{language}_training/_final_model".format(
        #     date_of_training=current_date(),
        #     mode=self.encoding_type,
        #     language=self.language
        # )
        
        model_save_dir = "/srv/upadro/models/language/{mode}/turkish_{date_of_training}__{mode}__{language}__{translated}__{model_name}_training/_final_model".format(
            date_of_training=current_date(),
            mode=self.encoding_type,
            language=self.language,
            translated="translated" if self.use_translations else "not_translated",
            model_name=f"{self.save_model_name}"
        )
        
        self.save_model_and_tokenizer(model, self.query_tokenizer, self.ctx_tokenizer, model_save_dir)
        self.logger.info("Training complete.")
        print("Training Complete.")

    def save_model_and_tokenizer(self, model, query_tokenizer, ctx_tokenizer, model_save_dir, is_checkpoint=False, epoch=None):
        # Create the main save directory if it doesn't exist
        os.makedirs(model_save_dir, exist_ok=True)
        
        if self.dual_encoders:
            # Create directories for query and context models
            query_model_dir = os.path.join(model_save_dir, "query_model")
            ctx_model_dir = os.path.join(model_save_dir, "ctx_model")
            
            os.makedirs(query_model_dir, exist_ok=True)
            os.makedirs(ctx_model_dir, exist_ok=True)
            
            # Save the pre-trained models separately
            model.query_encoder.save_pretrained(query_model_dir)
            model.context_encoder.save_pretrained(ctx_model_dir)

            # Save the respective tokenizers
            query_tokenizer.save_pretrained(query_model_dir)
            ctx_tokenizer.save_pretrained(ctx_model_dir)
        else:
            # Save the single encoder model and only the query tokenizer
            model.encoder.save_pretrained(model_save_dir)
            query_tokenizer.save_pretrained(model_save_dir)
        
        if is_checkpoint:
            self.logger.info(f"Checkpoint for epoch {epoch} saved to {model_save_dir}")
        else:
            self.logger.info(f"Final model and tokenizer saved to {model_save_dir}")


    # def save_model_and_tokenizer(self, model, tokenizer, model_save_dir):
    #     os.makedirs(model_save_dir, exist_ok=True)
    #     model.bert.save_pretrained(model_save_dir)
    #     tokenizer.save_pretrained(model_save_dir)
    #     self.logger.info(f"Model and tokenizer saved to {model_save_dir}")

if __name__ == "__main__":
    # languages = ["russian", "english", "french", "italian", "romanian", "turkish", "ukrainian"]
    languages = ["turkish"]
    for language in languages:
        # dual_encoders = [False, True]
        dual_encoders = [False]
        models = [
            ['bert-base-multilingual-cased', 'bert-base-multilingual-cased'],
            ['castorini/mdpr-tied-pft-msmarco', 'castorini/mdpr-tied-pft-msmarco'],
            # ['castorini/mdpr-tied-pft-msmarco-ft-all', 'castorini/mdpr-tied-pft-msmarco-ft-all'],
            # ["facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-ctx_encoder-single-nq-base"]
            # ['bert-base-uncased', 'bert-base-uncased']
        ]
        train_data_folder = f'input/train_infer/{language}/new_split/train_test_val'
        val_data_folder = f'input/train_infer/{language}/new_split/val'
        
        print(train_data_folder)
        # train_data_folder = 'input/test_train/new_split'
        # val_data_folder = 'input/test_val/new_split'
        for dual_encoder in dual_encoders:
            for model in models:
                try:
                    if model[0] != 'bert-base-uncased':
                        # translations = [True, False]
                        translations = [False]
                    else:
                        translations = [False]
                    for use_translation in translations:
                        try:
                            config_file = "src/models/vector_db/commons/config.yaml"
                            language = train_data_folder.split('/')[-3] 
                            trainer = ContrastiveTrainer(
                                use_dpr=False,
                                train_data_folder=train_data_folder,
                                val_data_folder=val_data_folder,
                                config_file=config_file,
                                device_str='cuda:1',
                                dual_encoders=dual_encoder,
                                language=language,
                                batch_size=12,
                                epochs=8,
                                lr=5e-6,
                                save_checkpoints=True,
                                step_validation=False,
                                query_model_name_or_path=model[0],
                                ctx_model_name_or_path=model[1],
                                # model_name_or_path='bert-base-multilingual-cased',
                                use_translations=False,
                            )
                            trainer.train()
                        except Exception as e:
                            with open("output/model_logs/error.txt", "a+") as file:
                                file.write(f"error occured in encoder : {'dual' if dual_encoder else 'single'} and {model} and {'trasnlation' if use_translation else 'no translation'} \n")
                except Exception as e:
                    with open("output/model_logs/error.txt", "a+") as file:
                        file.write(f"error occured in encoder : {'dual' if dual_encoder else 'single'} and {model}\n")
        print("done")