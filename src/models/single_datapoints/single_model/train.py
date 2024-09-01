
import torch
from dataclasses import dataclass, field
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from typing import Text

from src.models.single_datapoints.single_model.common.paragraph_dataset import ParagraphDataset
from src.models.single_datapoints.common.data_loader import InputLoader
from src.models.single_datapoints.common.utils import current_date

@dataclass
class RetreivalTrainer:
    data_file: Text = None
    config_file: Text = None
    
    def __post_init__(self):
        if data_file == None:
            raise ValueError("data file is not present. Please add proper data file.")
        if self.config_file is None:
            raise ValueError("data file is not present. Please add proper data file.")
        self.input_loader:InputLoader = InputLoader()
        self.config = self.input_loader.load_config(self.config_file)
        self.config = self.config["single_encoder"]
        
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

    def save_model(self, model_dir, trainer, tokenizer):
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)
        
    def main(self):
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        examples = self.load_data()
        # with open("src/bert/test_examples.txt","w+") as file:
        #     for item in examples:
        #         file.write(f"{item} \n")
        train_examples, val_examples = train_test_split(examples, test_size=0.1, random_state=42)

        tokenizer = self.tokenizer(model_name_or_path=self.config['model_name_or_path'])

        train_dataset = ParagraphDataset(train_examples, tokenizer)
        val_dataset = ParagraphDataset(val_examples, tokenizer)

        model = BertForSequenceClassification.from_pretrained(self.config['model_name_or_path'], num_labels=2)
        model.to(device)
        model_dir = self.config["save_path"].format(date_of_training=current_date())
        training_args = TrainingArguments(
            output_dir=model_dir+"model_outputs/",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            # gradient_accumulation_steps=1,
            learning_rate=5e-02,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=model_dir+'model_outputs/logs',
            logging_steps=10,
            evaluation_strategy="steps",
            save_steps=300,
            eval_steps=30,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
        )
        print(model)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).astype(float).mean().item()}
        )

        trainer.train()
        self.save_model(model_dir, trainer, tokenizer)
        
        # trainer.save_model('./bert_paragraph_retrieval')
        # tokenizer.save_pretrained('./bert_paragraph_retrieval')

if __name__ == "__main__":
    data_file = 'src/bert/common/test.json'
    config_file = "src/bert/common/config.yaml"
    trainer = RetreivalTrainer(
        data_file=data_file,
        config_file=config_file
    )
    trainer.main()
