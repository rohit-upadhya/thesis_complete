# train_model.py
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class ParagraphDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        query, paragraph, label = self.examples[idx]
        encoding = self.tokenizer.encode_plus(
            query, paragraph,
            max_length=self.max_length,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label)
        }

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        combined_query = ' '.join(item['query'])
        relevant_paragraphs = item['relevant_paragrpahs']
        for i, paragraphs in enumerate(item['all_paragraphs']):
            for paragraph in paragraphs:
                label = 1 if i in item['paragraph_numbers'] else 0
                examples.append((combined_query, paragraph, label))
    return examples

def main():
    # Load and preprocess data
    data_file = 'src/bert/test.json'
    examples = load_data(data_file)
    train_examples, val_examples = train_test_split(examples, test_size=0.1, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = ParagraphDataset(train_examples, tokenizer)
    val_dataset = ParagraphDataset(val_examples, tokenizer)

    # Load BERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./bert_paragraph_retrieval',
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        save_steps=500,
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).astype(float).mean().item()}
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model('./bert_paragraph_retrieval')
    tokenizer.save_pretrained('./bert_paragraph_retrieval')

if __name__ == "__main__":
    main()
