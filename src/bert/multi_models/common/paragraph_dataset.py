from torch.utils.data import Dataset
import torch

class DualEncoderDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        query, paragraph, label = self.examples[idx]
        
        query_encoding = self.tokenizer.encode_plus(
            query,
            max_length=self.max_length,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        paragraph_encoding = self.tokenizer.encode_plus(
            paragraph,
            max_length=self.max_length,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(),
            'paragraph_input_ids': paragraph_encoding['input_ids'].squeeze(),
            'paragraph_attention_mask': paragraph_encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }