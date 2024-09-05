from torch.utils.data import Dataset
import torch

class DualEncoderDataset(Dataset):
    def __init__(self, examples, query_tokenizer, paragraph_tokenizer, max_length=512):
        self.examples = examples
        self.query_tokenizer = query_tokenizer
        self.paragraph_tokenizer = paragraph_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        combined_query, paragraph, label, case_name, case_link = self.examples[idx]
        query_encoding = self.query_tokenizer.encode_plus(
            combined_query,
            max_length=self.max_length,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        paragraph_encoding = self.paragraph_tokenizer.encode_plus(
            paragraph,
            max_length=self.max_length,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        qid = f"{case_name}_{case_link}_{combined_query}"

        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(),
            'paragraph_input_ids': paragraph_encoding['input_ids'].squeeze(),
            'paragraph_attention_mask': paragraph_encoding['attention_mask'].squeeze(),
            'labels': label,#torch.tensor(label, dtype=torch.float),
            'qid': qid
        }