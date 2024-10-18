from torch.utils.data import Dataset
import torch

class DualEncoderDataset(Dataset):
    def __init__(self, examples, query_tokenizer, paragraph_tokenizer, max_length=512):
        self.examples = examples
        self.query_tokenizer = query_tokenizer
        self.paragraph_tokenizer = paragraph_tokenizer
        self.max_length = max_length
        
    def tokenize_data(self):
        
        