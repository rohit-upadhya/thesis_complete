from transformers import BertTokenizer, BertModel  # type: ignore
from typing import List
import torch  # type: ignore

class Encoder:
    def __init__(self, 
                # model_name: str = 'bert-base-multilingual-cased',
                model_name: str = 'bert-base-uncased',
                device: str = 'cpu'
                ) -> None:
        self.device = torch.device(device)
        self.model_name = model_name
        self._load_tokenizer()
        self._load_model()
            
    def _load_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    def _load_model(self):
        self.model = BertModel.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, sentences: List):
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings


