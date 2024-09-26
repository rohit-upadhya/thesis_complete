from transformers import BertTokenizer, BertModel  # type: ignore
from typing import List
import torch  # type: ignore

class Encoder:
    def __init__(self, 
                model_name: str = 'bert-base-multilingual-cased',
                # model_name: str = 'bert-base-uncased',
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







# from sentence_transformers import SentenceTransformer
# from typing import List

# class Encoder:
#     def __init__(
#         self, 
#         model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
#         device: str = 'cpu'
#     ) -> None:
#         self.device = device
#         self.model_name = model_name
#         self._load_model()
        
#     def _load_model(self):
#         self.model = SentenceTransformer(self.model_name, device=self.device)
        
#     def encode(self, sentences: List[str]):
#         embeddings = self.model.encode(
#             sentences, 
#             convert_to_tensor=True, 
#             device=self.device, 
#             normalize_embeddings=True
#         )
#         return embeddings
    
    
    
    # def encode(self, sentences: List[str]):
    #     inputs = self.tokenizer(
    #         sentences, return_tensors='pt', padding=True, truncation=True, max_length=512
    #     )
    #     inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
    #     with torch.no_grad():
    #         outputs = self.model(**inputs)
        
    #     token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)
    #     attention_mask = inputs['attention_mask']
        
    #     # Expand the attention mask for broadcasting
    #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
    #     # Perform mean pooling
    #     sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    #     sum_mask = input_mask_expanded.sum(dim=1)
    #     embeddings = sum_embeddings / sum_mask.clamp(min=1e-9)
    #     return embeddings
