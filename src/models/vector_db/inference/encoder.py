from transformers import BertTokenizer, BertModel  # type: ignore
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from typing import List
import torch  # type: ignore

class Encoder:
    def __init__(self, 
                # model_name: str = 'bert-base-multilingual-cased',
                # model_name: str = 'castorini/mdpr-tied-pft-msmarco',
                # model_name: str = 'castorini/mdpr-tied-pft-msmarco-ft-all',
                question_model_name_or_path: str = 'facebook/dpr-question_encoder-single-nq-base',
                ctx_model_name_or_path: str = 'facebook/dpr-ctx_encoder-single-nq-base',
                
                # model_name: str = 'bert-base-uncased',
                device: str = 'cpu',
                use_dpr: bool = False
                ) -> None:
        self.device = torch.device(device)
        self.question_model_name_or_path = question_model_name_or_path
        self.ctx_model_name_or_path = ctx_model_name_or_path
        print(self.question_model_name_or_path)
        print(self.ctx_model_name_or_path)
        self.use_dpr = use_dpr
        self._load_tokenizer()
        self._load_model()
        print(self.question_model_name_or_path )
        print(self.ctx_model_name_or_path)
            
    def _load_tokenizer(self):
        if self.use_dpr:
            self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.question_model_name_or_path, use_fast=False)
            # self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base', use_fast=False)
            # self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', use_fast=False)
            
            self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(self.ctx_model_name_or_path, use_fast=False)
        else:
            self.question_tokenizer = BertTokenizer.from_pretrained(self.question_model_name_or_path)
            
            self.ctx_tokenizer = BertTokenizer.from_pretrained(self.ctx_model_name_or_path)
        
        if self.question_tokenizer.pad_token is None:
            self.question_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if self.ctx_tokenizer.pad_token is None:
            self.ctx_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    def _load_model(self):
        
        if self.use_dpr:
            self.question_model = DPRQuestionEncoder.from_pretrained(self.question_model_name_or_path).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.ctx_model  = DPRContextEncoder.from_pretrained(self.ctx_model_name_or_path).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        else:
            self.question_model = BertModel.from_pretrained(self.question_model_name_or_path)
            self.ctx_model = BertModel.from_pretrained(self.ctx_model_name_or_path)
            
        
        self.question_model.resize_token_embeddings(len(self.question_tokenizer))
        self.question_model.to(self.device)
        self.question_model.eval()
        
        self.ctx_model.resize_token_embeddings(len(self.ctx_tokenizer))
        self.ctx_model.to(self.device)
        self.ctx_model.eval()
    
    def encode_question(self, sentences: List):
        inputs = self.question_tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.question_model(**inputs)
        
        if self.use_dpr:
            embeddings = outputs.pooler_output
        
        else:
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings

    
    def encode_ctx(self, sentences: List):
        inputs = self.ctx_tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.ctx_model(**inputs)
        if self.use_dpr:
            embeddings = outputs.pooler_output
        
        else:
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
