from transformers import BertTokenizer, BertModel, PreTrainedTokenizerFast, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer, RobertaModel  # type: ignore
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
                use_dpr: bool = False,
                use_roberta: bool = False,
                ) -> None:
        self.device = torch.device(device)
        self.question_model_name_or_path = question_model_name_or_path
        self.ctx_model_name_or_path = ctx_model_name_or_path
        print(self.question_model_name_or_path)
        print(self.ctx_model_name_or_path)
        self.use_dpr = use_dpr
        self.use_roberta = use_roberta
        self._load_tokenizer()
        self._load_model()
            
    def _load_tokenizer(self):
        if self.use_dpr:
            self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.question_model_name_or_path, use_fast=False)
            # self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base', use_fast=False)
            # self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', use_fast=False)
            
            self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(self.ctx_model_name_or_path, use_fast=False)
        elif self.use_roberta:
            print("here",self.question_model_name_or_path)
            self.question_tokenizer = PreTrainedTokenizerFast.from_pretrained(self.question_model_name_or_path, use_fast=True)
            
            self.ctx_tokenizer = PreTrainedTokenizerFast.from_pretrained(self.ctx_model_name_or_path,  use_fast=True)
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
        elif self.use_roberta:
            self.question_model = RobertaModel.from_pretrained(self.question_model_name_or_path).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.ctx_model = RobertaModel.from_pretrained(self.ctx_model_name_or_path).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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
