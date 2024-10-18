from typing import List
import torch  # type: ignore

class ValEncoder:
    def __init__(self, 
                question_model,
                ctx_model,
                question_tokenizer,
                ctx_tokenizer,
                device, 
                use_dpr = False,
                ) -> None:
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.question_tokenizer = question_tokenizer
        self.ctx_tokenizer = ctx_tokenizer
        self.use_dpr = use_dpr
        self.device = device
        # self._load_tokenizer()
        # self._load_model()
            
    # def _load_tokenizer(self):
    #     if self.use_dpr:
    #         self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.question_model_name_or_path, use_fast=False)
    #         # self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base', use_fast=False)
    #         # self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', use_fast=False)
            
    #         self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(self.ctx_model_name_or_path, use_fast=False)
    #     else:
    #         self.question_tokenizer = BertTokenizer.from_pretrained(self.question_model_name_or_path)
            
    #         self.ctx_tokenizer = BertTokenizer.from_pretrained(self.ctx_model_name_or_path)
        
    #     if self.question_tokenizer.pad_token is None:
    #         self.question_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     if self.ctx_tokenizer.pad_token is None:
    #         self.ctx_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # def _load_model(self):
        
    #     if self.use_dpr:
    #         self.question_model = DPRQuestionEncoder.from_pretrained(self.question_model_name_or_path).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    #         self.ctx_model  = DPRContextEncoder.from_pretrained(self.ctx_model_name_or_path).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    #     else:
    #         self.question_model = BertModel.from_pretrained(self.question_model_name_or_path)
    #         self.ctx_model = BertModel.from_pretrained(self.ctx_model_name_or_path)
            
        
    #     self.question_model.resize_token_embeddings(len(self.question_tokenizer))
    #     self.question_model.to(self.device)
    #     self.question_model.eval()
        
    #     self.ctx_model.resize_token_embeddings(len(self.ctx_tokenizer))
    #     self.ctx_model.to(self.device)
    #     self.ctx_model.eval()
    
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