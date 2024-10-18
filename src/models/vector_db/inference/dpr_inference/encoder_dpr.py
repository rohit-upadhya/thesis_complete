from transformers import BertTokenizer, BertModel, DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer  # type: ignore
from typing import List
import torch  # type: ignore

class Encoder:
    def __init__(self, 
                 model_type: str = 'bert',
                 model_name: str = 'bert-base-uncased',
                 context_model_name: str = 'castorini/mdpr-tied-pft-msmarco',
                 question_model_name: str = 'castorini/mdpr-tied-pft-msmarco',
                 device: str = 'cpu'
                ) -> None:
        self.device = torch.device(device)
        self.model_type = model_type
        self.model_name = model_name
        self.context_model_name = context_model_name
        self.question_model_name = question_model_name
        
        self._load_models_and_tokenizers()
    
    def _load_models_and_tokenizers(self):
        if self.model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model = BertModel.from_pretrained(self.model_name, output_hidden_states=True)
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.to(self.device)
            self.model.eval()
        elif self.model_type == 'dpr':
            self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.question_model_name)
            self.question_encoder = DPRQuestionEncoder.from_pretrained(self.question_model_name, output_hidden_states=True)
            self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(self.context_model_name)
            self.context_encoder = DPRContextEncoder.from_pretrained(self.context_model_name, output_hidden_states=True)
            self.question_encoder.to(self.device)
            self.context_encoder.to(self.device)
            self.question_encoder.eval()
            self.context_encoder.eval()
    
    def encode(self, sentences: List[str], is_query: bool = True) -> torch.Tensor:
        """
        Encode sentences based on whether they are queries or context passages.

        :param sentences: List of sentences to be encoded.
        :param is_query: If True, encodes using the question encoder. If False, uses the context encoder.
        :return: Concatenated embeddings of the pooler output and mean of the last hidden state.
        """
        if self.model_type == 'bert':
            inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            pooler_output = outputs.pooler_output
            mean_hidden_state = outputs.hidden_states[-1].mean(dim=1)
            embeddings = torch.cat((pooler_output, mean_hidden_state), dim=1)

        elif self.model_type == 'dpr':
            if is_query:
                tokenizer = self.question_tokenizer
                model = self.question_encoder
            else:
                tokenizer = self.context_tokenizer
                model = self.context_encoder
                
            inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            pooler_output = outputs.pooler_output
            mean_hidden_state = outputs.hidden_states[-1].mean(dim=1)
            embeddings = torch.cat((pooler_output, mean_hidden_state), dim=1)

        return mean_hidden_state
