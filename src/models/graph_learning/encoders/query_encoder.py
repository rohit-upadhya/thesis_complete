import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, DPRQuestionEncoder, DPRContextEncoder, RobertaModel

class QueryGraphEncoder(nn.Module):
    def __init__(self, 
                 query_model_name_or_path='facebook/dpr-question_encoder-single-nq-base',
                 use_dpr=True, 
                 use_roberta=False,
                 device='cpu'):
        super(QueryGraphEncoder, self).__init__()
        
        self.device = torch.device(device)
        self.use_dpr = use_dpr
        self.use_roberta = use_roberta

        if self.use_dpr:
            self.query_encoder = DPRQuestionEncoder.from_pretrained(query_model_name_or_path).to(self.device)
        elif self.use_roberta:
            self.query_encoder = RobertaModel.from_pretrained(query_model_name_or_path).to(self.device)
        else:
            self.query_encoder = BertModel.from_pretrained(query_model_name_or_path).to(self.device)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, query_inputs):
        query_output = self.query_encoder(
            input_ids=query_inputs['input_ids'].to(self.device),
            attention_mask=query_inputs['attention_mask'].to(self.device)
        )
        
        query_embedding = self._get_embedding(query_output)
        
        query_embedding = query_embedding.unsqueeze(1)

        return query_embedding

    def _get_embedding(self, model_output):
        if self.use_dpr:
            return self.dropout(model_output.pooler_output)
        else:
            return self.dropout(model_output.last_hidden_state.mean(dim=1))

