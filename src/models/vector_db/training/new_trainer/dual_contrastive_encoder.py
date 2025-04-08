import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, DPRQuestionEncoder, DPRContextEncoder, RobertaModel

class DualContrastiveModel(nn.Module):
    def __init__(self, 
                 query_model_name_or_path='facebook/dpr-question_encoder-single-nq-base',
                 ctx_model_name_or_path='facebook/dpr-ctx_encoder-single-nq-base',
                 use_dpr=True, 
                 use_roberta=False,
                 device='cpu'):
        super(DualContrastiveModel, self).__init__()
        
        self.device = torch.device(device)
        self.use_dpr = use_dpr
        self.use_roberta = use_roberta

        if self.use_dpr:
            self.query_encoder = DPRQuestionEncoder.from_pretrained(query_model_name_or_path).to(self.device)
            self.context_encoder = DPRContextEncoder.from_pretrained(ctx_model_name_or_path).to(self.device)
        elif self.use_roberta:
            self.query_encoder = RobertaModel.from_pretrained(query_model_name_or_path).to(self.device)
            self.context_encoder = RobertaModel.from_pretrained(ctx_model_name_or_path).to(self.device)
        else:
            self.query_encoder = BertModel.from_pretrained(query_model_name_or_path).to(self.device)
            self.context_encoder = BertModel.from_pretrained(ctx_model_name_or_path).to(self.device)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, query_inputs, pos_inputs, neg_inputs):
        query_output = self.query_encoder(
            input_ids=query_inputs['input_ids'].to(self.device),
            attention_mask=query_inputs['attention_mask'].to(self.device)
        )
        
        query_embedding = self._get_embedding(query_output)

        pos_output = self.context_encoder(
            input_ids=pos_inputs['input_ids'].to(self.device),
            attention_mask=pos_inputs['attention_mask'].to(self.device)
        )
        pos_embedding = self._get_embedding(pos_output)

        batch_size, num_negatives, seq_length = neg_inputs['input_ids'].shape

        neg_input_ids = neg_inputs['input_ids'].view(batch_size * num_negatives, seq_length).to(self.device)
        neg_attention_mask = neg_inputs['attention_mask'].view(batch_size * num_negatives, seq_length).to(self.device)

        neg_output = self.context_encoder(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask
        )
        neg_embedding = self._get_embedding(neg_output)

        neg_embedding = neg_embedding.view(batch_size, num_negatives, -1)

        pos_embedding = pos_embedding.unsqueeze(1)
        embeddings = torch.cat([pos_embedding, neg_embedding], dim=1) 

        query_embedding = query_embedding.unsqueeze(1)
        similarities = F.cosine_similarity(query_embedding, embeddings, dim=-1)

        return similarities

    def _get_embedding(self, model_output):
        if self.use_dpr:
            return self.dropout(model_output.pooler_output)
        else:
            return self.dropout(model_output.last_hidden_state.mean(dim=1))

