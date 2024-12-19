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

        # Load the appropriate models based on whether DPR is used or not
        if self.use_dpr:
            self.query_encoder = DPRQuestionEncoder.from_pretrained(query_model_name_or_path).to(self.device)
        elif self.use_roberta:
            self.query_encoder = RobertaModel.from_pretrained(query_model_name_or_path).to(self.device)
        else:
            self.query_encoder = BertModel.from_pretrained(query_model_name_or_path).to(self.device)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.1)

    def forward(self, query_inputs):
        # Process query using the appropriate encoder
        # print(query_inputs)
        query_output = self.query_encoder(
            input_ids=query_inputs['input_ids'].to(self.device),
            attention_mask=query_inputs['attention_mask'].to(self.device)
        )
        
        query_embedding = self._get_embedding(query_output)  # Shape: [batch_size, hidden_size]

        # Process positive samples using the context encoder
        
        query_embedding = query_embedding.unsqueeze(1)

        return query_embedding  # Return similarities as logits

    def _get_embedding(self, model_output):
        # Get the embedding depending on whether DPR or a different encoder is used
        if self.use_dpr:
            return self.dropout(model_output.pooler_output)  # Use pooler output for DPR models
        else:
            # For non-DPR models, use the mean of the last hidden state
            return self.dropout(model_output.last_hidden_state.mean(dim=1))

# Example instantiation
# model = UnifiedContrastiveModel(use_dpr=True, device='cuda' if torch.cuda.is_available() else 'cpu')
