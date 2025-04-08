import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel

class ContrastiveModel(nn.Module):
    def __init__(self, model_name_or_path, use_roberta):
        print("inside_single model")
        super(ContrastiveModel, self).__init__()
        self.encoder = RobertaModel.from_pretrained(model_name_or_path) if use_roberta else BertModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query_inputs, pos_inputs, neg_inputs):
        query_output = self.encoder(
            input_ids=query_inputs['input_ids'],
            attention_mask=query_inputs['attention_mask']
        )
        query_embedding = self.dropout(query_output.pooler_output) 

        pos_output = self.encoder(
            input_ids=pos_inputs['input_ids'],
            attention_mask=pos_inputs['attention_mask']
        )
        pos_embedding = self.dropout(pos_output.pooler_output)

        batch_size, num_negatives, seq_length = neg_inputs['input_ids'].shape

        neg_input_ids = neg_inputs['input_ids'].view(batch_size * num_negatives, seq_length)
        neg_attention_mask = neg_inputs['attention_mask'].view(batch_size * num_negatives, seq_length)

        neg_output = self.encoder(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask
        )
        neg_embedding = self.dropout(neg_output.pooler_output)

        neg_embedding = neg_embedding.view(batch_size, num_negatives, -1)

        pos_embedding = pos_embedding.unsqueeze(1) 
        embeddings = torch.cat([pos_embedding, neg_embedding], dim=1)  

        query_embedding = query_embedding.unsqueeze(1) 
        similarities = F.cosine_similarity(query_embedding, embeddings, dim=-1)  

        return similarities 
