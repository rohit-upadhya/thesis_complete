import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class ContrastiveModel(nn.Module):
    def __init__(self, model_name_or_path):
        super(ContrastiveModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query_inputs, pos_inputs, neg_inputs):
    # Process query
        query_output = self.bert(
            input_ids=query_inputs['input_ids'],
            attention_mask=query_inputs['attention_mask']
        )
        query_embedding = self.dropout(query_output.pooler_output)  # Shape: [batch_size, hidden_size]

        # Process positive samples
        pos_output = self.bert(
            input_ids=pos_inputs['input_ids'],
            attention_mask=pos_inputs['attention_mask']
        )
        pos_embedding = self.dropout(pos_output.pooler_output)  # Shape: [batch_size, hidden_size]

        # Process negative samples
        batch_size, num_negatives, seq_length = neg_inputs['input_ids'].shape

        # Flatten negatives to process them together
        neg_input_ids = neg_inputs['input_ids'].view(batch_size * num_negatives, seq_length)
        neg_attention_mask = neg_inputs['attention_mask'].view(batch_size * num_negatives, seq_length)

        neg_output = self.bert(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask
        )
        neg_embedding = self.dropout(neg_output.pooler_output)  # Shape: [batch_size * num_negatives, hidden_size]

        # Reshape back to [batch_size, num_negatives, hidden_size]
        neg_embedding = neg_embedding.view(batch_size, num_negatives, -1)

        # Combine embeddings
        pos_embedding = pos_embedding.unsqueeze(1)  # Shape: [batch_size, 1, hidden_size]
        embeddings = torch.cat([pos_embedding, neg_embedding], dim=1)  # Shape: [batch_size, 1 + num_negatives, hidden_size]

        # Compute cosine similarity
        query_embedding = query_embedding.unsqueeze(1)  # Shape: [batch_size, 1, hidden_size]
        similarities = F.cosine_similarity(query_embedding, embeddings, dim=-1)  # Shape: [batch_size, 1 + num_negatives]

        return similarities  # Return similarities as logits
