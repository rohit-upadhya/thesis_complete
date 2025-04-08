import torch.nn as nn
import torch
import torch.nn.functional as F

class SingleEncoderModel(nn.Module):
    def __init__(self, model):
        super(SingleEncoderModel, self).__init__()
        self.model = model
        self.temperature = 0.1

    def forward(self, query_input_ids, query_attention_mask, paragraph_input_ids, paragraph_attention_mask):
        batch_size, num_examples, seq_length = paragraph_input_ids.shape

        paragraph_input_ids = paragraph_input_ids.view(-1, seq_length)
        paragraph_attention_mask = paragraph_attention_mask.view(-1, seq_length)

        query_input_ids = query_input_ids.unsqueeze(1).expand(-1, num_examples, -1).contiguous().view(-1, seq_length)
        query_attention_mask = query_attention_mask.unsqueeze(1).expand(-1, num_examples, -1).contiguous().view(-1, seq_length)

        assert query_input_ids.shape[0] == paragraph_input_ids.shape[0], \
            f"Mismatch: query_input_ids.shape[0] ({query_input_ids.shape[0]}) != paragraph_input_ids.shape[0] ({paragraph_input_ids.shape[0]})"
            
        query_outputs = self.model(input_ids=query_input_ids, attention_mask=query_attention_mask)
        paragraph_outputs = self.model(input_ids=paragraph_input_ids, attention_mask=paragraph_attention_mask)

        query_embedding = query_outputs.last_hidden_state[:, 0, :]
        paragraph_embedding = paragraph_outputs.last_hidden_state[:, 0, :] 
        
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        paragraph_embedding = F.normalize(paragraph_embedding, p=2, dim=1)

        
        
        # query_embedding = F.normalize(query_embedding, p=2, dim=1)
        # paragraph_embedding = F.normalize(paragraph_embedding, p=2, dim=1)  
        # similarity = F.cosine_similarity(query_embedding, paragraph_embedding)
        # similarity = (similarity + 1) / self.temperature  # Scaling

        # similarity = similarity.view(batch_size, num_examples)

        # query_embedding = query_outputs.last_hidden_state[:, 0, :]  
        # paragraph_embedding = paragraph_outputs.last_hidden_state[:, 0, :]
        
        similarities = (query_embedding * paragraph_embedding).sum(dim=1)
        logits = similarities.view(batch_size, num_examples)
        logits = logits / self.temperature

        # logits = torch.matmul(query_embedding, paragraph_embedding.T)
        # logits = logits.diag().view(batch_size, num_examples)
        return logits
        # return similarity
