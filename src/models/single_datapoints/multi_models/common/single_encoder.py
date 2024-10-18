import torch.nn as nn
import torch

class SingleEncoderModel(nn.Module):
    def __init__(self, model, hidden_dim=768):
        super(SingleEncoderModel, self).__init__()
        self.model = model
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        # self.sigmoid = nn.Sigmoid()
        # self.ff_network = nn.Sequential(
        #     nn.Linear(2 * model.config.hidden_size, hidden_dim),  # Input size is twice the hidden size due to concatenation
        #     nn.ReLU(),  # Activation function
        #     # nn.Dropout(0.5),
        #     nn.Linear(hidden_dim, 1),
        # )
    def forward(self, query_input_ids, query_attention_mask, paragraph_input_ids, paragraph_attention_mask):
        query_outputs = self.model(input_ids=query_input_ids, attention_mask=query_attention_mask)
        paragraph_outputs = self.model(input_ids=paragraph_input_ids, attention_mask=paragraph_attention_mask)
        
        query_embedding = query_outputs.last_hidden_state.mean(dim=1)
        paragraph_embedding = paragraph_outputs.last_hidden_state.mean(dim=1)
        
        similarity = self.cosine_similarity(query_embedding, paragraph_embedding)
        return similarity
        
        # combined_embedding = torch.cat((query_embedding, paragraph_embedding), dim=1) 
        # similarity_score = self.ff_network(combined_embedding)
        # return similarity_score
        # print(similarity)
        
        
        # similarity_score = self.sigmoid(similarity)
        # print(similarity_score)
        
        