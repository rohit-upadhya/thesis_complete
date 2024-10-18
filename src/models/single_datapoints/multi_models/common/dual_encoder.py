import torch.nn as nn

class DualEncoderModel(nn.Module):
    def __init__(self, query_model, paragraph_model):
        super(DualEncoderModel, self).__init__()
        self.query_model = query_model
        self.paragraph_model = paragraph_model
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, query_input_ids, query_attention_mask, paragraph_input_ids, paragraph_attention_mask):
        query_outputs = self.query_model(input_ids=query_input_ids, attention_mask=query_attention_mask)
        paragraph_outputs = self.paragraph_model(input_ids=paragraph_input_ids, attention_mask=paragraph_attention_mask)
        
        query_embedding = query_outputs.last_hidden_state.mean(dim=1)
        paragraph_embedding = paragraph_outputs.last_hidden_state.mean(dim=1)
        
        similarity = self.cosine_similarity(query_embedding, paragraph_embedding)
        # print(similarity)
        # similarity_score = self.sigmoid(similarity)
        # print(similarity_score)
        # similarity_score = (similarity + 1) / 2
        
        return similarity