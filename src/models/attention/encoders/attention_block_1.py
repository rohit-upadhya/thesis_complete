import torch.nn as nn
import torch
import math

class AttentionBlock(nn.Module):
    def __init__(self, embedding_dim=768, no_heads=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mha1 = nn.MultiheadAttention(embedding_dim, no_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ff1 = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # self.mha2 = nn.MultiheadAttention(embedding_dim, no_heads, batch_first=True)
        # self.norm3 = nn.LayerNorm(embedding_dim)
        # self.dropout3 = nn.Dropout(dropout)

        # self.ff2 = nn.Sequential(
        #     nn.Linear(embedding_dim, ff_dim),
        #     nn.ReLU(),
        #     nn.Linear(ff_dim, embedding_dim)
        # )
        # self.norm4 = nn.LayerNorm(embedding_dim)
        # self.dropout4 = nn.Dropout(dropout)
        
        # self.mha3 = nn.MultiheadAttention(embedding_dim, no_heads, batch_first=True)
        # self.norm5 = nn.LayerNorm(embedding_dim)
        # self.dropout5 = nn.Dropout(dropout)

        # self.ff3 = nn.Sequential(
        #     nn.Linear(embedding_dim, ff_dim),
        #     nn.ReLU(),
        #     nn.Linear(ff_dim, embedding_dim)
        # )
        # self.norm6 = nn.LayerNorm(embedding_dim)
        # self.dropout6 = nn.Dropout(dropout)

    def _generate_positional_encoding(self, seq_len, device):
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, device=device).float() * -(math.log(10000.0) / self.embedding_dim))
        pos = torch.zeros(seq_len, self.embedding_dim, device=device)
        pos[:, 0::2] = torch.sin(position * div_term)
        pos[:, 1::2] = torch.cos(position * div_term)
        return pos
    
    def forward(self, x):
        seq_len, device = x.size(1), x.device
        pos_encoding = self._generate_positional_encoding(seq_len, device).unsqueeze(0)
        x = x + pos_encoding
        
        attn_out1, attn_weights1 = self.mha1(x, x, x)
        x = x + self.dropout1(attn_out1)
        x = self.norm1(x)

        ff_out1 = self.ff1(x)
        x = x + self.dropout2(ff_out1)
        x = self.norm2(x)

        # attn_out2, attn_weights2 = self.mha2(x, x, x)
        # x = x + self.dropout3(attn_out2)
        # x = self.norm3(x)

        # ff_out2 = self.ff2(x)
        # x = x + self.dropout4(ff_out2)
        # x = self.norm4(x)
        
        # attn_out3, attn_weights3 = self.mha3(x, x, x)
        # x = x + self.dropout5(attn_out3)
        # x = self.norm5(x)
        
        # ff_out3 = self.ff3(x)
        # x = x + self.dropout6(ff_out3)
        # x = self.norm6(x)

        return x, attn_weights1
        # return x, (attn_weights1, attn_weights2)
