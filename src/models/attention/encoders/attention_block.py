import torch.nn as nn # type: ignore
import torch # type: ignore
import math

class AttentionBlock(nn.Module):
    def __init__(self, embedding_dim=768, no_heads=8, ff_dim=2048, dropout=0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mha1 = nn.MultiheadAttention(embedding_dim, no_heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        
        self.mha2 = nn.MultiheadAttention(embedding_dim, no_heads, batch_first=True)
        
        self.mha3 = nn.MultiheadAttention(embedding_dim, no_heads, batch_first=True)

        self.dropout = nn.Dropout(dropout)

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
        x = x + self.dropout(attn_out1)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)

        attn_out2, attn_weights2 = self.mha2(x, x, x)
        x = x + self.dropout(attn_out2)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        
        attn_out3, attn_weights3 = self.mha3(x, x, x)
        x = x + self.dropout(attn_out3)
        
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)

        return x, attn_weights1
