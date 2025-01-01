import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm

class ParagraphGAT(nn.Module):
    def __init__(self, hidden_dim, edge_dim=1, heads=8, final_heads=4):
        super().__init__()
        # self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        # self.norm1 = LayerNorm(hidden_dim)
        # self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        # self.norm2 = LayerNorm(hidden_dim)
        # self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=final_heads, concat=False)
        # self.norm3 = LayerNorm(hidden_dim)
        
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.norm1 = LayerNorm(hidden_dim)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.norm2 = LayerNorm(hidden_dim)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=final_heads, concat=False)
        self.norm3 = LayerNorm(hidden_dim)
        
        self.activation = nn.ReLU()
        # self.activation = nn.PReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x_residual = x 
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = x + x_residual 
        x = self.activation(x)
        x = self.dropout(x)
        
        
        x_residual = x
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = x + x_residual
        x = self.activation(x)
        x = self.dropout(x)
        
        
        x_residual = x
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        
        data.x = x
        return data
