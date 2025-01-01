import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class ParagraphGATInference(nn.Module):
    def __init__(self, hidden_dim, heads=8, final_heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=final_heads, concat=False)

        self.activation = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x_residual = x
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = x + x_residual

        x_residual = x
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = x + x_residual

        x_residual = x
        x = self.conv3(x, edge_index)
        x = x + x_residual

        data.x = x  
        return data
