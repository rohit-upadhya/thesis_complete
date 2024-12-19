import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class ParagraphGAT(nn.Module):
    def __init__(self, hidden_dim, heads=8, final_heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        # self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=final_heads, concat=False)
        
        # self.activation = nn.ReLU()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(0.3)  # Add regularization

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First GAT layer
        x_residual = x  # Save raw node features
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = x + x_residual  # Skip connection
        
        # Second GAT layer
        # x_residual = x
        # x = self.conv2(x, edge_index)
        # x = self.activation(x)
        # x = self.dropout(x)
        # x = x + x_residual  # Skip connection
        
        # Third GAT layer
        x_residual = x
        x = self.conv3(x, edge_index)
        x = x + x_residual  # Final skip connection
        
        data.x = x
        return data
