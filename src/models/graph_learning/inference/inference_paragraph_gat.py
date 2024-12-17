import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torch_geometric.nn import GATConv # type: ignore

class ParagraphGAT(nn.Module):
    def __init__(self, hidden_dim, heads=8, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
            for _ in range(num_layers)
        ])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
        data.x = x
        return data