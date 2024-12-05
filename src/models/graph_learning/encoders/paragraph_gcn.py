import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torch_geometric.nn import GCNConv # type: ignore
from torch_geometric.data import Data # type: ignore    

class ParagraphGNN(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x) 
        return batch
