import torch # type: ignore
import torch.nn as nn # type: ignore
from torch_geometric.nn import GATv2Conv # type: ignore

class ParagraphGATGated(nn.Module):
    def __init__(self, hidden_dim, heads=8, final_heads=4, ff_dim=2048, dropout_rate=0.3):
        super().__init__()

        self.first_gat_conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.first_gat_conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.first_gat_conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=final_heads, concat=False)

        self.second_gat_conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.second_gat_conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.second_gat_conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=final_heads, concat=False)

        self.gating_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim)
        )

        self.gate_layer = nn.Linear(hidden_dim * 2, 1)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def process_graph(self, x, edge_index, num_paragraphs, conv1, conv2, conv3):
        
        if isinstance(num_paragraphs, torch.Tensor):
            num_paragraphs = num_paragraphs.item()
        
        if edge_index is None or edge_index.numel() == 0 or edge_index.dim() != 2 or edge_index.size(1) == 0:
            return x[:num_paragraphs]

        x_residual = x
        x = conv1(x, edge_index)
        x = x + x_residual
        x = self.activation(x)
        x = self.dropout(x)

        x_residual = x
        x = conv2(x, edge_index)
        x = x + x_residual
        x = self.activation(x)
        x = self.dropout(x)

        x_residual = x
        x = conv3(x, edge_index)
        x = x + x_residual

        return x[:num_paragraphs]

    def forward(self, data):
        first_x, first_edge_index = data[0].x, data[0].edge_index
        final_first_x = self.process_graph(
            first_x, first_edge_index, data[0].num_paragraphs,
            self.first_gat_conv1, self.first_gat_conv2, self.first_gat_conv3
        )

        second_x, second_edge_index = data[1].x, data[1].edge_index
        final_second_x = self.process_graph(
            second_x, second_edge_index, data[1].num_paragraphs,
            self.second_gat_conv1, self.second_gat_conv2, self.second_gat_conv3
        )
        combined_x = torch.cat([final_first_x, final_second_x], dim=-1)
        gate = torch.sigmoid(self.gate_layer(combined_x))
        x_fused = gate * final_first_x + (1 - gate) * final_second_x
        # gated_x = self.gating_layer(combined_x)

        return x_fused
