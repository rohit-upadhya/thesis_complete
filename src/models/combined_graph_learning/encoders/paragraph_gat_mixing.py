import torch # type: ignore
import torch.nn as nn # type: ignore
from torch_geometric.nn import GATv2Conv # type: ignore

class ParagraphGATAllMixing(nn.Module):
    def __init__(self, hidden_dim, heads=8, final_heads=4, ff_dim=2048, dropout_rate=0.3):
        super().__init__()

        self.first_gat_convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False),
            GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False),
            GATv2Conv(hidden_dim, hidden_dim, heads=final_heads, concat=False)
        ])

        self.second_gat_convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False),
            GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False),
            GATv2Conv(hidden_dim, hidden_dim, heads=final_heads, concat=False)
        ])

        self.mixing_ff = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, hidden_dim)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim * 2, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, hidden_dim)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim * 2, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, hidden_dim)
            )
        ])

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def mix_actual_nodes(self, x1, x2, num_actual_nodes1, num_actual_nodes2, mixing_layer):
        
        actual_x1 = x1[:num_actual_nodes1]
        actual_x2 = x2[:num_actual_nodes2]

        mixed_actual = mixing_layer(torch.cat([actual_x1, actual_x2], dim=-1))

        return mixed_actual

    def forward(self, data):
        first_x, first_edge_index = data[0].x, data[0].edge_index
        second_x, second_edge_index = data[1].x, data[1].edge_index

        num_actual_nodes1 = data[0].num_paragraphs
        num_actual_nodes2 = data[1].num_paragraphs

        first_residual, second_residual = first_x, second_x

        for i in range(len(self.first_gat_convs)):
            
        
            if first_edge_index is None or first_edge_index.numel() == 0 or first_edge_index.dim() != 2 or first_edge_index.size(1) == 0:
                first_x_new = first_x
            
            else:
                first_x_new = self.first_gat_convs[i](first_x, first_edge_index)
                first_x_new = self.dropout(self.activation(first_x_new))
                first_x_new += first_residual
            
            if second_edge_index is None or second_edge_index.numel() == 0 or second_edge_index.dim() != 2 or second_edge_index.size(1) == 0:
                second_x_new = second_x
            
            else:
                second_x_new = self.second_gat_convs[i](second_x, second_edge_index)
                second_x_new = self.dropout(self.activation(second_x_new))
                second_x_new += second_residual

            mixed_actual = self.mix_actual_nodes(
                first_x_new, second_x_new, num_actual_nodes1, num_actual_nodes2, self.mixing_ff[i]
            )

            mixed_actual = self.dropout(self.activation(mixed_actual))

            mixed_first_x = torch.cat([mixed_actual, first_x_new[num_actual_nodes1:]], dim=0)
            mixed_second_x = torch.cat([mixed_actual, second_x_new[num_actual_nodes2:]], dim=0)

            first_x, second_x = mixed_first_x, mixed_second_x

            first_residual, second_residual = first_x, second_x
            
        # combined_x = torch.cat([first_x[:num_actual_nodes1], second_x[:num_actual_nodes2]], dim=-1)
        return mixed_actual
