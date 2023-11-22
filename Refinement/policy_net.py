import torch.nn as nn
import torch
from torch_geometric.nn import GraphConv,GINConv,GCNConv


# Embedding updater
class GCN(nn.Module):
    def __init__(self, input_channels=65, hidden_channels=64):
        super(GCN, self).__init__()
        self.conv = GraphConv(input_channels, hidden_channels)

    def forward(self, x, edge_index, edge_weight):
        edge_weight_list = torch.tensor(edge_weight, dtype=torch.float)
        x = self.conv(x, edge_index,edge_weight_list)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hid = torch.tanh(self.fc1(x))
        hid = torch.tanh(self.fc2(hid))
        return self.fc3(hid)
