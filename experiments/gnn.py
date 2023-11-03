import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


"""
Should the output dim of fc2 be num_classes=5?
What's this batch parameter: is it the number of configs which share a graph?
"""

class KernelGNN(torch.nn.Module):
    def __init__(self, node_feature_dim, config_feature_dim):
        super(KernelGNN, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc1 = torch.nn.Linear(64 + config_feature_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, data):
        node_features, edge_index, config_features, batch = data.x, data.edge_index, data.config, data.batch
        x = F.relu(self.conv1(node_features, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = torch.cat([x, config_features], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    node_feature_dim = 140
    config_feature_dim = 24
    model = KernelGNN(node_feature_dim, config_feature_dim)
    print(model)
