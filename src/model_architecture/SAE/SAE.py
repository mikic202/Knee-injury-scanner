from torch import nn
import torch
import numpy as np


class SaeEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, max_hidden_features=128):
        super().__init__()
        self.max_hidden_features = max_hidden_features
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        hidden = torch.sigmoid(self.linear(x))
        _, topk_idx = torch.topk(hidden, self.max_hidden_features, dim=1)
        mask = torch.zeros_like(hidden, dtype=torch.bool)
        mask.scatter_(1, topk_idx, True)
        hidden = hidden * mask
        return hidden


class SaeDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.linear(x)


def generate_sparse_autoencoder_statistics(
    encoder: SaeEncoder, dataset: torch.utils.data.Dataset
):
    encoder.eval()
    hidden_activations = []

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for data in dataloader:
            hidden = encoder(data)
            hidden_activations.append(hidden.cpu().numpy())

    sparsity_pattern = np.concatenate(hidden_activations, axis=0)
    top_k_sparse_actiavtion_patterns = np.sum(sparsity_pattern != 0, axis=0)

    return sparsity_pattern, top_k_sparse_actiavtion_patterns
