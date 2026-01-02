from torch import nn
import torch
import numpy as np
from collections import defaultdict


class SaeEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, max_hidden_features=128):
        super().__init__()
        self.max_hidden_features = max_hidden_features
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        hidden = torch.sigmoid(self.linear(x))
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
    hidden_activations = defaultdict(list)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for scan, diagnosis in dataloader:
            scan = scan.to(device)
            hidden = encoder(scan)
            for i in range(hidden.size(0)):
                hidden_activations[diagnosis].append(hidden[i].cpu().numpy())

    top_k_sparse_actiavtion_patterns = {}
    sparsity_pattern = {}

    for diagnosis in hidden_activations:
        sparsity_pattern[diagnosis] = np.array(hidden_activations[diagnosis])
        top_k_sparse_actiavtion_patterns[diagnosis] = np.sum(
            sparsity_pattern[diagnosis] != 0, axis=0
        )

    return sparsity_pattern, top_k_sparse_actiavtion_patterns
