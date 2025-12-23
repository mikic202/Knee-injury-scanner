import torch.nn as nn
from typing import Callable

from src.model_architecture.SAE.SAE import SaeDecoder, SaeEncoder
from src.model_architecture.cnn_clasifier.cnn_clasifier import CnnKneeClassifier
from torch import optim
import torch
from src.model_training.training_helpers.knee_datasets import KneeScans3DDataset
import torchio as tio
import time
from src.model_training.training_helpers.loggers import WandbLogger


def sae_loss(reconstructed, original, hidden, rho=0.05, beta=0.02):
    eps = 1e-7
    reconstruction_loss = torch.nn.functional.mse_loss(reconstructed, original)

    rho_hat = hidden.mean(dim=0)
    rho_hat = torch.clamp(rho_hat, eps, 1 - eps)

    kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log(
        (1 - rho) / (1 - rho_hat)
    )
    sparsity_loss = kl.sum()

    return reconstruction_loss + beta * sparsity_loss, sparsity_loss


def train_sae(
    output_from_layer: Callable,
    dataset,
    input_size: int,
    hidden_size: int,
    max_number_of_hidden_features: int,
    num_of_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 0.001,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = SaeEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        max_hidden_features=max_number_of_hidden_features,
    ).to(device)
    decoder = SaeDecoder(hidden_size=hidden_size, output_size=input_size).to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    encoder_scheduler = optim.lr_scheduler.StepLR(
        encoder_optimizer, step_size=5, gamma=0.95
    )
    decoder_scheduler = optim.lr_scheduler.StepLR(
        decoder_optimizer, step_size=5, gamma=0.95
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    for epoch in range(num_of_epochs):
        epoch_loass = 0.0
        for batch, _ in train_dataloader:
            batch = batch.to(device)
            with torch.no_grad():
                base_encoded = output_from_layer(batch.float())

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            hidden = encoder(base_encoded)

            reconstructed = decoder(hidden)

            loss, sparsity_loss = sae_loss(reconstructed, base_encoded, hidden)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            epoch_loass += loss.item() * batch.size(0)

        print(
            f"Epoch {epoch+1}/{num_of_epochs}, Loss: {epoch_loass / len(dataset):.4f}, sparsity_losss: {sparsity_loss.item():.4f} "
        )
        encoder_scheduler.step()
        decoder_scheduler.step()
    return encoder


def explain_model_with_sae(
    model: nn.Module,
    dataset,
    layer_to_explain,
    input_size: int,
    hidden_size: int,
    max_number_of_hidden_features: int,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    layer_activations = [None]

    def model_hooks(_, __, output):
        layer_activations[0] = output.detach()

    layer_to_explain.register_forward_hook(model_hooks)

    def get_output_form_desierd_layer(inputs: torch.Tensor):
        model(inputs)
        return nn.Flatten()(layer_activations[0])

    return train_sae(
        get_output_form_desierd_layer,
        dataset,
        input_size,
        hidden_size,
        max_number_of_hidden_features,
    )


if __name__ == "__main__":

    dataset_transform = tio.transforms.Compose(
        [
            tio.transforms.Resize((64, 64, 64)),
        ]
    )

    dataset = KneeScans3DDataset(
        datset_filepath="/media/mikic202/Nowy1/uczelnia/semestr_9/SIWY/datasets/kneemri",
        transform=dataset_transform,
    )

    model = CnnKneeClassifier(num_classes=3, input_channels=1)
    model.load_state_dict(
        torch.load(
            "/home/mikic202/semestr_9/knee_scaner/models/basic_clasifier_model_1766343254.9682245.pth"
        )
    )

    print(
        get_sae_for_single_layer(
            model, dataset, model.last_feature, 64 * 16 * 16 * 16, 128, 50
        )
    )
