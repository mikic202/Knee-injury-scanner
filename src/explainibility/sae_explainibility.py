import torch.nn as nn
from typing import Callable
import numpy as np

from src.model_architecture.SAE.SAE import SaeDecoder, SaeEncoder
from src.model_architecture.SAE.ScansEncoder import ScansAutoencoder3d
from torch import optim
import torch
from src.model_training.training_helpers.knee_datasets import KneeScans3DDataset
import torchio as tio
from collections import defaultdict
from src.explainibility.visualization import display_sae_features
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def kl_divergence(p, q, eps=1e-6):
    q = torch.clamp(q, eps, 1 - eps)
    p = torch.clamp(p, eps, 1 - eps)
    return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))


def sae_loss(recon, original, hidden, beta=2000, rho=0.05):
    recon_loss = torch.nn.functional.mse_loss(recon, original, reduction="mean")

    rho_hat = hidden.mean(dim=0)
    rho_hat = torch.clamp(rho_hat, 1e-4, 1 - 1e-4)
    sparsity_loss = kl_divergence(torch.full_like(rho_hat, rho), rho_hat).mean()

    return recon_loss, sparsity_loss


def train_sae(
    output_from_layer: Callable,
    dataset,
    input_size: int,
    hidden_size: int,
    max_number_of_hidden_features: int,
    num_of_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    start_max_hidden_features: int = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not start_max_hidden_features:
        start_max_hidden_features = min(2 * max_number_of_hidden_features, hidden_size)

    encoder = SaeEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        max_hidden_features=start_max_hidden_features,
    ).to(device)
    hidden_feature_size_per_epoch = np.linspace(
        start_max_hidden_features, max_number_of_hidden_features, num_of_epochs // 2
    )
    hidden_feature_size_per_epoch = np.concatenate(
        [
            hidden_feature_size_per_epoch,
            num_of_epochs // 2 * [max_number_of_hidden_features],
        ],
    )
    decoder = SaeDecoder(hidden_size=hidden_size, output_size=input_size).to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    encoder_scheduler = optim.lr_scheduler.StepLR(
        encoder_optimizer, step_size=1, gamma=0.95
    )
    decoder_scheduler = optim.lr_scheduler.StepLR(
        decoder_optimizer, step_size=1, gamma=0.95
    )

    # labels = [dataset[i][1] for i in range(len(dataset))]

    # class_sample_count = np.array(
    #     [len(np.where(labels == t)[0]) for t in np.unique(labels)]
    # )

    # weight = 1.0 / class_sample_count
    # samples_weight = torch.from_numpy(np.array([weight[t] for t in labels])).double()

    # # 4. Create the sampler
    # sampler = torch.utils.data.WeightedRandomSampler(
    #     samples_weight, len(samples_weight)
    # )

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    for epoch in range(num_of_epochs):
        epoch_loass = 0.0
        act_freq = torch.zeros(hidden_size).to(device)
        for batch, _ in train_dataloader:
            batch = batch.to(device)
            with torch.no_grad():
                base_encoded = output_from_layer(batch.float())

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            hidden = encoder(base_encoded)
            act_freq += (hidden > 0).float().sum(0)  # Count activations

            reconstructed = decoder(hidden)

            loss, sparsity_loss = sae_loss(reconstructed, base_encoded, hidden)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            epoch_loass += loss.item() * batch.size(0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)

        print(
            f"Epoch {epoch+1}/{num_of_epochs}, Loss: {epoch_loass / len(dataset):.4f}, sparsity_losss: {sparsity_loss.item():.4f} "
        )
        dead_neurons = (act_freq == 0).nonzero(as_tuple=True)[0]
        if len(dead_neurons) > 0:
            print(f"Resampling {len(dead_neurons)} dead neurons...")
            # with torch.no_grad():
            #     replacement_weights = base_encoded[
            #         torch.randint(0, base_encoded.size(0), (len(dead_neurons),))
            #     ]
            #     encoder.linear.weight.data[dead_neurons] = replacement_weights
            #     decoder.linear.weight.data[:, dead_neurons] = replacement_weights.T

        with torch.no_grad():
            activation_rate = hidden.mean().item()
        print(f"Activation rate: {activation_rate:.4f}")
        encoder_scheduler.step()
        decoder_scheduler.step()
        encoder.max_hidden_features = int(
            hidden_feature_size_per_epoch[
                min(epoch, len(hidden_feature_size_per_epoch) - 1)
            ]
        )
        print("current max hidden features:", encoder.max_hidden_features)
    return encoder


def explain_model_with_sae(
    model: nn.Module,
    dataset,
    layer_to_explain,
    input_size: int,
    hidden_size: int,
    max_number_of_hidden_features: int,
    num_of_epochs: int = 15,
    learning_rate: float = 0.007,
    start_max_hidden_features: int = None,
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
        num_of_epochs=num_of_epochs,
        learning_rate=learning_rate,
        start_max_hidden_features=start_max_hidden_features,
    )


def calculate_sae_statistics_per_class(sae_statistics: dict[int, list[np.ndarray]]):
    feature_popularity_order_per_class = {}
    feature_counts_per_class = {}

    for diagnosis, hidden_representations in sae_statistics.items():
        hidden_representations = np.vstack(hidden_representations)
        feature_counts = np.sum(hidden_representations != 0, axis=0)
        feature_counts_per_class[diagnosis] = feature_counts

        feature_popularity_order = np.argsort(-feature_counts)
        feature_popularity_order_per_class[diagnosis] = feature_popularity_order

    return feature_popularity_order_per_class, feature_counts_per_class


def sae_statistics(
    sae_model: nn.Module,
    base_model: nn.Module,
    layer_to_explain,
    dataset: KneeScans3DDataset,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sae_model = sae_model.to(device)
    sae_model.eval()

    layer_activations = [None]

    def model_hooks(_, __, output):
        layer_activations[0] = output.detach()

    layer_to_explain.register_forward_hook(model_hooks)

    def get_output_form_desierd_layer(inputs: torch.Tensor):
        base_model(inputs)
        return nn.Flatten()(layer_activations[0])

    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    output_sparsity = defaultdict(list)

    for mri_scan, diagnosis in test_dataloader:
        mri_scan = mri_scan.to(device)
        with torch.no_grad():
            base_encoded = get_output_form_desierd_layer(mri_scan.float())
            hidden = sae_model(base_encoded)
            output_sparsity[diagnosis.item()].append(hidden.cpu().numpy())

    return (
        dict(output_sparsity),
        *calculate_sae_statistics_per_class(output_sparsity),
    )


def get_minimal_tree_from_sae_model(
    sae_model: nn.Module,
    base_model: nn.Module,
    layer_to_explain,
    dataset: KneeScans3DDataset,
):
    hidden_representations = []
    representations_diagnosis = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    sae_model = sae_model.to(device)
    sae_model.eval()

    layer_activations = [None]

    def model_hooks(_, __, output):
        layer_activations[0] = output.detach()

    layer_to_explain.register_forward_hook(model_hooks)

    def get_output_form_desierd_layer(inputs: torch.Tensor):
        base_model(inputs)
        return nn.Flatten()(layer_activations[0])

    for mri_scan, diagnosis in test_dataloader:
        mri_scan = mri_scan.to(device)
        with torch.no_grad():
            base_encoded = get_output_form_desierd_layer(mri_scan.float())
            hidden = sae_model(base_encoded)
            hidden_representations.append(hidden.cpu().numpy())
            representations_diagnosis.append(diagnosis.item())

    tree_precisions = {}
    for tree_depth in range(5, 26):
        tree = RandomForestClassifier(max_depth=tree_depth)
        tree.fit(np.vstack(hidden_representations), representations_diagnosis)
        tree_precisions[tree_depth] = (
            tree.score(np.vstack(hidden_representations), representations_diagnosis),
            tree,
            np.argsort(tree.feature_importances_)[-10:][::-1],
        )
    return tree_precisions


if __name__ == "__main__":

    dataset_transform = tio.transforms.Compose(
        [
            tio.transforms.Resize((64, 64, 64)),
        ]
    )

    dataset = KneeScans3DDataset(
        datset_filepath="/media/mikic202/Nowy/uczelnia/semestr_9/SIWY/datasets/kneemri",
        transform=dataset_transform,
    )

    model = torch.jit.load(
        "/home/mikic202/semestr_9/knee_scaner/models/autoencoder_model_1766763541.5195265.pt"
    )
    torch.save(
        model.state_dict(),
        "/home/mikic202/semestr_9/knee_scaner/models/autoencoder_model_1766763541.5195265.pth",
    )
    model = ScansAutoencoder3d(input_channels=1, output_channels=1, feature_dim=1024)
    model.load_state_dict(
        torch.load(
            "/home/mikic202/semestr_9/knee_scaner/models/autoencoder_model_1766763541.5195265.pth"
        )
    )

    model = model.encoder

    sae_model = explain_model_with_sae(
        model,
        dataset,
        model.fc,
        1024,
        8 * 4096,
        12000,
        num_of_epochs=25,
        learning_rate=0.004,
    )

    a, b, c = sae_statistics(sae_model, model, model.fc, dataset)

    np.save("sae_statistics_per_class.npy", a)

    import pandas as pd
    from sklearn.feature_extraction.text import TfidfTransformer

    df = pd.DataFrame.from_dict(c, orient="index")
    df.columns = [f"feature_{i}" for i in range(df.shape[1])]

    # 2. Apply TF-IDF
    # This penalizes features that appear in every class (like generic edges)
    # and boosts features that appear mostly in one class (like "pointed ears").
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(df.values).toarray()
    tfidf_df = pd.DataFrame(tfidf_matrix, index=df.index, columns=df.columns)

    # 3. Get Top 5 Discriminative Features for each class
    for class_name in sorted(tfidf_df.index):
        print(f"\n--- Top features for {class_name} ---")
        top_features = tfidf_df.loc[class_name].nlargest(10)
        print(top_features)

    torch.jit.save(
        torch.jit.script(sae_model.cpu()),
        "/home/mikic202/semestr_9/knee_scaner/models/sae_model_explainibility.pt",
    )

    print(get_minimal_tree_from_sae_model(sae_model, model, model.fc, dataset))

    display_sae_features(a, output_path=Path("."))
