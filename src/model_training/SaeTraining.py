from src.model_architecture.SAE.SAE import SaeDecoder, SaeEncoder
from torch import optim
import torch
from src.model_training.training_helpers.knee_datasets import KneeScans3DDataset
import torchio as tio
import time
from src.model_training.training_helpers.loggers import WandbLogger


INPUT_SIZE = 1024
HIDDEN_SIZE = 256
MAX_HIDDEN_FEATURES = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 25
BATCH_SIZE = 2
LR = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
helper_model = torch.jit.load(
    "/home/mikic202/semestr_9/knee_scaner/models/autoencoder_model_1764428499.501627.pt"
).encoder.to(device)
helper_model.eval()

encoder = SaeEncoder(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    max_hidden_features=MAX_HIDDEN_FEATURES,
).to(device)
decoder = SaeDecoder(hidden_size=HIDDEN_SIZE, output_size=INPUT_SIZE).to(device)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=LR)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=LR)

encoder_scheduler = optim.lr_scheduler.StepLR(
    encoder_optimizer, step_size=5, gamma=0.95
)
decoder_scheduler = optim.lr_scheduler.StepLR(
    decoder_optimizer, step_size=5, gamma=0.95
)


dataset_transform = tio.transforms.Compose(
    [
        tio.transforms.Resize((64, 64, 64)),
    ]
)


dataset = KneeScans3DDataset(
    datset_filepath="/media/mikic202/Nowy1/uczelnia/semestr_9/SIWY/datasets/kneemri",
    transform=dataset_transform,
)

print(f"Dataset size: {len(dataset)} samples")

train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True
)

encoder.train()
decoder.train()

data_logger = WandbLogger(
    "knee-scanner",
    config={
        "model": "SAE",
        "hidden_size": HIDDEN_SIZE,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "loss_function": "MSE",
    },
)


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


for epoch in range(NUM_EPOCHS):
    epoch_loass = 0.0
    for batch, diagnossis in train_dataloader:
        batch = batch.to(device)
        print(batch)
        with torch.no_grad():
            base_encoded = helper_model(batch.float())

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
        f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loass / len(dataset):.4f}, sparsity_losss: {sparsity_loss.item():.4f} "
    )
    encoder_scheduler.step()
    decoder_scheduler.step()
    data_logger.log({"loss": epoch_loass / len(dataset)})

model_scripted = torch.jit.script(encoder.cpu())
model_scripted.save(f"models/SAE_encoder_model_{time.time()}.pt")
