from src.model_architecture.SAE.ScansEncoder import ScansAutoencoder3d
from src.model_training.training_helpers.knee_datasets import KneeScans3DDataset
from torch import optim
import torchio as tio
import torch
import time
import os
from src.model_training.training_helpers.loggers import WandbLogger

ENCODED_DIM = 1024
NUM_EPOCHS = 50
BATCH_SIZE = 2
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ScansAutoencoder3d(
    input_channels=1, output_channels=1, feature_dim=ENCODED_DIM
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.94)


dataset_transform = tio.transforms.Compose(
    [
        tio.transforms.Resize((64, 64, 64)),
    ]
)


dataset = KneeScans3DDataset(
    datset_filepath="/media/mikic202/Nowy1/uczelnia/semestr_9/SIWY/datasets/kneemri",
    transform=dataset_transform,
)
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True
)

data_logger = WandbLogger(
    "knee-scanner",
    config={
        "model": "image-to-latent-autoencoder",
        "encoded_dim": ENCODED_DIM,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "loss_function": "MSE",
    },
)

try:
    if not os.path.exists("models"):
        os.makedirs("models")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch, _ in train_dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch.float())
            loss = torch.nn.functional.mse_loss(outputs, batch.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(train_dataloader.dataset)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")
        data_logger.log({"loss": epoch_loss})
        if (epoch + 1) % 5 == 0:
            scheduler.step()

except KeyboardInterrupt:
    print("Training interrupted. Saving the model...")

model_scripted = torch.jit.script(model.cpu())
model_scripted.save(f"models/autoencoder_model_{time.time()}.pt")
