from src.model_training.training_helpers.knee_datasets import KneeScans3DDataset
from src.model_architecture.cnn_clasifier.cnn_clasifier import CnnKneeClassifier
from torch import optim
import torchio as tio
import torch
import time
import os
import numpy as np
from src.model_training.training_helpers.loggers import WandbLogger

NUM_EPOCHS = 3
BATCH_SIZE = 2
LR = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CnnKneeClassifier(num_classes=3, input_channels=1).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.94)


dataset_transform = tio.transforms.Compose(
    [
        tio.transforms.Resize((64, 64, 64)),
    ]
)


dataset = KneeScans3DDataset(
    datset_filepath="/media/mikic202/Nowy/uczelnia/semestr_9/SIWY/datasets/kneemri",
    transform=dataset_transform,
)
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True
)

data_logger = WandbLogger(
    "knee-scanner",
    config={
        "model": "basic-cnn-classifier",
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "loss_function": "MSE",
    },
)

loss_function = torch.nn.CrossEntropyLoss(
    weight=torch.tensor([0.25, 0.93, 1.5]).to(device)
)

try:
    if not os.path.exists("models"):
        os.makedirs("models")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch, classes in train_dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch.float())
            loss = loss_function(
                outputs,
                torch.nn.functional.one_hot(classes, num_classes=3).to(device).float(),
            )
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

test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

predictions = []
real = []

for batch, classes in test_dataloader:
    batch = batch.to(device)
    outputs = model(batch.float())
    _, predicted = torch.max(outputs.data, 1)
    predictions.append(predicted.item())
    real.append(classes.item())
    # print(f"True: {classes.item()}, Predicted: {predicted.item()}")

np.save(f"models/basic_clasifier_predictions_{time.time()}.npy", predictions)
np.save(f"models/basic_clasifier_real_{time.time()}.npy", real)

torch.save(model.state_dict(), f"models/basic_clasifier_model_{time.time()}.pth")

model_scripted = torch.jit.script(model.cpu())
model_scripted.save(f"models/basic_clasifier_model_{time.time()}.pt")
