from torch.utils.data import Dataset
import torch
import pickle
import os


def load_pickle_image(file_path):
    with open(file_path, "rb") as f:
        img = pickle.load(f)
    return img


class KneeScans3DDataset(Dataset):
    def __init__(self, datset_filepath: str, transform=None):
        self.load_data(datset_filepath)
        self.transform = transform

    def load_data(self, filepath: str):
        self.data = []
        for dirname, _, filenames in os.walk(filepath):
            for filename in filenames:
                if filename.endswith(".pck"):
                    self.data.append(load_pickle_image(os.path.join(dirname, filename)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx]).unsqueeze(0)
        if self.transform:
            return self.transform(sample)
        return sample
