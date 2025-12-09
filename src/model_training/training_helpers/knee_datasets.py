from torch.utils.data import Dataset
import torch
import pickle
import os
import pandas as pd


def load_pickle_image(file_path):
    with open(file_path, "rb") as f:
        img = pickle.load(f)
    return img


class KneeScans3DDataset(Dataset):
    def __init__(self, datset_filepath: str, transform=None):
        self.load_data(datset_filepath)
        self.transform = transform

    def load_data(self, filepath: str):
        self.data = {}
        for dirname, _, filenames in os.walk(filepath):
            for filename in filenames:
                if filename.endswith(".csv"):
                    self.metadata = pd.read_csv(os.path.join(dirname, filename))
                if filename.endswith(".pck"):
                    self.data[filename] = load_pickle_image(
                        os.path.join(dirname, filename)
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(list(self.data.values())[idx]).unsqueeze(0)
        class_value = self.metadata.loc[
            self.metadata["volumeFilename"] == list(self.data.keys())[idx]
        ]["aclDiagnosis"].tolist()[0]
        if self.transform:
            return self.transform(sample), class_value
        return sample, class_value
