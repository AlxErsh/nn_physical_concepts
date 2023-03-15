import _pickle as cPickle
import gzip

import torch

from torch.utils.data import Dataset

class EllipticDataset(Dataset):
    def __init__(self, data_file: str):
        with gzip.open(data_file, 'rb') as file:
            self.data, self.states, self.ecc = cPickle.load(file)
        self.data = torch.from_numpy(self.data)
        self.states = torch.from_numpy(self.states)
        self.ecc = torch.from_numpy(self.ecc)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.states[idx], self.ecc[idx]
