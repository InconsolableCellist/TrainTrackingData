import torch, os, pickle
from torch.utils.data import Dataset

class VRCDataset(Dataset):
    data = {}
    def __init__(self, json_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(os.path.join(root_dir, json_file), 'rb') as f:
            self.data = pickle.load(f)
        self.n_samples = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]