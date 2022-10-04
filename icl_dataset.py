from torch.utils.data import Dataset
import json

class ICLDataset(Dataset):
    def __init__(self):
        with open('data.json') as f:
            data = json.load(f)
        self.data=data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def collate(batch):
        return [item for item in batch]
