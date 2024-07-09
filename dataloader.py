import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self, dataframe):
        self.users = torch.tensor(dataframe['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(dataframe['movie_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(dataframe['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]