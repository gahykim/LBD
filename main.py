import pandas as pd
import torch
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataloader import MovieLensDataset
from model import *




parser = argparse.ArgumentParser()
parser.add_argument("--total_epoch", type = int, default = 50, help = "the total number of epoch to train")
parser.add_argument("--num_hidden", type = int, default = 512, help = "dimension of hidden embedding")
parser.add_argument("--upsilon_layer_id", type = int, default = 3, help = "id number of upsilon layer")
parser.add_argument("--batch_size", type = int, default = 64, help = "size of batch in dataloader")
parser.add_argument("--bin_size", type = int, default = 1, help = "width of discrete rating distribution")
parser.add_argument("--global_bias", type = float, default=0.3, help = "initial value of global bias")
# parser.add_argument("--top_k", type = int, default = 10, help = "number of k items to measure metric in evaluation")
opt = parser.parse_args()
print(opt)

file_path = 'movielens/ratings.dat'

df = pd.read_csv(file_path, sep='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')

# Select relevant columns
df = df[['user_id', 'movie_id', 'rating']]

# Create user and item mappings
user_ids = df['user_id'].unique()
item_ids = df['movie_id'].unique()
user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
item2idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

df['user_id'] = df['user_id'].apply(lambda x: user2idx[x])
df['movie_id'] = df['movie_id'].apply(lambda x: item2idx[x])

train_df, test_df = train_test_split(df, test_size = 0.3, random_state = 42)

train_dataset = MovieLensDataset(train_df)
test_dataset = MovieLensDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

num_users = len(user2idx)
num_items = len(item2idx)
min_rating = df['rating'].min()
max_rating = df['rating'].max()
num_epochs = opt.total_epoch

model = LBD(num_users, num_items,min_rating, max_rating, opt)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for users, items, ratings in train_loader:
        optimizer.zero_grad()
        outputs = model(users, items)
        loss = criterion(outputs, ratings.long() - 1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for users, items, ratings in test_loader:
            outputs = model(users, items)
            loss = criterion(outputs, ratings.long() - 1)
            total_loss += loss.item()
    return total_loss / len(test_loader)


for epoch in tqdm(range(num_epochs)):
    train_loss = train_model(model, train_loader, criterion, optimizer)
    test_loss = evaluate_model(model, test_loader, criterion)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')