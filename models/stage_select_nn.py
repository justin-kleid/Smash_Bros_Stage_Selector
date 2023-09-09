import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import pickle

data = pd.read_csv('../data/cleaned_smash_tourney_data.csv')

# Encode character and stage names as integers
char_encoder = LabelEncoder()
stage_encoder = LabelEncoder()

data['Winner Character'] = char_encoder.fit_transform(data['Winner Character'])
data['Loser Character'] = char_encoder.transform(data['Loser Character'])
data['Stage Name'] = stage_encoder.fit_transform(data['Stage Name'])

class SmashDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        winner = self.data.iloc[idx]['Winner Character']
        loser = self.data.iloc[idx]['Loser Character']
        stage = self.data.iloc[idx]['Stage Name']

        return torch.tensor([winner, loser]), torch.tensor(stage)
    
class StageRecommender(nn.Module):
    def __init__(self, num_characters, num_stages, embedding_dim=16):
        super(StageRecommender, self).__init__()

        self.character_embedding = nn.Embedding(num_characters, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_stages)

    def forward(self, x):
        winner, loser = x[:, 0], x[:, 1]
        winner_embed = self.character_embedding(winner)
        loser_embed = self.character_embedding(loser)
        x = torch.cat([winner_embed, loser_embed], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = SmashDataset(train_data)
val_dataset = SmashDataset(val_data)

def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        for x, y in val_loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss / len(val_loader)}")

num_characters = len(char_encoder.classes_)
num_stages = len(stage_encoder.classes_)

model = StageRecommender(num_characters, num_stages)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Train the model
train_model(model, train_loader, val_loader, epochs=10)


# Save model and JSONs
torch.save(model.state_dict(), 'stage_recommender.pth')


# Save char_encoder as a pickle file
with open('char_encoder.pkl', 'wb') as outfile:
    pickle.dump(char_encoder, outfile)

# Save stage_encoder as a pickle file
with open('stage_encoder.pkl', 'wb') as outfile:
    pickle.dump(stage_encoder, outfile)


#model_path = 'stage_recommender.pth'