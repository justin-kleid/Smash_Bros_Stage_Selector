from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS
import torch
from torch import nn
import pickle
import os


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

# Load the model
num_characters = 86
num_stages = 11
model = StageRecommender(num_characters, num_stages)
model_path = os.path.join('..', 'models', 'stage_recommender.pth')
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Load char_encoder from the pickle file
char_path = os.path.join('..', 'models', 'char_encoder.pkl')
with open(char_path, 'rb') as infile:
    char_encoder = pickle.load(infile)

stage_path = os.path.join('..', 'models', 'stage_encoder.pkl')
with open(stage_path, 'rb') as infile:
    stage_encoder = pickle.load(infile)

app = Flask(__name__, static_folder='static')
CORS(app)
@app.route('/api/recommend-stage', methods=['POST'])
def recommend_stage():
    data = request.get_json()
    character = data['character']
    opponent = data['opponent']

    try:
        winner_encoded = char_encoder.transform([character])[0]
    except ValueError:
        return jsonify({'stage': 'Character not recognized'})

    try:
        loser_encoded = char_encoder.transform([opponent])[0]
    except ValueError:
        return jsonify({'stage': 'Opponent character not recognized'})

    inputs = torch.tensor([[winner_encoded, loser_encoded]])
    outputs = model(inputs)
    predicted_stage_idx = torch.argmax(outputs, dim=1).item()

    if 0 <= predicted_stage_idx < num_stages:
        recommended_stage = stage_encoder.inverse_transform([predicted_stage_idx])[0]
    else:
        recommended_stage = "Not enough data"

    return jsonify({'stage': recommended_stage})


@app.route('/')
def index():
    return render_template('character_select.html')

@app.route('/opponent')
def opponent():
    return render_template('opponent_select.html')

@app.route('/result')
def result():
    return render_template('stage_result.html')



if __name__ == '__main__':
    app.run()
