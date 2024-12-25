import json
import nltk
import numpy as np
import torch
from flask_cors import CORS
from data.preprocess import tokenize, bag_of_words
from models.models import NeuralNet
from utils.helper import chat_response
from flask import Flask, render_template, request, jsonify

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nltk.download('punkt')

# Load intents and model data
with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = 'chatbot_model.pth'
data = torch.load(FILE)

# Extract model parameters
input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

# Initialize and load the model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Chatbot setup
bot_name = 'Mina'


app = Flask(__name__)



@app.route('/')
def home():

    return render_template('home.html')
@app.route('/chat', methods=['POST'])
def chat():

    data = request.get_json()
    msg = data.get('message', " ")
    response = chat_response(msg,model, intents, all_words,tags)
    return jsonify({'response': response})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


CORS(app)