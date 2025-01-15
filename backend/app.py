from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import torch
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model as load_model2
from bson import ObjectId
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong secret key


# Database connection
client = MongoClient("mongodb+srv://shanuka:shanuka1234@cluster0.nbyqg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['weather_and_demand']
collection1 = db['demand_predictions']
collection2 = db['raw_tea_demand']
collection3 = db['labour_availability']  # For labor availability
collection4 = db['traffic_timeslots']  # Traffic timeslots


# Dummy user credentials---------------------------------------------------------------------------------------
USER_CREDENTIALS = {'username': 'admin', 'password': 'admin123'}

# weather demand ---------------------------------------------------------------
# Load models -weather and demand
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_model(model_path):
    input_size = 2
    hidden_size = 64
    num_layers = 2
    output_size = 1
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

models = {
    'colombo': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_colombo_v2.pth'),
    'gampaha': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_gampaha_v2.pth'),
    'kalutara': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_kalutara_v2.pth'),
}





@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)