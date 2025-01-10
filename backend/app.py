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
    'kandy': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_kandy_v2.pth'),
    'matale': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_matale_v2.pth'),
    'nuwaraeliya': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_nuwaraeliya_v2.pth'),
    'galle': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_galle_v2.pth'),
    'matara': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_matara_v2.pth'),
    'hambantota': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_hambantota_v2.pth'),
    'jaffna': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_jaffna_v2.pth'),
    'vanni': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_vanni_v2.pth'),
    'batticaloa': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_batticaloa_v2.pth'),
    'digamadulla': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_digamadulla_v2.pth'),
    'trincomalee': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_trincomalee_v2.pth'),
    'kurunegala': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_kurunegala_v2.pth'),
    'puttalam': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_puttalam_v2.pth'),
    'anuradhapura': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_anuradhapura_v2.pth'),
    'polonnaruwa': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_polonnaruwa_v2.pth'),
    'badulla': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_badulla_v2.pth'),
    'monaragala': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_monaragala_v2.pth'),
    'kegalle': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_kegalle_v2.pth'),
    'rathnapura': load_model('./models/demand_predictor_models/weather_tea_demand_predictor_rathnapura_v2.pth')
}

def infer_demand(data, model, weeks, district):

    demand_column = f'demand_{district}'  # Dynamic demand column based on district
    scaler = MinMaxScaler()
    data['demand_scaled'] = scaler.fit_transform(data[[demand_column]])  # Scale the demand column

    results = []
    sequence_length = 7  # Use the last 7 days for prediction

    for _ in range(weeks):
        # Prepare the last sequence for inference
        last_sequence = data.iloc[-sequence_length:][['precip_encoded', 'demand_scaled']].values
        last_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)

        # Model inference
        model.eval()
        with torch.no_grad():
            prediction = model(last_sequence).cpu().numpy()

        # Inverse scale the prediction to get the actual demand value
        predicted_demand = scaler.inverse_transform(prediction).flatten()[0]
        results.append(predicted_demand)

        # Update the dataset with the predicted value for sequential predictions
        new_row = pd.DataFrame([[0, predicted_demand]], columns=['precip_encoded', 'demand_scaled'])
        data = pd.concat([data, new_row], ignore_index=True)

    return results

# Dummy user credentials---------------------------------------------------------------------------------------
USER_CREDENTIALS = {'username': 'admin', 'password': 'admin123'}

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)