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

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == USER_CREDENTIALS['username'] and password == USER_CREDENTIALS['password']:
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid Credentials")
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')


# weather and demand---------------------------------------------------------------------------------
@app.route('/demand-prediction', methods=['GET', 'POST'])
def demand_prediction():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        weeks = int(request.form['weeks'])
        timestamps = [(datetime.now() + timedelta(days=7 * (i + 1))).strftime('%Y-%m-%d') for i in range(weeks)]

        results = {}
        for district, model in models.items():
            data = pd.read_csv(f'./datasets/demand_datasets/{district}.csv')
            data['scaler'] = MinMaxScaler().fit(data[['demand']])
            data['demand_scaled'] = data['scaler'].transform(data[['demand']])
            results[district] = infer_demand(data, model, weeks)

            # Save to database if not already present
            for i, timestamp in enumerate(timestamps):
                record = {
                    "timestamp": str(timestamp),  # Ensure timestamp is a string
                    "district": district,  # District is already a string
                    "week": int(i + 1),  # Week as an integer
                    "demand_value": float(results[district][i]),  # Convert demand_value to Python float
                }
                if collection1.count_documents(record) == 0:
                    collection1.insert_one(record)

        return render_template('demand_prediction.html', results=results, weeks=weeks)
    return render_template('demand_prediction.html')


@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Get the selected number of weeks from the form
        weeks = int(request.form['weeks'])  # Example: "1" for the first week
        districts = ['colombo', 'gampaha', 'kalutara', 'kandy', 'matale', 'nuwaraeliya', 'galle', 'matara', 'hambantota', 'jaffna', 'vanni', 'batticaloa', 'digamadulla', 'trincomalee', 'kurunegala', 'puttalam', 'anuradhapura','polonnaruwa', 'badulla', 'monaragala', 'kegalle','rathnapura']
        results = {}

        # Load the datasets and models for each district
        for district in districts:
            data = pd.read_csv(f'./datasets/demand_datasets/{district}.csv')
            model_path = f'./models/demand_predictor_models/weather_tea_demand_predictor_{district}_v2.pth'
            model = LSTMModel(input_size=2, hidden_size=64, num_layers=2, output_size=1)
            model.load_state_dict(torch.load(model_path))
            model.to(device)

            # Perform demand prediction for all 4 weeks
            predictions = infer_demand(data, model, 4, district)
            results[district] = predictions

            # Save results to MongoDB
            for i in range(4):  # Always process all 4 weeks
                week = f"Week {i + 1}"
                timestamp = (datetime.now() + timedelta(days=7 * (i + 1))).strftime('%Y-%m-%d')
                # Check if the record already exists
                existing = collection1.find_one({
                    "timestamp": timestamp,
                    "district": district,
                    "week": week
                })
                # If no existing record, insert the new record
                if not existing:
                    collection1.insert_one({
                        "timestamp": timestamp,
                        "district": district,
                        "week": week,
                        "demand_value": float(predictions[i])
                    })
        # Render results in the template
        return render_template('demand_prediction.html', results=results, weeks=weeks)

    except Exception as e:
        return str(e), 500



# raw material demand ---------------------------------------------------------------

# Load the trained LSTM model
model2 = load_model2("./models/lstm_raw_tea_demand_model_v2.h5", compile=False)

# Load and preprocess data
data2 = pd.read_csv('./datasets/weekly_raw_supply_for_inventory.csv')
data2['timestamp'] = pd.to_datetime(data2['timestamp'])
data2.sort_values('timestamp', inplace=True)

# Scale the data
scaler2 = MinMaxScaler()
scaled_values2 = scaler2.fit_transform(data2['value'].values.reshape(-1, 1))

# Define the forecasting route
@app.route('/raw-material-forecast', methods=['GET', 'POST'])
def raw_material_forecast():
    try:
        # Predefined week start offsets
        week_offsets = [28, 35, 42, 49]
        week_labels = [
            "The week started 28 days later from today",
            "The week started 35 days later from today",
            "The week started 42 days later from today",
            "The week started 49 days later from today"
        ]

        results = []

        if request.method == 'POST':
             # Get the selected week from the dropdown
            selected_weeks = int(request.form['weeks'])  # Example: "2" for the first 2 weeks

# Forecasting for the next 4 weeks
            last_sequence = scaled_values2[-4:]  # Sequence length is 4
            last_sequence = last_sequence.reshape(1, 4, 1)
            forecast = []

            for i in range(4):
                pred = model2.predict(last_sequence, verbose=0)
# Dummy user credentials---------------------------------------------------------------------------------------
USER_CREDENTIALS = {'username': 'admin', 'password': 'admin123'}

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)