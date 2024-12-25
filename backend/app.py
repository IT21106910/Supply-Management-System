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




@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)