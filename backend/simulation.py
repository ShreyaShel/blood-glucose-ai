import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta

class GlucoseSimulator:
    def __init__(self):
        # Resolve absolute path to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(base_dir, 'data', 'real_glucose_data.csv')
        self.model_path = os.path.join(base_dir, 'ml', 'model_multi.pkl') # Use multi model
        self.current_index = 0
        self.data = None
        self.model = None
        
        self.load_data()
        self.load_model()

    def load_data(self):
        if os.path.exists(self.data_path):
            self.data = pd.read_csv(self.data_path)
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        else:
            print(f"Warning: Data file {self.data_path} not found.")

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print("Multi-step RF Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")

    def get_next(self):
        if self.data is None or self.current_index >= len(self.data):
            return None
        
        row = self.data.iloc[self.current_index]
        ts = row['timestamp']
        self.current_index += 1
        
        actual_glucose = row['glucose']
        
        # Predict next 6 steps
        predictions = self.predict(self.current_index - 1)
        
        # Generate timestamps for predictions (every 5 mins)
        prediction_timestamps = [(ts + timedelta(minutes=5 * i)).strftime("%H:%M") for i in range(1, 7)]
        
        status = "normal"
        if actual_glucose < 70: status = "low"
        elif actual_glucose > 180: status = "high"
        
        return {
            "timestamp": ts,
            "actual": float(actual_glucose),
            "predictions": [float(p) for p in predictions],
            "prediction_timestamps": prediction_timestamps,
            "status": status,
            "bolus": float(row['bolus']),
            "carbs": float(row['carbs']),
            "activity": float(row['activity'])
        }

    def predict(self, index):
        if self.model and index < len(self.data):
            try:
                row = self.data.iloc[index]
                core_cols = ['glucose', 'bolus', 'carbs', 'activity', 'hour']
                lag_cols = [f'glucose_lag_{i}' for i in range(1, 13)]
                roll_cols = ['glucose_roll_6', 'glucose_roll_12']
                feature_cols = core_cols + lag_cols + roll_cols
                
                features = row[feature_cols].values.reshape(1, -1)
                preds = self.model.predict(features)[0] # Array of 6
                return preds
            except Exception as e:
                print(f"Prediction error: {e}")
        
        return [120.0] * 6 # Default

simulator = GlucoseSimulator()
