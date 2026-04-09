import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class GlucosePreprocessor:
    def __init__(self, window_size=24, prediction_horizon=6):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler()
        self.feature_cols = ['glucose', 'bolus', 'carbs', 'steps']
        
    def load_and_clean(self, file_path):
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Handle missing values - interpolate glucose, fill 0 for others
        df['glucose'] = df['glucose'].interpolate(method='linear')
        df['bolus'] = df['bolus'].fillna(0)
        df['carbs'] = df['carbs'].fillna(0)
        df['steps'] = df['steps'].fillna(0)
        
        return df

    def create_sequences(self, data):
        X, y = [], []
        scaled_data = self.scaler.fit_transform(data[self.feature_cols])
        
        for i in range(len(scaled_data) - self.window_size - self.prediction_horizon):
            X.append(scaled_data[i : i + self.window_size])
            # Target is the glucose value 'prediction_horizon' steps ahead
            # Scale target separately or use the glucose column index
            y.append(scaled_data[i + self.window_size + self.prediction_horizon - 1, 0])
            
        return np.array(X), np.array(y)

    def save_scaler(self, path='ml/scaler.gz'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to {path}")

    def load_scaler(self, path='ml/scaler.gz'):
        self.scaler = joblib.load(path)

if __name__ == "__main__":
    # Test loading
    preprocessor = GlucosePreprocessor()
    if os.path.exists('data/glucose_data.csv'):
        df = preprocessor.load_and_clean('data/glucose_data.csv')
        X, y = preprocessor.create_sequences(df)
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        preprocessor.save_scaler()
