import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def build_real_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1) # Predict glucose
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    return model

def create_real_sequences(data, feature_cols, window_size=24, prediction_horizon=6):
    X, y = [], []
    for i in range(len(data) - window_size - prediction_horizon):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size + prediction_horizon - 1, 0]) # Glucose is col 0
    return np.array(X), np.array(y)

def train():
    df_path = 'data/real_glucose_data.csv'
    if not os.path.exists(df_path):
        print("Real data CSV not found. Preprocess XML first.")
        return
        
    df = pd.read_csv(df_path)
    
    # Feature Selection
    # Based on preprocess_xml: [glucose, basal, bolus, carbs, activity, lags..., rolls..., hour, is_night, windows]
    # We'll use glucose, bolus, carbs, activity as core, plus lags.
    core_cols = ['glucose', 'bolus', 'carbs', 'activity', 'hour']
    lag_cols = [f'glucose_lag_{i}' for i in range(1, 13)]
    roll_cols = ['glucose_roll_6', 'glucose_roll_12']
    feature_cols = core_cols + lag_cols + roll_cols
    
    print(f"Training on features: {feature_cols}")
    
    # Scale per patient or global? User asked for multi-patient model.
    # We'll scale globally for simplicity, but in production per-patient or patient-embedding is better.
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    patient_ids = df['patient_id'].unique()
    X_total, y_total = [], []
    
    for pid in patient_ids:
        print(f"Generating sequences for patient {pid}...")
        patient_data = df_scaled[df_scaled['patient_id'] == pid][feature_cols].values
        X_p, y_p = create_real_sequences(patient_data, feature_cols, window_size=24, prediction_horizon=6)
        X_total.append(X_p)
        y_total.append(y_p)
        
    X_all = np.concatenate(X_total, axis=0)
    y_all = np.concatenate(y_total, axis=0)
    
    # Split
    split = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]
    
    joblib.dump(scaler, 'ml/scaler_real.gz')
    
    model = build_real_model((X_train.shape[1], X_train.shape[2]))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    print("Fitting LSTM model on real Ohio data...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop]
    )
    
    results = model.evaluate(X_test, y_test)
    print(f"Test Loss: {results[0]}, MAE: {results[1]}, RMSE: {results[2]}")
    
    model.save('ml/model_real.h5')
    print("Model saved to ml/model_real.h5")

if __name__ == "__main__":
    train()
