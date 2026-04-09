import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np
from preprocess import GlucosePreprocessor
import os

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1) # Predict continuous glucose value
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train():
    print("Starting training pipeline...")
    preprocessor = GlucosePreprocessor(window_size=24, prediction_horizon=6) # 2hr history -> 30m prediction
    
    if not os.path.exists('data/glucose_data.csv'):
        print("Data not found. Please run generate_data.py first.")
        return

    df = preprocessor.load_and_clean('data/glucose_data.csv')
    
    # Simple split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    X_train, y_train = preprocessor.create_sequences(train_df)
    X_test, y_test = preprocessor.create_sequences(test_df)
    
    preprocessor.save_scaler('ml/scaler.gz')
    
    model = build_model((X_train.shape[1], X_train.shape[2]))
    
    print("Fitting model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {mae}")
    
    os.makedirs('ml', exist_ok=True)
    model.save('ml/model.h5')
    print("Model saved to ml/model.h5")

if __name__ == "__main__":
    train()
