import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_multi():
    df_path = 'data/real_glucose_data.csv'
    if not os.path.exists(df_path):
        print("Real data CSV not found.")
        return
        
    df = pd.read_csv(df_path)
    
    # Feature Selection
    core_cols = ['glucose', 'bolus', 'carbs', 'activity', 'hour']
    lag_cols = [f'glucose_lag_{i}' for i in range(1, 13)]
    roll_cols = ['glucose_roll_6', 'glucose_roll_12']
    feature_cols = core_cols + lag_cols + roll_cols
    
    print(f"Training Multi-Step RF on features: {feature_cols}")
    
    X = df[feature_cols]
    
    # Predict 6 future steps (30 minutes)
    y = []
    for i in range(1, 7):
        y.append(df['glucose'].shift(-i))
    
    y = pd.concat(y, axis=1).dropna()
    X = X.iloc[:len(y)]
    
    # Split
    last_patient = df['patient_id'].unique()[-1]
    train_mask = df.iloc[:len(y)]['patient_id'] != last_patient
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    print("Fitting multi-output model...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    os.makedirs('ml', exist_ok=True)
    joblib.dump(model, 'ml/model_multi.pkl')
    print("Multi-step model saved to ml/model_multi.pkl")

if __name__ == "__main__":
    train_multi()
