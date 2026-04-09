import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

def train_rf():
    df_path = 'data/real_glucose_data.csv'
    if not os.path.exists(df_path):
        print("Real data CSV not found.")
        return
        
    df = pd.read_csv(df_path)
    
    # Feature Selection (must match preprocessing)
    core_cols = ['glucose', 'bolus', 'carbs', 'activity', 'hour']
    lag_cols = [f'glucose_lag_{i}' for i in range(1, 13)]
    roll_cols = ['glucose_roll_6', 'glucose_roll_12']
    feature_cols = core_cols + lag_cols + roll_cols
    
    print(f"Training RandomForest on features: {feature_cols}")
    
    # Multiple patients training
    X = df[feature_cols]
    # Predict 30 mins ahead (glucose 6 steps later)
    # Since preprocessing already aligned them? No, we need to create the 'y'
    y = df['glucose'].shift(-6) # 30 mins ahead
    
    # Drop the rows with NaNs at the end due to shift
    X = X[:-6]
    y = y[:-6]
    
    # Split by patient (last patient for test)
    last_patient = df['patient_id'].unique()[-1]
    train_mask = df.iloc[:-6]['patient_id'] != last_patient
    test_mask = df.iloc[:-6]['patient_id'] == last_patient
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print("Fitting model...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"RandomForest Test Results:")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    os.makedirs('ml', exist_ok=True)
    joblib.dump(model, 'ml/model_real.h5') # Saving as .h5 (it's actually a pickle but backend expects this name)
    # Wait, renaming to .pkl is better but I'll stick to a consistent path
    joblib.dump(model, 'ml/model_real.pkl')
    print("Model saved to ml/model_real.pkl")

if __name__ == "__main__":
    train_rf()
