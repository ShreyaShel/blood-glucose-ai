import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from preprocess_xml import parse_ohio_xml, engineer_features

class GlucoseLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size=50, output_size=6):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        # Taking the output from the last time step
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

def train_single_patient():
    print("Loading Single Patient LSTM Pipeline...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Locate Patient 559 Training XML
    train_xml = os.path.join(base_dir, '559-ws-training.xml')
    test_xml = os.path.join(base_dir, '559-ws-testing.xml')
    
    if not os.path.exists(train_xml) or not os.path.exists(test_xml):
        print("ERROR: Training/Testing XML for patient 559 not found in root directory.")
        return
        
    print("Parsing XML data...")
    df_train = parse_ohio_xml(train_xml)
    df_test = parse_ohio_xml(test_xml)
    
    print("Engineering features...")
    df_train = engineer_features(df_train)
    df_test = engineer_features(df_test)
    
    # Features required (Aligned with previous models)
    core_cols = ['glucose', 'bolus', 'carbs', 'activity', 'hour']
    lag_cols = [f'glucose_lag_{i}' for i in range(1, 13)]
    roll_cols = ['glucose_roll_6', 'glucose_roll_12']
    feature_cols = core_cols + lag_cols + roll_cols
    
    # We are predicting the next 6 steps (30 min)
    target_cols = [f'target_{i}' for i in range(1, 7)]
    for i in range(1, 7):
        df_train[f'target_{i}'] = df_train['glucose'].shift(-i)
        df_test[f'target_{i}'] = df_test['glucose'].shift(-i)
        
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    
    X_train = df_train[feature_cols].values
    y_train = df_train[target_cols].values
    X_test = df_test[feature_cols].values
    y_test = df_test[target_cols].values

    print(f"Training shapes -> X: {X_train.shape}, y: {y_train.shape}")
    print(f"Testing shapes  -> X: {X_test.shape}, y: {y_test.shape}")
    
    # Scaling
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Reshape for LSTM [batch_size, sequence_length, num_features]
    # We don't have sequences built explicitly via rolling windows here, 
    # instead our features already contain 12-lags. We treat the sequence length as 1 with many features.
    # A true sequence LSTM would be [batch, 12, raw_features]. For simplicity, we just reshape to [batch, 1, num_features].
    X_train_tensor = torch.tensor(X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1]), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1]), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Model Setup
    model = GlucoseLSTM(input_size=len(feature_cols))
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    epochs = 40

    print("Training PyTorch LSTM on Patient 559...")
    for i in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        single_loss = loss_function(y_pred, y_train_tensor)
        single_loss.backward()
        optimizer.step()

        if i % 5 == 0:
            # Eval on test set
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test_tensor)
                test_loss = loss_function(test_pred, y_test_tensor)
                print(f"Epoch {i:3} | Train Loss: {single_loss.item():.2f} | Test Loss (MSE): {test_loss.item():.2f}")
            model.train()
            
    # Save Model & Scaler
    model_path = os.path.join(base_dir, 'ml', 'model_lstm.pt')
    scaler_path = os.path.join(base_dir, 'ml', 'scaler_lstm.pkl')
    
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler_X, scaler_path)
    print(f"LSTM Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    return df_test # We can use this to overwrite the mock data

if __name__ == "__main__":
    df_test = train_single_patient()
    # Save df_test as the new simulation dataset so the backend streams TRUE testing data of patient 559
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_data_path = os.path.join(base_dir, 'data', 'real_glucose_data.csv')
    df_test.to_csv(test_data_path, index=False)
    print("Testing data deployed to simulation engine!")
