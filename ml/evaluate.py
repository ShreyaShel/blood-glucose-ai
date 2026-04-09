import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

def evaluate():
    model_path = 'ml/model_real.h5'
    data_path = 'data/real_glucose_data.csv'
    
    if not all(os.path.exists(p) for p in [data_path]):
        print("Missing real data CSV.")
        return
        
    if os.path.exists(model_path.replace('.h5', '.pkl')):
        model = joblib.load(model_path.replace('.h5', '.pkl'))
        model_type = "RF"
    else:
        print("Model file (.pkl) not found. Please train with train_rf.py.")
        return
        
    df = pd.read_csv(data_path)
    
    # Feature Selection (must match train_real.py)
    core_cols = ['glucose', 'bolus', 'carbs', 'activity', 'hour']
    lag_cols = [f'glucose_lag_{i}' for i in range(1, 13)]
    roll_cols = ['glucose_roll_6', 'glucose_roll_12']
    feature_cols = core_cols + lag_cols + roll_cols
    
    # Use raw features for test subset (RF was trained on raw)
    pid = df['patient_id'].unique()[-1]
    test_df = df[df['patient_id'] == pid].copy()
    
    print(f"Predicting with {model_type}...")
    if model_type == "RF":
        X_eval = []
        y_actual = []
        for i in range(len(test_df) - 6):
            X_eval.append(test_df.iloc[i][feature_cols].values)
            y_actual.append(test_df.iloc[i+6]['glucose'])
        X_eval = np.array(X_eval)
        y_true = np.array(y_actual)
        y_pred = model.predict(X_eval)
    else:
        # Placeholder for other models
        print("Model type not supported for full evaluation.")
        return
    
    # Metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    print(f"Evaluation Results for Patient {pid}:")
    print(f"MAE: {mae:.2f} mg/dL")
    print(f"RMSE: {rmse:.2f} mg/dL")
    
    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(y_true[:288], label='Actual Glucose', color='#3b82f6', linewidth=2)
    plt.plot(y_pred[:288], label='AI Prediction (30m)', color='#f97316', linestyle='--', linewidth=2)
    plt.fill_between(range(288), y_true[:288], y_pred[:288], color='gray', alpha=0.2)
    plt.title(f'Continuous Glucose Monitoring - Patient {pid}')
    plt.ylabel('Glucose (mg/dL)')
    plt.xlabel('Time Steps (5 min)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    os.makedirs('ml/plots', exist_ok=True)
    plt.savefig('ml/plots/evaluation_plot.png')
    print("Plot saved to ml/plots/evaluation_plot.png")

if __name__ == "__main__":
    evaluate()
