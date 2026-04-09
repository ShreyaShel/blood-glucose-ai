import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(days=30, interval_minutes=5):
    """
    Generates synthetic blood glucose data mimicking OhioT1DM schema.
    """
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    num_steps = (days * 24 * 60) // interval_minutes
    
    timestamps = [start_time + timedelta(minutes=i * interval_minutes) for i in range(num_steps)]
    
    # Blood Glucose (mg/dL) - baseline with some oscillation and noise
    # T1DM models often use differential equations, but we'll use a simplified stochastic approach
    bg = np.zeros(num_steps)
    bg[0] = 120
    
    # Insulin and Meals (events)
    basal = 0.5 # units/hour
    bolus = np.zeros(num_steps)
    carbs = np.zeros(num_steps)
    steps = np.zeros(num_steps)
    
    for i in range(1, num_steps):
        t = timestamps[i]
        
        # Basal effect (slow lowering)
        bg[i] = bg[i-1] - 0.05
        
        # Meals (approx 3 meals a day + snacks)
        hour = t.hour
        if hour in [8, 13, 19] and t.minute == 0:
            if np.random.rand() > 0.1: # 90% chance of meal
                meal_carbs = np.random.randint(30, 80)
                carbs[i] = meal_carbs
                # Glucose rise over next 2 hours
                for j in range(1, 24): # 24 * 5 mins = 120 mins
                    if i + j < num_steps:
                        bg[i+j] += (meal_carbs / 10) * np.exp(-j/10) * (j/2)
        
        # Bolus (insulin) - usually taken with meals
        if carbs[i] > 0:
            units = carbs[i] / 10 + np.random.normal(0, 0.5)
            bolus[i] = max(0, units)
            # Insulin lowering effect over next 4 hours
            for j in range(1, 48):
                if i + j < num_steps:
                    bg[i+j] -= (bolus[i] * 2) * np.exp(-j/15) * (j/5)

        # Random activity (steps)
        if 7 < hour < 22:
            steps[i] = np.random.poisson(100)
            if steps[i] > 200:
                bg[i] -= (steps[i] / 500)

        # Brownian noise and stabilization
        bg[i] += np.random.normal(0, 1.5)
        
        # Safety bounds
        bg[i] = max(40, min(400, bg[i]))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'glucose': bg,
        'finger_stick': [bg[i] if i % 144 == 0 else np.nan for i in range(num_steps)], # once a day
        'basal': [basal] * num_steps,
        'bolus': bolus,
        'carbs': carbs,
        'steps': steps
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data(days=60) # 2 months of data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/glucose_data.csv', index=False)
    print("Data saved to data/glucose_data.csv")
