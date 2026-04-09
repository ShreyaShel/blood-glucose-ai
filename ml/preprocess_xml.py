import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
from datetime import datetime

def parse_ohio_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    patient_id = root.attrib.get('id', 'unknown')
    
    data = {}
    
    tags = {
        'glucose': ('glucose_level', 'value'),
        'finger_stick': ('finger_stick', 'value'),
        'basal': ('basal', 'value'),
        'bolus': ('bolus', 'dose'),
        'carbs': ('meal', 'carbs'),
        'activity': ('exercise', 'intensity')
    }
    
    for key, (tag_name, val_attr) in tags.items():
        results = []
        node = root.find(tag_name)
        if node is not None:
            for event in node.findall('event'):
                if 'ts' in event.attrib:
                    row = {'ts': event.attrib['ts']}
                    if key == 'activity':
                        # Exercise has intensity and duration
                        intensity = float(event.attrib.get('intensity', 0))
                        duration = float(event.attrib.get('duration', 0))
                        row['activity'] = intensity * duration
                    else:
                        row[key] = float(event.attrib.get(val_attr, 0))
                    results.append(row)
        data[key] = pd.DataFrame(results)
    
    # Convert all 'ts' to datetime
    for k in data:
        if not data[k].empty:
            data[k]['ts'] = pd.to_datetime(data[k]['ts'], format='%d-%m-%Y %H:%M:%S')
            data[k] = data[k].set_index('ts')
            
    # Resample and Merge
    # Start/End from glucose
    start = data['glucose'].index.min()
    end = data['glucose'].index.max()
    new_index = pd.date_range(start=start, end=end, freq='5min')
    
    df_main = pd.DataFrame(index=new_index)
    
    # Merge Glucose (mean for the interval)
    df_main = df_main.join(data['glucose'].resample('5min').mean())
    df_main['glucose'] = df_main['glucose'].interpolate(method='linear')
    
    # Merge Basal (forward fill)
    if not data['basal'].empty:
        df_main = df_main.join(data['basal'].resample('5min').mean())
        df_main['basal'] = df_main['basal'].ffill().fillna(0)
    else:
        df_main['basal'] = 0
        
    # Merge Bolus (sum in interval)
    if not data['bolus'].empty:
        df_main = df_main.join(data['bolus'].resample('5min').sum())
        df_main['bolus'] = df_main['bolus'].fillna(0)
    else:
        df_main['bolus'] = 0
        
    # Merge Meals
    if not data['carbs'].empty:
        df_main = df_main.join(data['carbs'].resample('5min').sum())
        df_main['carbs'] = df_main['carbs'].fillna(0)
    else:
        df_main['carbs'] = 0
        
    # Merge Exercise
    if not data['activity'].empty:
        df_main = df_main.join(data['activity'].resample('5min').sum())
        df_main['activity'] = df_main['activity'].fillna(0)
    else:
        df_main['activity'] = 0
        
    df_main['patient_id'] = patient_id
    
    return df_main.reset_index().rename(columns={'index': 'timestamp'})

def process_all_patients(directory='.'):
    all_dfs = []
    for file in os.listdir(directory):
        if file.endswith('.xml') and ('training' in file or 'testing' in file):
            print(f"Processing {file}...")
            df = parse_ohio_xml(os.path.join(directory, file))
            all_dfs.append(df)
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df

def engineer_features(df):
    # Lag features
    for i in range(1, 13):
        df[f'glucose_lag_{i}'] = df['glucose'].shift(i)
    
    # Rolling features
    df['glucose_roll_6'] = df['glucose'].rolling(window=6).mean() # 30 mins
    df['glucose_roll_12'] = df['glucose'].rolling(window=12).mean() # 60 mins
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
    
    # Meal/Insulin window (simple decay)
    # Note: Real modeling uses PK/PD, but here we add simple windows
    df['meal_window'] = df['carbs'].rolling(window=24, min_periods=1).sum() # Carbs in last 2 hours
    df['insulin_window'] = df['bolus'].rolling(window=48, min_periods=1).sum() # Insulin in last 4 hours
    
    # Drop rows with NaN from lags
    df = df.dropna().reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("Starting XML parsing...")
    df = process_all_patients('c:/Users/shrey/GLUCOSE_PREGICTION')
    print("Engineering features...")
    df = engineer_features(df)
    df.to_csv('data/real_glucose_data.csv', index=False)
    print(f"Saved {len(df)} rows to data/real_glucose_data.csv")
