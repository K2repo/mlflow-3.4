# utils/data_loader.py
import pandas as pd
import json

def load_campaign_data():
    """Cargar datos desde Laravel"""
    with open('data/raw/campaign_data.json', 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Feature engineering básico
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    return df