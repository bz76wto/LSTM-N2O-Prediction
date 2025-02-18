# utils/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_excel(file_path)

def add_features(df):
    df['DoY_sin'] = np.sin(2 * np.pi * df['Avg Temperature'] / 365)
    df['DoY_cos'] = np.cos(2 * np.pi * df['Avg Temperature'] / 365)
    df['Temp_Roll_Mean'] = df['Avg Temperature'].rolling(window=3).mean()
    return df

def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)
