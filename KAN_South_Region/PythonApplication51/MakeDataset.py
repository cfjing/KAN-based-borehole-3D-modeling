# MakeDataset.py

#-*- coding : utf-32 -*-
import os
import joblib 
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from torch.utils.data import random_split, Dataset, DataLoader
import CONFIG
import PATH

class CSVInMemoryDataset(Dataset):
    def __init__(self, file_path, scalers):
        self.data = pd.read_csv(file_path)
        self.x_pos_data = self.data[CONFIG.POS_AXES].values
        self.x_fac_data = self.data[CONFIG.FAC_LSIT].values
        self.y_data = self.data['Cls'].values
        
        self.pos_scaler = scalers['pos']
        self.fac_scaler = scalers['fac']
        
        self.x_pos_norm = self.pos_scaler.transform(self.x_pos_data)
        self.x_pos_norm = self.x_pos_norm.astype(np.float32)
        
        self.x_fac_norm = self.fac_scaler.transform(self.x_fac_data)
        self.x_fac_norm = self.x_fac_norm.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.x_pos_norm[index], self.x_fac_norm[index], self.y_data[index]

def make_dataset(full_data: pd.DataFrame):
    pos_scaler = sklearn.preprocessing.MinMaxScaler()
    pos_scaler.fit(full_data[CONFIG.POS_AXES].values) 
    
    fac_scaler = sklearn.preprocessing.MinMaxScaler()
    fac_scaler.fit(full_data[CONFIG.FAC_LSIT].values)

    scalers = {
        'pos': pos_scaler,
        'fac': fac_scaler
    }
    
    pkl_dataset = CSVInMemoryDataset(file_path=PATH.CSV_DATA_PATH, scalers=scalers)
    
    generator_0 = torch.Generator().manual_seed(CONFIG.INITIAL_SEED)
    train_dataset, val_dataset, test_dataset = random_split(pkl_dataset, [CONFIG.TRAINSET_RATIO, CONFIG.VALSET_RATIO, CONFIG.TESTSET_RATIO], generator_0)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.TEST_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.TEST_BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, scalers

def save_scalers(scalers):
    """保存包含多个Scaler的字典"""
    joblib.dump(scalers, PATH.SCALER_PATH)
    print(f"Scalers dictionary saved to {PATH.SCALER_PATH}")

def load_scalers():
    """加载包含多个Scaler的字典"""
    try:
        scalers = joblib.load(PATH.SCALER_PATH)
        print(f"Scalers dictionary loaded from {PATH.SCALER_PATH}")
        return scalers
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {PATH.SCALER_PATH}. Please train the model first.")
        return None