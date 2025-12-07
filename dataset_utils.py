import importlib.util
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FHNN_TimeSeries_Dataset(Dataset):
    def __init__(self, dataframe_X, dataframe_Y, sequence_length=8, target_length=8):
        self.sequence_length = sequence_length
        self.data = dataframe_X
        self.target = dataframe_Y
        self.target_length = target_length

    def __len__(self):
        return len(self.data) - self.sequence_length - self.target_length + 1

    def __getitem__(self, idx):
        # Each sequence is a consecutive chunk of data
        sequence = self.data[idx : idx + self.sequence_length + self.target_length].copy()
        
        # ALWAYS mask the stream_temp column for the "future" portion
        # assumes last column is the stream_temp feature
        sequence[-self.target_length :, -1] = 0

        target = self.target[idx + self.sequence_length : idx + self.sequence_length + self.target_length]
        return torch.FloatTensor(sequence), torch.FloatTensor(target)

    def fill_na_forecast(self, idx, val):
        self.target.iloc[idx] = val.numpy()

    def fill_na_pretrain(self, idx, val):
        self.data.iloc[idx] = val.numpy()

def normalize_columns(features, target, feature_indices, mean_X, std_X, mean_Y, std_Y, eps=1e-10):
    # Normalize selected columns of the features
    features_normalized = features.copy()  # Make a copy to preserve original data
    features_normalized[:, feature_indices] = (features[:, feature_indices] - mean_X[feature_indices]) / (std_X[feature_indices] + eps)

    # Normalize the selected columns of the target
    target_normalized = target.copy()
    target_normalized = (target - mean_Y) / (std_Y + eps)

    return features_normalized, target_normalized

def forecast_data_prep(site_idx, sites, E=0, num_sites=70, num_features=6, sequence_length=360, prediction_length=8, data_dir="./forecast_data/"):
    site_ids = list(sites.keys())
    one_hot_encoding = np.zeros((1, num_sites))
    one_hot_encoding[:, site_ids.index(site_idx)] = 1

    feature_index = list(range(num_features - 1)) + [-1]
    data = np.load(f"{data_dir}/sequence_length_360/{site_idx}.npz", allow_pickle=True)
    norm_info = np.load(f"{data_dir}/norm_information.npz", allow_pickle=True)
    norms_X, norms_Y = norm_info["norms_X"], norm_info["norms_Y"]
    forecast_X, forecast_Y = data[f"forecast_E{E}_X"], data[f"forecast_E{E}_Y"]

    if forecast_X.shape[-1] < num_features:
        num_features_to_add = num_features - forecast_X.shape[-1]
        new_columns = np.zeros((forecast_X.shape[0], num_features_to_add))
        forecast_X = np.concatenate((forecast_X[:, :-1], new_columns, forecast_X[:, -1, None]), axis=1) # 0 feature for pretraining
    
    forecast_X = np.concatenate((forecast_X[:, :-1], np.repeat(one_hot_encoding, forecast_X.shape[0], axis=0), forecast_X[:, -1, None]), axis=1)
    forecast_X, forecast_Y = normalize_columns(forecast_X, forecast_Y, feature_index, *norms_X, *norms_Y, eps=1e-10)
    
    forecast_dataset = FHNN_TimeSeries_Dataset(
        forecast_X, forecast_Y,
        sequence_length=sequence_length,
        target_length=prediction_length 
    )

    return forecast_dataset