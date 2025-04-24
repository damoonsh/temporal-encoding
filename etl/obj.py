import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F


class IndexData(Dataset):
    def __init__(self, df, col, indexes, patch_length, stride, return_y=False):
        self.open_values = df[col].values.astype(np.float32)
        self.patch_length = patch_length
        self.stride = stride
        self.return_y = return_y
        
        total_length = len(self.open_values)
        self.index_map, self.indice_pos = [], []
        self.indices = []

        if indexes == 'all': indexes = df['Index'].unique()
        
        for index_name in indexes:
            sub = df[df['Index'] == index_name].index
            self.indice_pos.append((sub.min(), sub.max()))
            sub_indices = [(i,i+patch_length) for i in range(sub.min(), sub.max() - patch_length + 1, STRIDE)]
            self.indices.extend(sub_indices)

    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        start_idx, end_idx = self.indices[idx]
        patch = self.open_values[start_idx : end_idx]
        patch = torch.tensor(patch).unsqueeze(1)  # Shape: [patch_length]
        if self.return_y: return patch, self.open_values[end_idx]
        return patch
