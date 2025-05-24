import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

class IndexData(Dataset):
    def __init__(self, df, col='Open', indexes='all', patch_length=32, num_patches=10, stride=1, return_y=False):
        
        self.open_values = df[col].values.astype(np.float32)
        self.patch_length = patch_length
        self.num_patches = num_patches
        self.return_y = return_y
        
        # Calculate total length needed for each sequence
        self.total_length = patch_length * num_patches
        self.indices = []  # Will store start indices for valid sequences
        
        if indexes == 'all':
            indexes = df['Index'].unique()
        
        for index_name in indexes:
            sub = df[df['Index'] == index_name].index
            start, end = sub.min(), sub.max()
            
            required_length = self.total_length + (patch_length if return_y else 0)
            for seq_start in range(start, end - required_length + 1, stride):
                if seq_start + required_length <= end:
                    self.indices.append(seq_start)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        
        # Get the full sequence including y if required
        sequence_length = self.total_length + (self.patch_length if self.return_y else 0)
        sequence = torch.Tensor(self.open_values[start_idx:start_idx + sequence_length])
        
        # Split into x and y if required
        x_sequence = sequence[:self.total_length]
        
        # Calculate statistics for normalization
        means = torch.mean(x_sequence)
        stds = torch.std(x_sequence) + 1e-3
        
        # Reshape x into (num_patches, patch_length)
        patches = x_sequence.reshape(self.num_patches, self.patch_length)
        patches = (patches - means) / stds
        
        if self.return_y:
            # Get and normalize y using the same statistics as x
            y_sequence = sequence[self.total_length:]
            y_normalized = (y_sequence - means) / stds
            return patches, y_normalized
        
        return patches
