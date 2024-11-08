from torch.utils.data import Dataset
import os
import torch
import numpy as np

class BinaryTokenDataset(Dataset):
    def __init__(self, bin_file, seq_len=1024):
        self.bin_file = bin_file
        self.seq_len = seq_len
        self.record_size = seq_len * 4
        
        self.total_records = os.path.getsize(bin_file) // self.record_size
        print(f"Total sequences available: {self.total_records}")

    def __len__(self):
        return self.total_records

    def __getitem__(self, idx):
        if idx >= self.total_records:
            raise IndexError("Index out of range")

        with open(self.bin_file, 'rb') as f:
            f.seek(idx * self.record_size)
            data = np.frombuffer(f.read(self.record_size), dtype=np.int32)
            
            input_ids = torch.tensor(data, dtype=torch.long)
            return {'input_ids': input_ids[:-1], 'labels': input_ids[1:]}
