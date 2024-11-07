import torch
from torch.utils.data import Dataset
import gzip
from tqdm import tqdm

class TokenizedChunksDataset(Dataset):
    def __init__(self, file_path, seq_len=1024, lines_to_read: int | None = None):
        self.file_path = file_path
        self.seq_len = seq_len
        self.data = []

        with gzip.open(self.file_path, 'rt') as f:
            i = 0
            for line in tqdm(f):
                if lines_to_read is not None and i > lines_to_read:
                    break
                tokens = list(map(int, line.strip().split()))
                assert len(tokens) == self.seq_len, "Each sequence must have 1024 tokens."
                self.data.append(tokens)
                i += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.data[idx], dtype=torch.long)

        return {'input_ids': input_ids[:-1], 'labels': input_ids[1:]}
