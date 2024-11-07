import subprocess
import torch
from torch.utils.data import Dataset
import gzip
from tqdm import tqdm

class TokenizedChunksDataset(Dataset):
    def __init__(self, file_path, seq_len=1024, lines_to_read: int | None = None):
        self.file_path = file_path
        self.seq_len = seq_len
        self.data = []
        total_lines = 1209636

        total_lines = total_lines if lines_to_read is None else min(total_lines, lines_to_read)
        print("Total lines: ", total_lines)

        with gzip.open(self.file_path, 'rt') as f:
            i = 0
            for line in tqdm(f, total=total_lines, desc="Reading file and loading it to the memory"):
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
