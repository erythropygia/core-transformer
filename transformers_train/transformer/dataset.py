import torch
from torch.utils.data import Dataset

class TransformerDataset(Dataset):
    def __init__(self, tokens, block_size, stride=None):
        self.tokens = tokens
        self.block_size = block_size
        self.stride = stride or block_size // 2
        
        # Memory efficient: sadece start pozisyonlarını sakla
        self.sequences = []
        max_start = len(tokens) - block_size
        
        # Stride ile overlapping sequences oluştur
        for i in range(0, max_start, self.stride):
            if i + block_size < len(tokens):
                self.sequences.append(i)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        start_pos = self.sequences[idx]
        input_seq = self.tokens[start_pos:start_pos + self.block_size]
        target_seq = self.tokens[start_pos + 1:start_pos + self.block_size + 1]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)