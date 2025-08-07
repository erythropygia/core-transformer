import torch
from torch.utils.data import Dataset
import re
from tqdm import tqdm
from datasets import load_dataset
import unicodedata

from .utils import cleanup_memory

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
    

def load_and_preprocess_data(max_samples=150000):
    def clean_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    dataset = load_dataset("musabg/wikipedia-tr-summarization", split='train')
    processed_texts = []
    
    print(f"Dataset total size: {len(dataset):,}")
    
    max_samples = min(len(dataset), max_samples)
    
    for i in tqdm(range(max_samples), desc="Processing texts"):
        text = clean_text(dataset[i]["text"])
        if len(text) > 100:  # Longer texts for better training
            processed_texts.append(text)
        
        if i % 10000 == 0:
            cleanup_memory()
    
    print(f"Processed {len(processed_texts):,} texts")
    return processed_texts