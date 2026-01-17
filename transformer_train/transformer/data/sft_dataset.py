"""
SFT Dataset with Loss Masking Support

This dataset handles conversation-format data with proper loss masking.
Only assistant responses are supervised (mask=1), user messages are not (mask=0).
"""

import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """SFT dataset with loss masking for conversation training"""
    
    def __init__(self, conversations, tokenizer, block_size):
        """
        Args:
            conversations: List of conversation dicts with 'messages' key
            tokenizer: Tokenizer with render_conversation method
            block_size: Maximum sequence length
        """
        self.data = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        for conv in conversations:
            # render_conversation returns (ids, mask)
            ids, mask = tokenizer.render_conversation(conv)
            
            # Pad or truncate to block_size
            if len(ids) >= block_size:
                ids = ids[:block_size]
                mask = mask[:block_size]
            else:
                # Pad with zeros if needed
                pad_len = block_size - len(ids)
                ids = ids + [0] * pad_len
                mask = mask + [0] * pad_len
            
            self.data.append((ids, mask))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ids, mask = self.data[idx]
        
        # Create input/target pairs (shift by 1)
        inputs = torch.tensor(ids[:-1], dtype=torch.long)
        targets = torch.tensor(ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(mask[1:], dtype=torch.float)
        
        return inputs, targets, loss_mask
