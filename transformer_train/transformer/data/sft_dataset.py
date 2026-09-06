import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, conversations, tokenizer, block_size):
        self.data = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_id = tokenizer.get_pad_token_id()

        for conv in conversations:
            ids, mask = tokenizer.render_conversation(conv, max_tokens=block_size)
            if len(ids) < 2 or sum(mask) == 0:
                continue
            self.data.append((ids, mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids, mask = self.data[idx]
        inputs = torch.tensor(ids[:-1], dtype=torch.long)
        targets = torch.tensor(ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(mask[1:], dtype=torch.float)
        return inputs, targets, loss_mask


def sft_collate(batch, pad_id):
    max_len = max(x[0].size(0) for x in batch)
    inputs, targets, masks = [], [], []
    for inp, tgt, msk in batch:
        pad = max_len - inp.size(0)
        if pad:
            inputs.append(torch.cat([inp, torch.full((pad,), pad_id, dtype=torch.long)]))
            targets.append(torch.cat([tgt, torch.full((pad,), pad_id, dtype=torch.long)]))
            masks.append(torch.cat([msk, torch.zeros(pad, dtype=torch.float)]))
        else:
            inputs.append(inp)
            targets.append(tgt)
            masks.append(msk)
    return torch.stack(inputs), torch.stack(targets), torch.stack(masks)
