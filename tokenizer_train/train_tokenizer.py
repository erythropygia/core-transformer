import os
import sys
import time
import argparse
import torch
from pathlib import Path

# Add parent directory to path so we can import transformer_train
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import RustBPE tokenizer
from transformer_train.transformer.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
def text_iterator_from_dataset(dataset_name="musabg/wikipedia-tr-summarization", max_chars=10_000_000_000, doc_cap=10_000):
    from datasets import load_dataset
    from tqdm import tqdm
    import unicodedata
    import re
    
    def clean_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split='train')
    print(f"Dataset size: {len(dataset):,} documents")
    
    nchars = 0
    for i, item in enumerate(tqdm(dataset, desc="Processing documents")):
        # Get text from dataset item
        if isinstance(item, dict):
            text = item.get("text", item.get("content", ""))
        else:
            text = str(item)
        
        doc_text = clean_text(text)
        
        # Crop to doc_cap
        if len(doc_text) > doc_cap:
            doc_text = doc_text[:doc_cap]
        
        # Skip very short documents
        if len(doc_text) < 20:
            continue
        
        nchars += len(doc_text)
        yield doc_text
        
        if nchars > max_chars:
            print(f"Reached max_chars limit: {nchars:,} characters")
            break


def text_iterator_from_file(corpus_file, max_chars=10_000_000_000, doc_cap=10_000):
    nchars = 0
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc_text = line.strip()
            if len(doc_text) < 20:  # Skip very short lines
                continue
            
            # Crop to doc_cap
            if len(doc_text) > doc_cap:
                doc_text = doc_text[:doc_cap]
            
            nchars += len(doc_text)
            yield doc_text
            
            if nchars > max_chars:
                print(f"Reached max_chars limit: {nchars:,} characters")
                break


# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer using RustBPE')
parser.add_argument('--max_chars', type=int, default=10_000_000_000, 
                   help='Maximum characters to train on (default: 10B)')
parser.add_argument('--doc_cap', type=int, default=10_000, 
                   help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab_size', type=int, default=65536, 
                   help='Vocabulary size (default: 65536 = 2^16)')
parser.add_argument('--output_dir', type=str, default='tokenizer', 
                   help='Output directory for tokenizer (default: tokenizer)')
parser.add_argument('--dataset', type=str, default='musabg/wikipedia-tr-summarization',
                   help='HuggingFace dataset name (default: musabg/wikipedia-tr-summarization)')
parser.add_argument('--corpus_file', type=str, default=None,
                   help='Alternative: path to text file (one document per line)')
args = parser.parse_args()

print(f"Tokenizer Training Configuration:")
print(f"  max_chars: {args.max_chars:,}")
print(f"  doc_cap: {args.doc_cap:,}")
print(f"  vocab_size: {args.vocab_size:,}")
print(f"  output_dir: {args.output_dir}")
print(f"  dataset: {args.dataset}")
print(f"  corpus_file: {args.corpus_file}")

# -----------------------------------------------------------------------------
# Text iterator

if args.corpus_file:
    print(f"Using corpus file: {args.corpus_file}")
    text_iter = text_iterator_from_file(args.corpus_file, args.max_chars, args.doc_cap)
else:
    print(f"Using HuggingFace dataset: {args.dataset}")
    text_iter = text_iterator_from_dataset(args.dataset, args.max_chars, args.doc_cap)

# -----------------------------------------------------------------------------
# Train the tokenizer

print("\nStarting tokenizer training...")
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training completed in {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
tokenizer.save(str(output_dir))
print(f"Tokenizer saved to: {output_dir}")

# -----------------------------------------------------------------------------
# Quick inline sanity check

test_text = """Merhaba dünya! Bu bir test.
Türkiye'nin başkenti Ankara'dır.
Yapay zeka teknolojisi hızla gelişiyor.
Numbers: 123, 4567, 89
Special chars: @#$%^&*()"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text, f"Round-trip test failed!\nOriginal: {test_text}\nDecoded: {decoded}"
print("\nRound-trip encoding test passed!")

# -----------------------------------------------------------------------------
# Save token bytes for bits-per-byte evaluation

print("\nComputing token bytes mapping...")
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id]  # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0)  # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8"))  # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = output_dir / "token_bytes.pt"
torch.save(token_bytes, token_bytes_path)
print(f"Saved token_bytes to {token_bytes_path}")

# Print statistics
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
print(f"\nToken bytes statistics:")
print(f"  min: {int(token_bytes_nonzero.min().item())}")
print(f"  max: {int(token_bytes_nonzero.max().item())}")
print(f"  mean: {token_bytes_nonzero.mean().item():.2f}")
print(f"  std: {token_bytes_nonzero.std().item():.2f}")
print(f"  num_special_tokens: {len(special_set)}")
print(f"  vocab_size: {vocab_size:,}")

print(f"\nTokenizer training completed successfully!")
print(f"  Output directory: {output_dir}")
print(f"  Training time: {train_time:.2f}s")

# Log to report
try:
    from transformer_train.transformer.report import get_report
    get_report().log(section="Tokenizer training", data=[
        vars(args), # argparse command line arguments
        {"train_time": train_time},
        {"num_special_tokens": len(special_set)},
        {"vocab_size": vocab_size},
    ])
except Exception as e:
    print(f"Warning: Could not log to report: {e}")
