import os
import sys
import time
import argparse
import torch
from pathlib import Path

"""
python3 tokenizer_train/train_tokenizer.py
--data_dir dataset/base_data
--text_column text
--vocab_size 65536
--doc_cap 100000
--max_chars 50000000000
--output_dir tokenizer
--progress
"""

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
                   help='Maximum characters to train on (default: 10B, set <=0 for unlimited)')
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
parser.add_argument('--data_dir', type=str, default=None,
                   help='Alternative: directory containing parquet files (uses text_column)')
parser.add_argument('--text_column', type=str, default='text',
                   help='Text column name in parquet files (default: text)')
parser.add_argument('--progress', action='store_true', help='Show tqdm progress while iterating data')
args = parser.parse_args()

print(f"Tokenizer Training Configuration:")
print(f"  max_chars: {args.max_chars:,}")
print(f"  doc_cap: {args.doc_cap:,}")
print(f"  vocab_size: {args.vocab_size:,}")
print(f"  output_dir: {args.output_dir}")
print(f"  dataset: {args.dataset}")
print(f"  corpus_file: {args.corpus_file}")
print(f"  data_dir: {args.data_dir}")
print(f"  text_column: {args.text_column}")
print(f"  progress: {args.progress}")

# -----------------------------------------------------------------------------
# Text iterator

max_chars = args.max_chars
if max_chars is not None and max_chars <= 0:
    max_chars = float('inf')

def with_progress(iterable, desc="Documents"):
    if not args.progress:
        return iterable
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, smoothing=0.01)
    except ImportError:
        print("tqdm not installed; progress disabled. Install via `pip install tqdm`.")
        return iterable


if args.corpus_file:
    print(f"Using corpus file: {args.corpus_file}")
    text_iter = text_iterator_from_file(args.corpus_file, max_chars, args.doc_cap)
    text_iter = with_progress(text_iter, desc="Corpus lines")
elif args.data_dir:
    # Load local parquet files with HuggingFace datasets (streaming to save memory)
    from datasets import load_dataset

    def text_iterator_from_parquet_dir(data_dir: str, text_column: str, max_chars: int, doc_cap: int):
        import re
        import unicodedata
        data_files = {"train": str(Path(data_dir) / "*.parquet")}
        ds = load_dataset("parquet", data_files=data_files, split="train", streaming=True)

        def clean_text(text: str) -> str:
            text = unicodedata.normalize("NFKC", text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        nchars = 0
        for row in ds:
            raw = row.get(text_column, "")
            doc_text = clean_text(str(raw))

            if len(doc_text) < 20:
                continue
            if len(doc_text) > doc_cap:
                doc_text = doc_text[:doc_cap]

            nchars += len(doc_text)
            yield doc_text

            if max_chars != float('inf') and nchars > max_chars:
                print(f"Reached max_chars limit: {nchars:,} characters")
                break

    print(f"Using parquet directory: {args.data_dir} (column: {args.text_column})")
    text_iter = text_iterator_from_parquet_dir(args.data_dir, args.text_column, max_chars, args.doc_cap)
    text_iter = with_progress(text_iter, desc="Parquet docs")
else:
    print(f"Using HuggingFace dataset: {args.dataset}")
    text_iter = text_iterator_from_dataset(args.dataset, max_chars, args.doc_cap)
    text_iter = with_progress(text_iter, desc="HF docs")

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

# Test 1: Basic round-trip
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text, f"Round-trip test failed!\nOriginal: {test_text}\nDecoded: {decoded}"
print("\nRound-trip encoding test passed!")

# Test 2: BOS/EOS token functionality
bos_id = tokenizer.get_bos_token_id()
eos_id = tokenizer.get_eos_token_id()
print(f"\nSpecial Tokens:")
print(f"  BOS token ID: {bos_id} -> '{tokenizer.decode([bos_id])}'")
print(f"  EOS token ID: {eos_id} -> '{tokenizer.decode([eos_id])}'")

# Test 3: Encode with BOS and EOS
encoded_with_special = tokenizer.encode(test_text, prepend=bos_id, append=eos_id)
assert encoded_with_special[0] == bos_id, f"BOS token not at start! Got {encoded_with_special[0]}"
assert encoded_with_special[-1] == eos_id, f"EOS token not at end! Got {encoded_with_special[-1]}"
print(f"BOS/EOS prepend/append test passed!")
print(f"  Encoded length: {len(encoded_with_special)} (including BOS+EOS)")

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
