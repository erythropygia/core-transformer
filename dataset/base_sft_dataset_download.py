from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
from tqdm import tqdm
import time

repo_id = "lumees/turkish-corpus-100b"
local_dir = "turkish_corpus_100b"

api = HfApi()
files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

files = [f for f in files if f.endswith(".parquet")]

pbar = tqdm(total=len(files), desc="Downloading files", unit="file")

for f in files:
    if Path(local_dir, f).exists():
        pbar.update(1)

def monitor():
    downloaded_prev = pbar.n
    while pbar.n < pbar.total:
        downloaded_now = sum(Path(local_dir, f).exists() for f in files)
        pbar.update(downloaded_now - downloaded_prev)
        downloaded_prev = downloaded_now
        time.sleep(1)

import threading
t = threading.Thread(target=monitor, daemon=True)
t.start()

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)

pbar.close()
