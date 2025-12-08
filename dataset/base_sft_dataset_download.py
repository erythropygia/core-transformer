from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
from tqdm import tqdm
import time

repo_id = "lumees/turkish-corpus-100b"
local_dir = "turkish_corpus_100b"

# 1) Repo’daki tüm dosyaların listesini al
api = HfApi()
files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

# Sadece parquet dosyalarını takip etmek istersen:
files = [f for f in files if f.endswith(".parquet")]

pbar = tqdm(total=len(files), desc="Downloading files", unit="file")

# 2) Zaten indirilmiş dosya sayısını başlangıçta güncelle
for f in files:
    if Path(local_dir, f).exists():
        pbar.update(1)

# 3) snapshot_download çalışırken periyodik olarak dosyaları kontrol et
# (kaldığı yerden devam ettiği için uyumludur)
def monitor():
    downloaded_prev = pbar.n
    while pbar.n < pbar.total:
        downloaded_now = sum(Path(local_dir, f).exists() for f in files)
        pbar.update(downloaded_now - downloaded_prev)
        downloaded_prev = downloaded_now
        time.sleep(1)  # diski çok yormaması için

# Background monitoring
import threading
t = threading.Thread(target=monitor, daemon=True)
t.start()

# 4) Download
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)

pbar.close()
