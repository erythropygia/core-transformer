"""
from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
from tqdm import tqdm
import time

repo_id = "Ba2han/Turkish-Random-QA_set"
local_dir = "random-qa"

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
"""

import pandas as pd
import re

# 1) Mevcut parquet dosyasını oku
df = pd.read_parquet("random-qa/tr_qa_f.parquet")

# 2) QA kolonundan Soru ve Cevap ayıklama fonksiyonu
def parse_qa(text):
    soru_pattern = r"Soru:\s*(.*?)(?=Cevap:)"
    cevap_pattern = r"Cevap:\s*(.*)"

    soru = re.search(soru_pattern, text, flags=re.S)
    cevap = re.search(cevap_pattern, text, flags=re.S)

    if soru and cevap:
        question = soru.group(1).strip()
        answer = cevap.group(1).strip()
        return pd.Series([question, answer])
    else:
        return pd.Series([None, None])

# 3) QA kolonuna uygula
df[["question", "answer"]] = df["QA"].apply(parse_qa)

# 4) sadece question-answer kolonlarını içeren yeni dataset
df_clean = df[["question", "answer"]].dropna()

# 5) parquet olarak kaydet
df_clean.to_parquet("qa_clean.parquet", index=False)

print("Yeni question-answer dataset oluşturuldu: qa_clean.parquet")

# 6) kaydedilen dataset'i tekrar oku ve göster
df_loaded = pd.read_parquet("qa_clean.parquet")
print("\n--- QA Clean Dataset Örneği ---")
print(df_loaded.head())
