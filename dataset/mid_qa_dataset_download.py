
import pandas as pd
import re

df = pd.read_parquet("random-qa/tr_qa_f.parquet")

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

df[["question", "answer"]] = df["QA"].apply(parse_qa)

df_clean = df[["question", "answer"]].dropna()

df_clean.to_parquet("qa_clean.parquet", index=False)

print("Yeni question-answer dataset oluşturuldu: qa_clean.parquet")

df_loaded = pd.read_parquet("qa_clean.parquet")
print("\n--- QA Clean Dataset Örneği ---")
print(df_loaded.head())
