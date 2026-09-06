

import pandas as pd

train_df = pd.read_parquet("news/data/train-00000-of-00001.parquet")
val_df = pd.read_parquet("news/data/validation-00000-of-00001.parquet")

train_df_new = train_df[["Kategori", "Link", "Icerik"]].rename(
    columns={"Kategori": "category", "Link": "link", "Icerik": "text"}
)

val_df_new = val_df[["Kategori", "Link", "Icerik"]].rename(
    columns={"Kategori": "category", "Link": "link", "Icerik": "text"}
)

combined_df = pd.concat([train_df_new, val_df_new], ignore_index=True)

combined_df.to_parquet("text_dataset.parquet", index=False)

print("Yeni parquet dataset oluşturuldu: text_dataset.parquet")

df_loaded = pd.read_parquet("text_dataset.parquet")
print(df_loaded.head())
