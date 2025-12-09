# Core-Transformer Eğitim Rehberi

Bu rehber, sıfırdan başlayarak model eğitimi için adım adım talimatlar içerir.

## Genel Bakış

Eğitim pipeline'ı şu aşamalardan oluşur:
1. **Tokenizer Eğitimi** - Veri tokenize etmek için tokenizer'ı eğit
2. **Base Training** - Büyük metin verisi üzerinde pretraining
3. **Mid Training** (Opsiyonel) - Yapılandırılmış görevler (Wikipedia, QA, vb.)
4. **SFT Training** - Chat formatında fine-tuning
5. **RL Training** (Opsiyonel) - Reinforcement learning ile iyileştirme
6. **Evaluation** - Model performansını değerlendir

---

## Adım 1: Veri Hazırlığı

### 1.1 Base Training Verisi (Parquet Formatı)

Base training için parquet dosyaları hazırla:

```bash
# Örnek: base_data/ klasörüne parquet dosyalarını yerleştir
# Her parquet dosyası 'text' kolonuna sahip olmalı
mkdir -p base_data
# Parquet dosyalarını buraya kopyala
```

**Parquet dosya formatı:**
- Her dosya bir `text` kolonu içermeli
- Dosyalar `base_data/` klasöründe olmalı
- Örnek: `base_data/chunk_0001.parquet`, `base_data/chunk_0002.parquet`, vb.

### 1.2 Config Ayarları

`transformer_train/transformer/config.py` dosyasını kontrol et:

```python
# Base training için
TRAINING_CONFIG = {
    'training_stage': 'base',  # 'base', 'mid', veya 'sft'
    # ... diğer ayarlar
}

BASE_DATASET_CONFIG = {
    'type': 'parquet',
    'data_dir': 'base_data',  # Parquet dosyalarının bulunduğu klasör
    'text_column': 'text',
}
```

---

## Adım 2: Tokenizer Eğitimi

### 2.1 Tokenizer Eğitimi

```bash
# Tokenizer'ı eğit
cd core-transformer
python tokenizer_train/train_tokenizer.py \
    --data_dir base_data \
    --output_dir tokenizer \
    --vocab_size 32000
```

**Parametreler:**
- `--data_dir`: Parquet dosyalarının bulunduğu klasör
- `--output_dir`: Tokenizer'ın kaydedileceği klasör (varsayılan: `tokenizer`)
- `--vocab_size`: Vocabulary boyutu (önerilen: 32000)

**Çıktı:**
- `tokenizer/tokenizer.pkl` - Eğitilmiş tokenizer
- `tokenizer/token_bytes.pt` - Token byte sayıları (evaluation için)

### 2.2 Tokenizer Testi

```bash
# Tokenizer'ı test et
python tokenizer_train/test_tokenizer.py
```

---

## Adım 3: Base Model Eğitimi (Pretraining)

### 3.1 Config Kontrolü

`transformer_train/transformer/config.py`:

```python
TRAINING_CONFIG = {
    'training_stage': 'base',  # Base training için
    'max_steps': 100000,  # Toplam adım sayısı
    'batch_size': 32,  # Per-device batch size
    'learning_rate': 6e-4,
    'warmup_steps': 2000,
    # ... diğer hyperparameters
}

BASE_DATASET_CONFIG = {
    'type': 'parquet',
    'data_dir': 'base_data',
    'text_column': 'text',
}
```

### 3.2 Base Training Başlat

**Single GPU:**
```bash
python transformer_train/run_train.py
```

**Multi-GPU (8 GPU örneği):**
```bash
torchrun --standalone --nproc_per_node=8 transformer_train/run_train.py
```

**Checkpoint'ler:**
- Checkpoint'ler `checkpoints/` klasörüne kaydedilir
- Format: `checkpoint_step_XXXXXX.safetensors`

### 3.3 Base Model Evaluation

Eğitim sırasında veya sonrasında modeli değerlendir:

**CORE Metric (Base Model):**
```bash
# CORE metrik için (eval_bundle.zip gerekir, ~162MB)
python transformer_train/scripts/base_eval.py --checkpoint checkpoints/checkpoint_step_100000.safetensors
```

**Bits Per Byte (Loss Evaluation):**
```bash
# Bits per byte metrik
python transformer_train/scripts/base_loss.py
```

**Text Generation Test:**
```bash
# Model'den text üret
python transformer_train/generate_text.py \
    --checkpoint checkpoints/checkpoint_step_100000.safetensors \
    --prompt "Türkiye'nin başkenti"
```

---

## Adım 4: Mid Training (Opsiyonel)

Mid training, yapılandırılmış görevler üzerinde eğitim yapar (Wikipedia, QA, matematik, vb.).

### 4.1 Config Ayarları

```python
TRAINING_CONFIG = {
    'training_stage': 'mid',  # Mid training için
    'max_steps': 10000,
    # ... diğer ayarlar
}

MID_DATASET_CONFIG = {
    'datasets': [
        {'type': 'wikipedia', 'split': 'train', 'max_samples': 10000},
        {'type': 'news', 'split': 'train', 'max_samples': 5000},
        {'type': 'qa', 'split': 'train', 'max_samples': 2000},
        {'type': 'math', 'split': 'train', 'max_samples': 1000},
    ]
}
```

**Not:** `dataset_utils.py` içindeki placeholder fonksiyonları gerçek dataset loader'larla değiştirmen gerekebilir.

### 4.2 Mid Training Başlat

```bash
# Config'de training_stage='mid' olduğundan emin ol
python transformer_train/run_train.py \
    --resume_from checkpoints/checkpoint_step_100000.safetensors
```

---

## Adım 5: SFT Training (Chat Fine-tuning)

SFT, modeli chat formatında kullanmak için fine-tune eder.

### 5.1 Config Ayarları

```python
TRAINING_CONFIG = {
    'training_stage': 'sft',  # SFT için
    'max_steps': 5000,
    'learning_rate': 1e-5,  # Daha düşük learning rate
    # ... diğer ayarlar
}

SFT_DATASET_CONFIG = {
    'datasets': [
        {'type': 'chat', 'split': 'train', 'max_samples': 5000},
    ]
}
```

**Not:** Chat dataset'leri conversation formatında olmalı:
```python
{
    "messages": [
        {"role": "user", "content": "Merhaba"},
        {"role": "assistant", "content": "Merhaba! Size nasıl yardımcı olabilirim?"}
    ]
}
```

### 5.2 SFT Training Başlat

```bash
# Config'de training_stage='sft' olduğundan emin ol
python transformer_train/run_train.py \
    --resume_from checkpoints/checkpoint_step_100000.safetensors
```

### 5.3 Chat Evaluation

```bash
# Chat modelini değerlendir
python transformer_train/scripts/chat_eval.py \
    --checkpoint checkpoints/checkpoint_step_5000.safetensors \
    --task GSM8K \
    --num-samples 1 \
    --max-new-tokens 256
```

**Mevcut task'lar:**
- `GSM8K` - Matematik problemleri
- `ARC-Easy` - Mantıksal akıl yürütme (kolay)
- `ARC-Challenge` - Mantıksal akıl yürütme (zor)
- `MMLU` - Çoklu seçim soruları

---

## Adım 6: RL Training (Opsiyonel)

RL training, GSM8K gibi görevlerde performansı artırmak için kullanılır.

### 6.1 RL Training Başlat

```bash
python transformer_train/scripts/chat_rl.py \
    --checkpoint checkpoints/checkpoint_step_5000.safetensors \
    --run gsm8k_rl \
    --num-samples 16 \
    --examples-per-step 16 \
    --num-epochs 1
```

**Parametreler:**
- `--checkpoint`: SFT checkpoint'i
- `--run`: WandB run adı (veya 'dummy' logging için)
- `--num-samples`: Her örnek için sample sayısı
- `--examples-per-step`: Her adımda işlenecek örnek sayısı
- `--num-epochs`: Epoch sayısı

**Checkpoint'ler:**
- RL checkpoint'leri `chatrl_checkpoints/` klasörüne kaydedilir

---

## Adım 7: Final Evaluation

### 7.1 Base Model Metrics

```bash
# CORE metric
python transformer_train/scripts/base_eval.py \
    --checkpoint checkpoints/checkpoint_step_100000.safetensors

# Bits per byte
python transformer_train/scripts/base_loss.py
```

### 7.2 Chat Model Metrics

```bash
# Tüm task'ları değerlendir
python transformer_train/scripts/chat_eval.py \
    --checkpoint checkpoints/checkpoint_step_5000.safetensors \
    --task GSM8K

python transformer_train/scripts/chat_eval.py \
    --checkpoint checkpoints/checkpoint_step_5000.safetensors \
    --task ARC-Easy

python transformer_train/scripts/chat_eval.py \
    --checkpoint checkpoints/checkpoint_step_5000.safetensors \
    --task MMLU
```

### 7.3 Interactive Chat

```bash
# Chat modunda kullan
python transformer_train/generate_text.py \
    --checkpoint checkpoints/checkpoint_step_5000.safetensors \
    --chat
```

---

## Resume Training (Eğitimi Devam Ettirme)

Eğitim yarıda kesilirse, checkpoint'ten devam edebilirsin:

```bash
python transformer_train/run_train.py \
    --resume_from checkpoints/checkpoint_step_50000.safetensors
```

**Not:** `train.py` içinde `dataloader_state_dict` ile approximate resume desteği var.

---

## 📝 Özet: Tam Pipeline

```bash
# 1. Tokenizer eğitimi
python tokenizer_train/train_tokenizer.py --data_dir base_data --vocab_size 32000

# 2. Base training (100k steps)
# config.py: training_stage='base'
python transformer_train/run_train.py  # veya torchrun ile multi-GPU

# 3. Base evaluation
python transformer_train/scripts/base_eval.py --checkpoint checkpoints/checkpoint_step_100000.safetensors

# 4. Mid training (opsiyonel, 10k steps)
# config.py: training_stage='mid'
python transformer_train/run_train.py --resume_from checkpoints/checkpoint_step_100000.safetensors

# 5. SFT training (5k steps)
# config.py: training_stage='sft'
python transformer_train/run_train.py --resume_from checkpoints/checkpoint_step_100000.safetensors

# 6. Chat evaluation
python transformer_train/scripts/chat_eval.py --checkpoint checkpoints/checkpoint_step_5000.safetensors --task GSM8K

# 7. RL training (opsiyonel)
python transformer_train/scripts/chat_rl.py --checkpoint checkpoints/checkpoint_step_5000.safetensors

# 8. Final evaluation
python transformer_train/scripts/chat_eval.py --checkpoint chatrl_checkpoints/step_XXXXXX/model_step_XXXXXX.safetensors --task GSM8K
```

---

## Önemli Notlar

1. **Config Yönetimi:** Her training stage için `config.py`'de `training_stage` değerini değiştir
2. **Checkpoint Yolu:** Her aşamada doğru checkpoint'i kullan
3. **Dataset Formatı:** Base için parquet, SFT için conversation formatı
4. **Memory:** Büyük modeller için gradient accumulation kullan
5. **Distributed:** Multi-GPU için `torchrun` kullan

---

## Sorun Giderme

**Token Bytes Bulunamadı:**
```bash
# Tokenizer eğitimi sırasında token_bytes.pt oluşturulmalı
# Eğer yoksa, tokenizer_train/train_tokenizer.py'yi kontrol et
```

**Checkpoint Yüklenemiyor:**
```bash
# Checkpoint path'ini kontrol et
# Metadata'da tokenizer_path doğru mu?
```

**Out of Memory:**
```bash
# Batch size'ı azalt veya gradient accumulation kullan
# config.py: 'batch_size': 16, 'gradient_accumulation_steps': 2
```

---

## İlgili Dosyalar

- `transformer_train/transformer/config.py` - Tüm config ayarları
- `transformer_train/transformer/train.py` - Ana training loop
- `transformer_train/run_train.py` - Training script
- `tokenizer_train/train_tokenizer.py` - Tokenizer eğitimi
- `transformer_train/scripts/` - Evaluation script'leri

