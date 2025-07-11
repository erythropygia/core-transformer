import os
import json
import time
import logging
import argparse
import sentencepiece as spm
from pathlib import Path
from typing import List, Dict, Optional

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tokenizer_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TurkishTokenizerTrainer:
    def __init__(self, 
                 vocab_size: int = 32000,
                 model_type: str = 'bpe',
                 character_coverage: float = 0.9995,
                 output_dir: str = 'turkish_tokenizer'):
        """
        130M+ parametre modeller için optimize edilmiş Türkçe tokenizer eğiticisi
        
        Args:
            vocab_size: Vocabulary boyutu (32K, 130M model için optimal)
            model_type: Model tipi ('bpe' önerilen)
            character_coverage: Karakter kapsamı (Türkçe için 0.9995)
            output_dir: Çıktı dizini
        """
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Türkçe için optimize edilmiş SentencePiece parametreleri
        self.sp_params = {
            'model_type': model_type,
            'vocab_size': vocab_size,
            'character_coverage': character_coverage,
            'normalization_rule_name': 'nfkc',  # Türkçe karakterler için
            'remove_extra_whitespaces': True,
            'split_by_unicode_script': True,    # Türkçe morfoloji için kritik
            'split_by_whitespace': True,
            'split_by_number': True,
            'split_digits': True,
            'treat_whitespace_as_suffix': False,
            'allow_whitespace_only_pieces': True,
            'max_sentence_length': 8192,
            'shuffle_input_sentence': True,
            'input_sentence_size': 10000000,    # 10M cümle limit
            'seed_sentencepiece_size': 1000000,
            'shrinking_factor': 0.75,
            'num_threads': os.cpu_count() or 4,
            'max_sentencepiece_length': 16,
            'num_sub_iterations': 2,
            # Token ID'leri açık şekilde tanımla
            'unk_id': 0,    # <unk>
            'bos_id': 1,    # <s> 
            'eos_id': 2,    # </s>
            'pad_id': 3,    # <pad>
        }
        
        # Minimal ama gelecek odaklı özel token seti
        # Total: 16 özel token (vocab'un %0.05'i)
        self.special_tokens = [
            # === Core Functionality (5 tokens) ===
            '<mask>',               # Masked Language Modeling
            '<turkish>',            # Language identifier
            '<instruction>',        # Instruction tuning başlangıç
            '</instruction>',       # Instruction tuning bitiş
            '<context>',           # Context marker
            
            # === Chat & Dialog (4 tokens) ===
            '<|system|>',          # System prompt
            '<|user|>',            # User message  
            '<|assistant|>',       # Assistant response
            '<|end|>',             # Turn end marker
            
            # === Safety & Control (2 tokens) ===
            '<safe>',              # Safe content marker
            '<unsafe>',            # Unsafe content marker
            
            # === Future Reserve (5 tokens) ===
            '<reserved1>',         # Gelecek özellikler için
            '<reserved2>',         # Function calling, tools vs.
            '<reserved3>',         # Multi-modal extensions
            '<reserved4>',         # Task-specific needs
            '<reserved5>',         # Emergency reserve
        ]

    def get_corpus_file(self) -> str:
        """
        Mevcut OSCAR corpus dosyasını kontrol et ve döndür
        
        Returns:
            Corpus dosyasının yolu
        """
        script_dir = Path(__file__).parent
        corpus_file = script_dir / 'data' / 'oscar_turkish.txt'
        
        if not corpus_file.exists():
            raise FileNotFoundError(f"OSCAR corpus dosyası bulunamadı: {corpus_file}")
        
        file_size = corpus_file.stat().st_size / (1024*1024*1024)  # GB
        logger.info(f"✅ OSCAR corpus bulundu: {corpus_file}")
        logger.info(f"📊 Dosya boyutu: {file_size:.2f} GB")
        
        return str(corpus_file)

    def preprocess_corpus(self, corpus_file: str) -> str:
        """
        Corpus dosyasını önişlemden geçir - 130M model için optimize edilmiş
        
        Args:
            corpus_file: Girdi corpus dosyası
            
        Returns:
            Önişlenmiş corpus dosyasının yolu
        """
        logger.info("🔄 Corpus önişleme başlıyor...")
        
        processed_file = self.output_dir / 'processed_corpus.txt'
        total_lines = 0
        valid_lines = 0
        
        with open(processed_file, 'w', encoding='utf-8') as outfile:
            with open(corpus_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    total_lines += 1
                    line = line.strip()
                    
                    # Katı filtreleme - 130M model için kaliteli veri
                    if self._is_high_quality_text(line):
                        outfile.write(line + '\n')
                        valid_lines += 1
                    
                    if total_lines % 100000 == 0:
                        logger.info(f"  📝 {total_lines:,} satır işlendi, {valid_lines:,} geçerli satır")
                        
                    # Memory efficiency için batch processing
                    if total_lines % 1000000 == 0:
                        logger.info(f"  💾 {total_lines//1000000}M satır tamamlandı")
        
        retention_rate = (valid_lines / total_lines) * 100
        logger.info(f"✅ Önişleme tamamlandı:")
        logger.info(f"   📊 {valid_lines:,}/{total_lines:,} satır korundu (%{retention_rate:.1f})")
        return str(processed_file)

    def _is_high_quality_text(self, text: str) -> bool:
        """130M model için yüksek kaliteli metin kontrolü"""
        text = text.strip()
        
        # Minimum uzunluk - çok kısa cümleler işe yaramaz
        if len(text) < 20:
            return False
        
        # Maximum uzunluk - çok uzun cümleler problematik
        if len(text) > 512:
            return False
        
        # Türkçe karakter oranı kontrolü
        turkish_chars = sum(1 for c in text if c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZçğıöşüÇĞIİÖŞÜ')
        if len(text) > 0:
            turkish_ratio = turkish_chars / len(text)
            if turkish_ratio < 0.7:  # En az %70 harf olmalı
                return False
        
        # Sayı yoğunluğu kontrolü
        digit_ratio = sum(c.isdigit() for c in text) / len(text)
        if digit_ratio > 0.3:  # %30'dan fazla sayı olmasın
            return False
        
        # Tekrar eden karakter kontrolü (spam filtreleme)
        if any(char * 4 in text for char in set(text)):
            return False
        
        # Büyük harf yoğunluğu (SPAM filtreleme)
        upper_ratio = sum(c.isupper() for c in text) / len(text)
        if upper_ratio > 0.5:  # %50'den fazla büyük harf
            return False
            
        return True

    def train_tokenizer(self, corpus_file: str) -> Dict[str, str]:
        """
        130M+ model için optimize edilmiş SentencePiece tokenizer eğit
        
        Args:
            corpus_file: Eğitim korpusu dosyası
            
        Returns:
            Eğitilen modelin dosya yolları
        """
        logger.info("🚀 SentencePiece tokenizer eğitimi başlıyor...")
        logger.info(f"🎯 Target: 130M parametre model için optimize edilmiş tokenizer")
        
        model_prefix = str(self.output_dir / 'turkish_tokenizer')
        
        # SentencePiece eğitim parametrelerini hazırla
        sp_params = self.sp_params.copy()
        sp_params.update({
            'input': corpus_file,
            'model_prefix': model_prefix,
            'user_defined_symbols': ','.join(self.special_tokens),
        })
        
        # Kritik parametreleri logla
        logger.info("📋 Eğitim parametreleri:")
        logger.info(f"  📝 Vocabulary boyutu: {sp_params['vocab_size']:,}")
        logger.info(f"  🔤 Model tipi: {sp_params['model_type'].upper()}")
        logger.info(f"  🎯 Karakter kapsamı: {sp_params['character_coverage']}")
        logger.info(f"  ⭐ Özel token sayısı: {len(self.special_tokens)}")
        logger.info(f"  🧵 Thread sayısı: {sp_params['num_threads']}")
        
        try:
            # SentencePiece modelini eğit
            logger.info("🔥 SentencePiece eğitimi başlıyor...")
            start_time = time.time()
            
            spm.SentencePieceTrainer.train(**sp_params)
            
            training_time = time.time() - start_time
            logger.info(f"✅ Tokenizer eğitimi tamamlandı! Süre: {training_time:.2f}s")
            
            # Çıktı dosyalarını kontrol et
            model_file = f"{model_prefix}.model"
            vocab_file = f"{model_prefix}.vocab"
            
            if os.path.exists(model_file) and os.path.exists(vocab_file):
                logger.info(f"📁 Model dosyası: {model_file}")
                logger.info(f"📁 Vocabulary dosyası: {vocab_file}")
                
                # Model kalitesini analiz et
                self._analyze_tokenizer_quality(model_file)
                
                return {
                    'model': model_file,
                    'vocab': vocab_file,
                    'training_time': training_time
                }
            else:
                raise FileNotFoundError("Model dosyaları oluşturulamadı")
                
        except Exception as e:
            logger.error(f"❌ Tokenizer eğitimi başarısız: {e}")
            raise

    def _analyze_tokenizer_quality(self, model_file: str) -> None:
        """Tokenizer kalitesini analiz et ve raporla"""
        try:
            sp = spm.SentencePieceProcessor()
            sp.load(model_file)
            
            logger.info("📊 Tokenizer Kalite Analizi:")
            logger.info(f"  📝 Toplam vocabulary: {sp.vocab_size():,}")
            
            # Özel tokenları kontrol et
            special_count = 0
            special_ids = []
            for token in self.special_tokens:
                token_id = sp.piece_to_id(token)
                if token_id != sp.unk_id():
                    special_count += 1
                    special_ids.append(f"{token}:{token_id}")
            
            logger.info(f"  ⭐ Özel tokenlar: {special_count}/{len(self.special_tokens)} başarılı")
            
            # Temel tokenları kontrol et
            core_tokens = {
                '<unk>': sp.unk_id(),
                '<s>': sp.bos_id(), 
                '</s>': sp.eos_id(),
                '<pad>': sp.pad_id()
            }
            
            logger.info("  🔧 Core tokenlar:")
            for token, token_id in core_tokens.items():
                logger.info(f"     {token}: ID {token_id}")
            
            # Türkçe test cümleleri
            test_sentences = [
                "Merhaba, nasılsınız?",
                "Türkiye'nin başkenti Ankara'dır.",
                "Yapay zeka teknolojisi hızla gelişiyor."
            ]
            
            logger.info("  🧪 Tokenizasyon testi:")
            for sentence in test_sentences:
                tokens = sp.encode(sentence, out_type=str)
                token_count = len(tokens)
                logger.info(f"     '{sentence}' -> {token_count} token")
            
            # Efficiency score (ortalama token/karakter oranı)
            total_chars = sum(len(s) for s in test_sentences)
            total_tokens = sum(len(sp.encode(s)) for s in test_sentences)
            efficiency = total_chars / total_tokens if total_tokens > 0 else 0
            
            logger.info(f"  ⚡ Encoding efficiency: {efficiency:.2f} karakter/token")
            
            if efficiency >= 3.0:
                logger.info("     ✅ Excellent efficiency for Turkish!")
            elif efficiency >= 2.5:
                logger.info("     ✅ Good efficiency")
            else:
                logger.info("     ⚠️  Low efficiency - consider retraining")
                
        except Exception as e:
            logger.warning(f"Kalite analizi yapılamadı: {e}")

    def save_config(self, result: Dict[str, str]) -> str:
        """Tokenizer konfigürasyonunu kaydet"""
        config = {
            'model_info': {
                'name': 'Turkish SentencePiece Tokenizer',
                'version': '1.0',
                'target_model_size': '130M+',
                'language': 'Turkish',
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'parameters': {
                'vocab_size': self.vocab_size,
                'model_type': self.model_type,
                'character_coverage': self.character_coverage,
            },
            'special_tokens': {
                'count': len(self.special_tokens),
                'tokens': self.special_tokens
            },
            'files': result,
            'usage': {
                'loading': f"sp.load('{result['model']}')",
                'encoding': "tokens = sp.encode('metin', out_type=str)",
                'decoding': "text = sp.decode(tokens)"
            }
        }
        
        config_file = self.output_dir / 'tokenizer_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Konfigürasyon kaydedildi: {config_file}")
        return str(config_file)

    def run_training(self) -> Dict[str, str]:
        """Tam eğitim sürecini çalıştır"""
        logger.info("🚀 Türkçe Tokenizer Eğitimi Başlıyor!")
        logger.info("🎯 130M+ Parametre Modeller İçin Optimize Edilmiş")
        logger.info("=" * 60)
        
        try:
            # 1. Corpus dosyasını kontrol et
            logger.info("📊 1. Aşama: Corpus dosyası kontrol ediliyor...")
            corpus_file = self.get_corpus_file()
            
            # 2. Corpus'u önişle
            logger.info("🔄 2. Aşama: Yüksek kaliteli corpus hazırlanıyor...")
            processed_corpus = self.preprocess_corpus(corpus_file)
            
            # 3. Tokenizer'ı eğit
            logger.info("🎓 3. Aşama: Tokenizer eğitimi...")
            result = self.train_tokenizer(processed_corpus)
            
            # 4. Konfigürasyonu kaydet
            logger.info("💾 4. Aşama: Konfigürasyon kaydediliyor...")
            config_file = self.save_config(result)
            result['config'] = config_file
            
            # 5. Özet bilgi
            logger.info("🎉 Eğitim Başarıyla Tamamlandı!")
            logger.info("=" * 60)
            logger.info(f"📁 Model dosyası: {result['model']}")
            logger.info(f"📁 Vocabulary dosyası: {result['vocab']}")  
            logger.info(f"📄 Konfigürasyon: {result['config']}")
            logger.info(f"⏱️  Toplam eğitim süresi: {result['training_time']:.2f}s")
            logger.info("")
            logger.info("🚀 130M model için hazır! Test etmek için:")
            logger.info("   python test_turkish_tokenizer.py")
            logger.info("")
            logger.info("💡 Gelecek genişletmeler için 5 rezerve token hazır!")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Eğitim başarısız: {e}")
            raise

def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description="130M+ Parametre Modeller İçin Türkçe Tokenizer Eğiticisi"
    )
    parser.add_argument('--vocab-size', type=int, default=32000, 
                       help='Vocabulary boyutu (varsayılan: 32000)')
    parser.add_argument('--model-type', choices=['bpe', 'unigram'], 
                       default='bpe', help='Model tipi (varsayılan: bpe)')
    parser.add_argument('--coverage', type=float, default=0.9995,
                       help='Karakter kapsamı (varsayılan: 0.9995)')
    parser.add_argument('--output-dir', type=str, default='turkish_tokenizer',
                       help='Çıktı dizini (varsayılan: turkish_tokenizer)')
    
    args = parser.parse_args()
    
    # Trainer'ı başlat
    trainer = TurkishTokenizerTrainer(
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.coverage,
        output_dir=args.output_dir
    )
    
    # Eğitimi çalıştır
    trainer.run_training()

if __name__ == "__main__":
    main() 