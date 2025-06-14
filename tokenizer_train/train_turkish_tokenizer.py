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
        Türkçe tokenizer eğiticisi
        
        Args:
            vocab_size: Vocabulary boyutu (32K Türkçe için optimal)
            model_type: Model tipi ('bpe', 'unigram', 'word', 'char')
            character_coverage: Karakter kapsamı (Türkçe için 0.9995)
            output_dir: Çıktı dizini
        """
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Türkçe için optimal SentencePiece parametreleri
        self.sp_params = {
            'model_type': model_type,
            'vocab_size': vocab_size,
            'character_coverage': character_coverage,
            'normalization_rule_name': 'nfkc',  # Türkçe karakterler için
            'remove_extra_whitespaces': True,
            'split_by_unicode_script': True,    # Türkçe morfoloji için önemli
            'split_by_whitespace': True,
            'split_by_number': True,
            'split_digits': True,
            'treat_whitespace_as_suffix': False,
            'allow_whitespace_only_pieces': True,
            'max_sentence_length': 8192,
            'shuffle_input_sentence': True,
            'input_sentence_size': 10000000,    # 10M cümle
            'seed_sentencepiece_size': 1000000,
            'shrinking_factor': 0.75,
            'num_threads': os.cpu_count() or 4,
            'max_sentencepiece_length': 16,
            'num_sub_iterations': 2,
        }
        
        # Türkçe özel tokenlar
        self.special_tokens = [
            '<s>', '</s>', '<pad>',                     # Temel tokenlar (unk otomatik)
            '<mask>', '<cls>', '<sep>',                # BERT-style tokenlar  
            '<turkish>', '<TR>',                       # Dil tokenları
            '<question>', '<answer>',                  # QA tokenları
            '<news>', '<social>', '<formal>',          # Domain tokenları
            '<thinking>', '</thinking>',                # Think tokenları (modern AI için)
            '<thought>', '</thought>',                  # Alternatif düşünce tokenları
            '<reasoning>', '</reasoning>',              # Mantık yürütme
            '<analysis>', '</analysis>',                # Analiz bölümleri
            '<|endoftext|>', '<|startoftext|>',        # End of text ve start of text tokenları
            
            # Instruction Tuning Tokenları
            '<|system|>', '<|user|>', '<|assistant|>', # Chat rolleri (ChatML style)
            '<|im_start|>', '<|im_end|>',              # Instruction message markers
            '<instruction>', '</instruction>',          # Instruction wrapper
            '<input>', '</input>',                     # Input wrapper
            '<output>', '</output>',                   # Output wrapper
            '<context>', '</context>',                 # Context information
            
            # Multi-turn Conversation
            '<turn>', '</turn>',                       # Conversation turn markers
            '<conversation>', '</conversation>',        # Conversation wrapper
            '<history>', '</history>',                 # Chat history
            
            # Function Calling & Tool Use
            '<function_call>', '</function_call>',     # Function call wrapper
            '<tool_use>', '</tool_use>',               # Tool usage marker
            '<json>', '</json>',                       # JSON structured data
            '<code>', '</code>',                       # Code blocks
            
            # Safety & Control
            '<safe>', '<unsafe>',                      # Safety markers
            '<warning>', '</warning>',                 # Warning wrapper
            '<filter>', '</filter>',                   # Content filtering
            
            # Task-Specific (Türkçe için)
            '<çeviri>', '</çeviri>',                   # Translation task
            '<özet>', '</özet>',                       # Summarization task
            '<soru>', '</soru>',                       # Question marker
            '<cevap>', '</cevap>',                     # Answer marker
            
            # Multi-modal (gelecek için)
            '<image>', '</image>',                     # Image content
            '<audio>', '</audio>',                     # Audio content
            '<video>', '</video>',                     # Video content
        ]

    def get_corpus_file(self) -> str:
        """
        Mevcut OSCAR corpus dosyasını kontrol et ve döndür
        
        Returns:
            Corpus dosyasının yolu
        """
        # Script'in bulunduğu dizinden data klasörüne erişim
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
        Corpus dosyasını önişlemden geçir
        
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
                    
                    # Basit filtreleme
                    if self._is_valid_text(line):
                        outfile.write(line + '\n')
                        valid_lines += 1
                    
                    if total_lines % 100000 == 0:
                        logger.info(f"  📝 {total_lines} satır işlendi, {valid_lines} geçerli satır")
        
        logger.info(f"✅ Önişleme tamamlandı: {valid_lines}/{total_lines} satır korundu")
        return str(processed_file)

    def _is_valid_text(self, text: str) -> bool:
        """Basit metin kontrolü"""
        # Çok kısa metinleri filtrele
        if len(text.strip()) < 10:
            return False
        
        # Çok uzun metinleri filtrele
        if len(text) > 1000:
            return False
        
        # Sayısal içerik kontrolü
        digit_ratio = sum(c.isdigit() for c in text) / len(text)
        if digit_ratio > 0.5:
            return False
        
        return True

    def train_tokenizer(self, corpus_file: str) -> Dict[str, str]:
        """
        SentencePiece tokenizer eğit
        
        Args:
            corpus_file: Eğitim korpusu dosyası
            
        Returns:
            Eğitilen modelin dosya yolları
        """
        logger.info("🚀 SentencePiece tokenizer eğitimi başlıyor...")
        
        model_prefix = str(self.output_dir / 'turkish_tokenizer')
        
        # SentencePiece eğitim parametrelerini hazırla
        sp_params = self.sp_params.copy()
        sp_params.update({
            'input': corpus_file,
            'model_prefix': model_prefix,
            'user_defined_symbols': self.special_tokens,
        })
        
        # Parametreleri logla
        logger.info("📋 Eğitim parametreleri:")
        for key, value in sp_params.items():
            logger.info(f"  {key}: {value}")
        
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
                
                # Model istatistiklerini göster
                self._show_model_stats(model_file)
                
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

    def _show_model_stats(self, model_file: str) -> None:
        """Model istatistiklerini göster"""
        try:
            sp = spm.SentencePieceProcessor()
            sp.load(model_file)
            
            logger.info("📊 Model İstatistikleri:")
            logger.info(f"  📝 Vocabulary boyutu: {sp.vocab_size()}")
            logger.info(f"  🔤 Model tipi: {self.model_type.upper()}")
            logger.info(f"  🎯 Karakter kapsamı: {self.character_coverage}")
            
            # Özel tokenları kontrol et
            special_tokens_found = []
            for token in self.special_tokens:
                if sp.piece_to_id(token) != sp.unk_id():
                    special_tokens_found.append(token)
            
            logger.info(f"  ⭐ Özel tokenlar: {len(special_tokens_found)}/{len(self.special_tokens)}")
            logger.info(f"     {', '.join(special_tokens_found[:5])}...")
            
        except Exception as e:
            logger.warning(f"Model istatistikleri gösterilemedi: {e}")

    def run_training(self) -> Dict[str, str]:
        """Tam eğitim sürecini çalıştır"""
        logger.info("🚀 Türkçe Tokenizer Eğitimi Başlıyor!")
        logger.info("=" * 50)
        
        try:
            # 1. Corpus dosyasını kontrol et
            logger.info("📊 1. Aşama: Corpus dosyası kontrol ediliyor...")
            corpus_file = self.get_corpus_file()
            
            # 2. Corpus'u önişle
            logger.info("🔄 2. Aşama: Corpus önişleme...")
            processed_corpus = self.preprocess_corpus(corpus_file)
            
            # 3. Tokenizer'ı eğit
            logger.info("🎓 3. Aşama: Tokenizer eğitimi...")
            result = self.train_tokenizer(processed_corpus)
            
            # 4. Özet bilgi
            logger.info("🎉 Eğitim Tamamlandı!")
            logger.info("=" * 50)
            logger.info(f"📁 Model dosyası: {result['model']}")
            logger.info(f"📁 Vocabulary dosyası: {result['vocab']}")
            logger.info(f"⏱️  Toplam eğitim süresi: {result['training_time']:.2f}s")
            logger.info("")
            logger.info("🚀 Tokenizer'ınız hazır! Test etmek için:")
            logger.info("   python test_turkish_tokenizer.py")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Eğitim başarısız: {e}")
            raise

def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="Türkçe SentencePiece Tokenizer Eğitici")
    parser.add_argument('--vocab-size', type=int, default=32000, 
                       help='Vocabulary boyutu (varsayılan: 32000)')
    parser.add_argument('--model-type', choices=['bpe', 'unigram', 'word', 'char'], 
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