import os
import json
import time
import logging
import argparse
import sentencepiece as spm
from pathlib import Path
from typing import List, Dict, Optional

class TokenizerTrainer:
    def __init__(self, 
                 vocab_size: int = 32000,
                 model_type: str = 'bpe',
                 character_coverage: float = 0.9995,
                 output_dir: str = 'turkish_tokenizer'):
        
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.sp_params = {
            'model_type': model_type,
            'vocab_size': vocab_size,
            'character_coverage': character_coverage,
            'normalization_rule_name': 'nfkc',  # Critical for Turkish characters
            'remove_extra_whitespaces': True,
            'split_by_unicode_script': True,    # Critical for Turkish morphology
            'split_by_whitespace': True,
            'split_by_number': True,
            'split_digits': True,
            'treat_whitespace_as_suffix': False,
            'allow_whitespace_only_pieces': True,
            'max_sentence_length': 8192,
            'shuffle_input_sentence': True,
            'input_sentence_size': 10000000,    # 10M sentence limit
            'seed_sentencepiece_size': 1000000,
            'shrinking_factor': 0.75,
            'num_threads': os.cpu_count() or 4,
            'max_sentencepiece_length': 16,
            'num_sub_iterations': 2,
            'unk_id': 0,    # <unk>
            'bos_id': 1,    # <s> 
            'eos_id': 2,    # </s>
            'pad_id': 3,    # <pad>
        }
        
        self.special_tokens = [
            '<|beginoftext|>',      # Document/text beginning marker
            '<|endoftext|>',        # Document/text end marker
            '<|startoftext|>',      # Alternative start marker
            '<newline>',            # Explicit newline token
            
            '<mask>',               # Masked Language Modeling
            '<turkish>',            # Language identifier
            '<instruction>',        # Instruction tuning start
            '</instruction>',       # Instruction tuning end
            '<context>',           # Context marker
            
            '<|system|>',          # System prompt
            '<|user|>',            # User message  
            '<|assistant|>',       # Assistant response
            '<|end|>',             # Turn end marker
            
            '<safe>',              # Safe content marker
            '<unsafe>',            # Unsafe content marker
            '<filtered>',          # Filtered content marker
            
            '<translate>',         # Translation task marker
            '<summarize>',         # Summarization task marker
            '<classify>',          # Classification task marker
            
            '<reserved1>',         # Future features
            '<reserved2>',         # Function calling, tools etc.
            '<reserved3>',         # Multi-modal extensions
        ]

    def get_corpus_file(self) -> str:
        script_dir = Path(__file__).parent
        corpus_file = script_dir / 'data' / 'oscar_turkish.txt'
        
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
        file_size = corpus_file.stat().st_size / (1024*1024*1024)  # GB
        print(f"Corpus found: {corpus_file}")
        print(f"File size: {file_size:.2f} GB")
        
        return str(corpus_file)

    def preprocess_corpus(self, corpus_file: str) -> str:
        print("Corpus preprocessing starting...")
        
        processed_file = self.output_dir / 'processed_corpus.txt'
        total_lines = 0
        valid_lines = 0
        
        with open(processed_file, 'w', encoding='utf-8') as outfile:
            with open(corpus_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    total_lines += 1
                    line = line.strip()
                    
                    if self._is_high_quality_text(line):
                        outfile.write(line + '\n')
                        valid_lines += 1
                    
                    if total_lines % 100000 == 0:
                        print(f"{total_lines:,} lines processed, {valid_lines:,} valid lines")
                        
                    # Memory efficiency for batch processing
                    if total_lines % 1000000 == 0:
                        print(f"{total_lines//1000000}M lines completed")
        
        retention_rate = (valid_lines / total_lines) * 100
        print(f"Preprocessing completed:")
        print(f"{valid_lines:,}/{total_lines:,} lines retained ({retention_rate:.1f}%)")
        return str(processed_file)

    def _is_high_quality_text(self, text: str) -> bool:
        text = text.strip()
        
        # Minimum length - too short sentences are not useful
        if len(text) < 20:
            return False
        
        # Maximum length - too long sentences are problematic
        if len(text) > 512:
            return False
        
        # Turkish character ratio check
        turkish_chars = sum(1 for c in text if c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZçğıöşüÇĞIİÖŞÜ')
        if len(text) > 0:
            turkish_ratio = turkish_chars / len(text)
            if turkish_ratio < 0.7:  # At least 70% of characters should be Turkish
                return False
        
        # Digit density check
        digit_ratio = sum(c.isdigit() for c in text) / len(text)
        if digit_ratio > 0.3:  # More than 30% of digits
            return False
        
        # Repeated character check (spam filtering)
        if any(char * 4 in text for char in set(text)):
            return False
        
        # Uppercase ratio check (SPAM filtering)
        upper_ratio = sum(c.isupper() for c in text) / len(text)
        if upper_ratio > 0.5:  # More than 50% uppercase
            return False
            
        return True

    def train_tokenizer(self, corpus_file: str) -> Dict[str, str]:

        print("Tokenizer training starting...")
        
        model_prefix = str(self.output_dir / 'turkish_tokenizer')
        
        sp_params = self.sp_params.copy()
        sp_params.update({
            'input': corpus_file,
            'model_prefix': model_prefix,
            'user_defined_symbols': ','.join(self.special_tokens),
        })
        
        # Critical parameters
        print("Training parameters:")
        print(f"Vocabulary size: {sp_params['vocab_size']:,}")
        print(f"Model type: {sp_params['model_type'].upper()}")
        print(f"Character coverage: {sp_params['character_coverage']}")
        print(f"Special token count: {len(self.special_tokens)}")
        print(f"Thread count: {sp_params['num_threads']}")
        
        try:
            print("SentencePiece training starting...")
            start_time = time.time()
            
            spm.SentencePieceTrainer.train(**sp_params)
            
            training_time = time.time() - start_time
            print(f"Tokenizer training completed! Time: {training_time:.2f}s")
            
            # Check output files
            model_file = f"{model_prefix}.model"    
            vocab_file = f"{model_prefix}.vocab"
            
            if os.path.exists(model_file) and os.path.exists(vocab_file):
                print(f"Model file: {model_file}")
                print(f"Vocabulary file: {vocab_file}")
                
                self._analyze_tokenizer_quality(model_file)
                
                return {
                    'model': model_file,
                    'vocab': vocab_file,
                    'training_time': training_time
                }
            else:
                raise FileNotFoundError("Model files could not be created")
                
        except Exception as e:
            print(f"Tokenizer training failed: {e}")
            raise

    def _analyze_tokenizer_quality(self, model_file: str) -> None:
        try:
            sp = spm.SentencePieceProcessor()
            sp.load(model_file)
            
            print("Tokenizer quality analysis:")
            print(f"Total vocabulary: {sp.vocab_size():,}")
            
            # Check special tokens
            special_count = 0
            special_ids = []
            for token in self.special_tokens:
                token_id = sp.piece_to_id(token)
                if token_id != sp.unk_id():
                    special_count += 1
                    special_ids.append(f"{token}:{token_id}")
            
            print(f"Special tokens: {special_count}/{len(self.special_tokens)} successful")
            
            # Check core tokens
            core_tokens = {
                '<unk>': sp.unk_id(),
                '<s>': sp.bos_id(), 
                '</s>': sp.eos_id(),
                '<pad>': sp.pad_id()
            }
            
            print("Core tokens:")
            for token, token_id in core_tokens.items():
                print(f"    {token}: ID {token_id}")
            
            # Turkish test sentences
            test_sentences = [
                "Merhaba, nasılsınız?",
                "Türkiye'nin başkenti Ankara'dır.",
                "Yapay zeka teknolojisi hızla gelişiyor."
            ]
            
            print("Tokenization test:")
            for sentence in test_sentences:
                tokens = sp.encode(sentence, out_type=str)
                token_count = len(tokens)
                print(f"     '{sentence}' -> {token_count} tokens")
            
            # Efficiency score (average chars per token)
            total_chars = sum(len(s) for s in test_sentences)
            total_tokens = sum(len(sp.encode(s)) for s in test_sentences)
            efficiency = total_chars / total_tokens if total_tokens > 0 else 0
            
            print(f"Encoding efficiency: {efficiency:.2f} characters/token")
            
            if efficiency >= 3.0:
                print("Excellent efficiency")
            elif efficiency >= 2.5:
                print("Good efficiency")
            else:
                print("Low efficiency - consider retraining")
                
        except Exception as e:
            print.warning(f"Quality analysis failed: {e}")

    def run_training(self) -> Dict[str, str]:
        print("Tokenizer Training Pipeline")
        try:
            print("Corpus file checking...")
            corpus_file = self.get_corpus_file()
            
            print("High quality corpus preparing...")
            processed_corpus = self.preprocess_corpus(corpus_file)
            
            print("Tokenizer training...")
            result = self.train_tokenizer(processed_corpus)
            
            print("Training completed successfully!")
            print(f"Model file: {result['model']}")
            print(f"Vocabulary file: {result['vocab']}")  
            print(f"Configuration: {result['config']}")
            print(f"Total training time: {result['training_time']:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"Training failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(
        description="Tokenizer training"
    )
    parser.add_argument('--vocab-size', type=int, default=32000, 
                       help='Vocabulary size (default: 32000)')
    parser.add_argument('--model-type', choices=['bpe', 'unigram'], 
                       default='bpe', help='Model type (default: bpe)')
    parser.add_argument('--coverage', type=float, default=0.9995,
                       help='Character coverage (default: 0.9995)')
    parser.add_argument('--output-dir', type=str, default='turkish_tokenizer',
                       help='Output directory (default: turkish_tokenizer)')
    
    args = parser.parse_args()
    
    trainer = TokenizerTrainer(
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.coverage,
        output_dir=args.output_dir
    )

    trainer.run_training()

if __name__ == "__main__":
    main() 