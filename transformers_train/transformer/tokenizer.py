import os
from typing import List

_MESSAGES_PRINTED = False

try:
    import sentencepiece as smp
    SENTENCEPIECE_AVAILABLE = True
    if not _MESSAGES_PRINTED:
        print("SentencePiece available")
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    if not _MESSAGES_PRINTED:
        print("SentencePiece not available, install with: pip install sentencepiece")

_MESSAGES_PRINTED = True

def create_tokenizer(model_path: str = None):
    if not SENTENCEPIECE_AVAILABLE:
        raise ImportError("SentencePiece not available, install with: pip install sentencepiece")
    
    if not model_path:
        model_path = "turkish_tokenizer/turkish_tokenizer.model"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tokenizer file not found: {model_path}")
    
    sp = smp.SentencePieceProcessor()
    sp.load(model_path)

    class TokenizerWrapper:
        def __init__(self, sp_processor):
            self.sp = sp_processor
            
        def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
            if add_special_tokens:
                return [self.sp.bos_id()] + self.sp.encode_as_ids(text)
            else:
                return self.sp.encode_as_ids(text)
        
        def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
            if skip_special_tokens:
                filtered_ids = [
                    tid for tid in token_ids 
                    if tid not in [self.sp.pad_id(), self.sp.bos_id(), self.sp.eos_id(), self.sp.unk_id()]
                ]
                return self.sp.decode_ids(filtered_ids)
            else:
                return self.sp.decode_ids(token_ids)
        
        def encode_as_pieces(self, text: str) -> List[str]:
            return self.sp.encode_as_pieces(text)
        
        @property
        def vocab_size(self) -> int:
            return self.sp.vocab_size()
        
        def get_vocab_size(self) -> int:
            return self.sp.vocab_size()
    
    return TokenizerWrapper(sp)