import os
import sentencepiece as spm
from pathlib import Path
import time
import sys

def test_tokenizer():
    model_file = Path('turkish_tokenizer/turkish_tokenizer.model')
    
    if not model_file.exists():
        print("Model file not found!")
        print(f"Expected location: {model_file}")
        print("First train the tokenizer: python train_turkish_tokenizer.py")
        return
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))
    
    print(f"Model: {model_file}")
    print(f"Vocabulary size: {sp.vocab_size():,}")
    print(f"UNK token ID: {sp.unk_id()}")
    print(f"BOS token ID: {sp.bos_id()}")
    print(f"EOS token ID: {sp.eos_id()}")
    print(f"PAD token ID: {sp.pad_id()}")
    print()
    
    special_tokens = [
        # Document structure tokens
        '<|beginoftext|>', '<|endoftext|>', '<|startoftext|>', '<newline>',
        # Core tokens
        '<s>', '</s>', '<pad>', '<mask>', '<turkish>', '<instruction>', '</instruction>', '<context>',
        # Chat tokens
        '<|system|>', '<|user|>', '<|assistant|>', '<|end|>',
        # Safety tokens
        '<safe>', '<unsafe>', '<filtered>',
        # Task specific tokens
        '<translate>', '<summarize>', '<classify>',
        # Reserved tokens
        '<reserved1>', '<reserved2>', '<reserved3>'
    ]
    
    print("Special Token Verification:")
    found_tokens = 0
    for token in special_tokens:
        token_id = sp.piece_to_id(token)
        if token_id != sp.unk_id():
            print(f"   {token} -> ID: {token_id}")
            found_tokens += 1
        else:
            print(f"   {token} -> Not found")
    
    print(f"\nToken Coverage: {found_tokens}/{len(special_tokens)} ({found_tokens/len(special_tokens)*100:.1f}%)")
    print()
    
    print("Document Structure Token Test:")
    
    document_example = f"{sp.id_to_piece(sp.piece_to_id('<|beginoftext|>'))}" + \
                      "Bu bir örnek Türkçe belge içeriğidir. " + \
                      "Tokenizer bu metni nasıl işliyor test ediyoruz." + \
                      f"{sp.id_to_piece(sp.piece_to_id('<|endoftext|>'))}"
    
    doc_tokens = sp.encode(document_example, out_type=str)
    print(f"Document with structure tokens:")
    print(f"   Input: {document_example}")
    print(f"   Token count: {len(doc_tokens)}")
    print(f"     First 5 tokens: {doc_tokens[:5]}")
    print(f"     Last 5 tokens: {doc_tokens[-5:]}")
    print()
    
    # Basic test sentences
    print("Basic Tokenization Tests:")
    print("-" * 40)
    
    test_sentences = [
        "Merhaba dünya! 🌍 Türkçe tokenizer nasıl çalışıyor?",
        "Türkiye'nin başkenti Ankara'dır ve en büyük şehri İstanbul'dur.",
        "Öğrencilerimiz sınavlarında çok başarılı oldular.",
        "Evlerimizde, okullarımızda, iş yerlerimizde hep birlikte yaşıyoruz.",
        "Teknolojik gelişmeler hayatımızı kolaylaştırıyor.",
        "Arkadaşlarımızla birlikte güzel anılar oluşturuyoruz."
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"Test {i}: {sentence}")
        
        tokens = sp.encode(sentence, out_type=str)
        token_ids = sp.encode(sentence, out_type=int)
        
        print(f"   Tokens: {tokens[:8]}{'...' if len(tokens) > 8 else ''}")
        print(f"   Token count: {len(tokens)}")
        print(f"   Compression ratio: {len(sentence) / len(tokens):.2f} chars/token")
        
        decoded = sp.decode(token_ids)
        is_correct = sentence == decoded
        print(f"   {'Decode successful' if is_correct else 'Decode failed'}")
        if not is_correct:
            print(f"   Original: {sentence}")
            print(f"   Decoded:  {decoded}")
        print()
    
    # Turkish morphology test
    print("Turkish Morphology Analysis:")
    
    morphology_examples = [
        ("ev", "evler", "evlerimiz", "evlerimizde"),
        ("öğretmen", "öğretmenler", "öğretmenlerimiz", "öğretmenlerimizin"),
        ("çocuk", "çocuklar", "çocuklarımız", "çocuklarımızın"),
        ("kitap", "kitaplar", "kitaplarımız", "kitaplarımızdan")
    ]
    
    for base, plural, possessive, locative in morphology_examples:
        base_tokens = sp.encode(base, out_type=str)
        plural_tokens = sp.encode(plural, out_type=str)
        poss_tokens = sp.encode(possessive, out_type=str)
        loc_tokens = sp.encode(locative, out_type=str)
        
        print(f"   {base} → {plural} → {possessive} → {locative}")
        print(f"   Token counts: {len(base_tokens)} → {len(plural_tokens)} → {len(poss_tokens)} → {len(loc_tokens)}")
        print(f"   {base}: {base_tokens}")
        print(f"   {possessive}: {poss_tokens}")
        print()
    
    # Instruction Tuning token test
    print("Instruction Tuning Token Test:")
    print("-" * 40)
    
    instruction_example = """<|beginoftext|><|system|>Sen yardımcı bir Türkçe asistansın. 🤖</|system|>
<|user|><instruction>Bu metni özetle:</instruction>
Türkiye Cumhuriyeti Anadolu ve Doğu Trakya'da kurulmuş bir ülkedir. 
Başkenti Ankara, en büyük şehri İstanbul'dur.</|user|>
<|assistant|><translate>Türkiye, Anadolu ve Doğu Trakya'da kurulmuş cumhuriyettir. 
Başkent Ankara, büyük şehir İstanbul.</translate></|assistant|><|endoftext|>"""
    
    print("Instruction example:")
    print(instruction_example)
    print()
    
    inst_tokens = sp.encode(instruction_example, out_type=str)
    inst_ids = sp.encode(instruction_example, out_type=int)
    
    print(f"   Total token count: {len(inst_tokens)}")
    print(f"   Average chars per token: {len(instruction_example) / len(inst_tokens):.2f}")
    
    begin_token_present = '<|beginoftext|>' in inst_tokens
    end_token_present = '<|endoftext|>' in inst_tokens
    
    print(f"   {'OK' if begin_token_present else 'NO'} <|beginoftext|> token found")
    print(f"   {'OK' if end_token_present else 'NO'} <|endoftext|> token found")
    
    decoded_inst = sp.decode(inst_ids)
    decode_success = instruction_example == decoded_inst
    print(f"   {'OK' if decode_success else 'NO'} Instruction decode successful")
    print()
    
    print("Performance Benchmark:")
    
    
    test_text = "Bu bir performans testidir." * 100 
    
    start_time = time.time()
    for _ in range(100):
        tokens = sp.encode(test_text, out_type=int)
    encoding_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(100):
        decoded = sp.decode(tokens)
    decoding_time = time.time() - start_time
    
    print(f"   Test text length: {len(test_text):,} characters")
    print(f"   Token count: {len(tokens):,}")
    print(f"   Encoding: {encoding_time:.4f}s (100 operations)")
    print(f"   Decoding: {decoding_time:.4f}s (100 operations)")
    print(f"   Encoding speed: {len(test_text) * 100 / encoding_time:,.0f} chars/sec")
    print(f"   Decoding speed: {len(test_text) * 100 / decoding_time:,.0f} chars/sec")
    print()
    
    print("All tests completed successfully!")

def interactive_test():
    
    model_file = Path('turkish_tokenizer/turkish_tokenizer.model')
    
    if not model_file.exists():
        print("Model file not found!")
        return
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))
    
    print("Interactive Turkish Tokenizer Test")
    print("=" * 50)
    print("Type 'exit' to quit")
    print("Try using special tokens like <|beginoftext|> in your input!")
    print()
    
    while True:
        try:
            text = input("Enter text to tokenize: ").strip()
            
            if text.lower() in ['exit', 'quit']:
                print("See you later!")
                break
            
            if not text:
                continue
            
            tokens = sp.encode(text, out_type=str)
            token_ids = sp.encode(text, out_type=int)
            
            print(f"   Tokens: {tokens}")
            print(f"   Token IDs: {token_ids}")
            print(f"   Token count: {len(tokens)}")
            print(f"   Compression ratio: {len(text) / len(tokens):.2f} chars/token")
            
            decoded = sp.decode(token_ids)
            print(f"   Decoded: {decoded}")
            print(f"   {'Perfect match!' if text == decoded else 'Mismatch detected'}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()
    else:
        test_tokenizer() 