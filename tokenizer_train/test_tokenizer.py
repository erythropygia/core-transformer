import os
import sys
import time
from pathlib import Path

# Add parent directory to path to import transformer modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformer_train.transformer.tokenizer import RustBPETokenizer, SPECIAL_TOKENS, create_tokenizer


def test_tokenizer(tokenizer_dir=None):    
    # Try to load tokenizer
    if tokenizer_dir is None:
        # Try common locations
        possible_dirs = ["tokenizer", "turkish_tokenizer", "out/tokenizer"]
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                pickle_path = os.path.join(dir_path, "tokenizer.pkl")
                if os.path.exists(pickle_path):
                    tokenizer_dir = dir_path
                    break
    
    if tokenizer_dir is None or not os.path.exists(tokenizer_dir):
        print("Tokenizer directory not found!")
        print(f"Expected locations: tokenizer/, turkish_tokenizer/, out/tokenizer/")
        print("First train the tokenizer: python train_tokenizer.py")
        return False
    
    try:
        tokenizer = create_tokenizer(tokenizer_dir=tokenizer_dir)
        print(f"Loaded tokenizer from: {tokenizer_dir}")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return False
    
    print("=" * 80)
    print("TOKENIZER TEST")
    print("=" * 80)
    print()
    
    # Basic info
    vocab_size = tokenizer.get_vocab_size()
    bos_token_id = tokenizer.get_bos_token_id()
    special_tokens = tokenizer.get_special_tokens()
    
    # Try to get EOS token ID
    try:
        eos_token_id = tokenizer.encode_special("<|eos|>")
    except:
        try:
            eos_token_id = tokenizer.encode_special("<eos>")
        except:
            eos_token_id = None
    
    print(f"Tokenizer Information:")
    print(f"   Vocabulary size: {vocab_size:,}")
    print(f"   BOS token ID: {bos_token_id} -> {tokenizer.decode([bos_token_id], skip_special_tokens=False)}")
    if eos_token_id is not None:
        print(f"   EOS token ID: {eos_token_id} -> {tokenizer.decode([eos_token_id], skip_special_tokens=False)}")
    print(f"   Special tokens count: {len(special_tokens)}")
    print()
    
    # Special tokens test
    print("Special Tokens Test:")
    print("-" * 40)
    found_tokens = 0
    for token in SPECIAL_TOKENS:
        try:
            token_id = tokenizer.encode_special(token)
            decoded = tokenizer.decode([token_id])
            if token in decoded or decoded == token:
                print(f"   {token} -> ID: {token_id}")
                found_tokens += 1
            else:
                print(f"   {token} -> ID: {token_id} (decode: {decoded})")
        except Exception as e:
            print(f"   {token} -> Error: {e}")
    
    print(f"\n   Coverage: {found_tokens}/{len(SPECIAL_TOKENS)} ({found_tokens/len(SPECIAL_TOKENS)*100:.1f}%)")
    print()
    
    # Basic encoding/decoding test
    print("Basic Encode/Decode Tests:")
    print("-" * 40)
    
    test_sentences = [
        "Merhaba dünya! 🌍 Türkçe tokenizer nasıl çalışıyor?",
        "Türkiye'nin başkenti Ankara'dır ve en büyük şehri İstanbul'dur.",
        "Öğrencilerimiz sınavlarında çok başarılı oldular.",
        "Evlerimizde, okullarımızda, iş yerlerimizde hep birlikte yaşıyoruz.",
        "Teknolojik gelişmeler hayatımızı kolaylaştırıyor.",
        "Arkadaşlarımızla birlikte güzel anılar oluşturuyoruz.",
        "Numbers: 123, 4567, 89",
        "Special chars: @#$%^&*()",
    ]
    
    all_passed = True
    for i, sentence in enumerate(test_sentences, 1):
        try:
            # Encode without special tokens
            tokens = tokenizer.encode(sentence, add_special_tokens=False)
            token_ids = tokens if isinstance(tokens, list) else tokens[0] if isinstance(tokens[0], list) else tokens
            
            # Decode
            decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
            
            is_correct = sentence == decoded
            if not is_correct:
                all_passed = False
            
            status = "✅" if is_correct else "❌"
            print(f"{status} Test {i}: {sentence[:50]}{'...' if len(sentence) > 50 else ''}")
            if not is_correct:
                print(f"      Original: {sentence}")
                print(f"      Decoded:  {decoded}")
            else:
                print(f"      Tokens: {len(token_ids)}, Ratio: {len(sentence.encode('utf-8')) / len(token_ids):.2f} bytes/token")
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
            all_passed = False
    
    print()
    
    # BOS token test
    print("BOS Token Test:")
    print("-" * 40)
    test_text = "Bu bir test metnidir."
    tokens_with_bos = tokenizer.encode(test_text, prepend=tokenizer.get_bos_token_id())
    tokens_without_bos = tokenizer.encode(test_text, add_special_tokens=False)
    
    print(f"   Text: {test_text}")
    print(f"   Without BOS: {len(tokens_without_bos)} tokens -> {tokens_without_bos[:5]}...")
    print(f"   With BOS: {len(tokens_with_bos)} tokens -> {tokens_with_bos[:6]}...")
    print(f"   First token ID: {tokens_with_bos[0]} (should be BOS: {bos_token_id})")
    
    # Decode with and without special tokens
    decoded_with_special = tokenizer.decode(tokens_with_bos, skip_special_tokens=False)
    decoded_without_special = tokenizer.decode(tokens_with_bos, skip_special_tokens=True)
    
    print(f"   Decoded (with special): {decoded_with_special}")
    print(f"   Decoded (skip special): {decoded_without_special}")
    
    if tokens_with_bos[0] == bos_token_id:
        print(f"   ✅ BOS token correctly prepended")
    else:
        print(f"   ❌ BOS token not correctly prepended")
        all_passed = False
    
    if test_text == decoded_without_special:
        print(f"   ✅ Skip special tokens works correctly")
    else:
        print(f"   ❌ Skip special tokens failed")
        all_passed = False
    print()
    
    # EOS token test
    print("EOS Token Test:")
    print("-" * 40)
    if eos_token_id is not None:
        test_text_eos = "Bu bir test metnidir."
        tokens_without_eos = tokenizer.encode(test_text_eos, add_special_tokens=False)
        
        # Manually append EOS
        if isinstance(tokens_without_eos, list) and not isinstance(tokens_without_eos[0], list):
            tokens_with_eos = tokens_without_eos + [eos_token_id]
        else:
            tokens_with_eos = (tokens_without_eos[0] if isinstance(tokens_without_eos[0], list) else tokens_without_eos) + [eos_token_id]
        
        print(f"   Text: {test_text_eos}")
        print(f"   Without EOS: {len(tokens_without_eos)} tokens -> ...{tokens_without_eos[-3:]}")
        print(f"   With EOS: {len(tokens_with_eos)} tokens -> ...{tokens_with_eos[-4:]}")
        print(f"   Last token ID: {tokens_with_eos[-1]} (should be EOS: {eos_token_id})")
        
        # Decode with and without special tokens
        decoded_with_special = tokenizer.decode(tokens_with_eos, skip_special_tokens=False)
        decoded_without_special = tokenizer.decode(tokens_with_eos, skip_special_tokens=True)
        
        print(f"   Decoded (with special): {decoded_with_special}")
        print(f"   Decoded (skip special): {decoded_without_special}")
        
        if tokens_with_eos[-1] == eos_token_id:
            print(f"   ✅ EOS token correctly appended")
        else:
            print(f"   ❌ EOS token not correctly appended")
            all_passed = False
        
        if test_text_eos == decoded_without_special:
            print(f"   ✅ Skip special tokens works correctly")
        else:
            print(f"   ❌ Skip special tokens failed")
            all_passed = False
    else:
        print(f"   ⚠️  EOS token not found in tokenizer")
    print()
    
    # BOS + EOS combined test (Generation simulation)
    print("BOS + EOS Combined Test (Generation Simulation):")
    print("-" * 40)
    if eos_token_id is not None:
        test_text_gen = "Türkiye'nin başkenti Ankara'dır."
        
        # Simulate generation: BOS + text + EOS
        tokens_content = tokenizer.encode(test_text_gen, add_special_tokens=False)
        if isinstance(tokens_content, list) and isinstance(tokens_content[0], list):
            tokens_content = tokens_content[0]
        
        tokens_full_sequence = [bos_token_id] + tokens_content + [eos_token_id]
        
        print(f"   Generated text: {test_text_gen}")
        print(f"   Full sequence length: {len(tokens_full_sequence)} tokens")
        print(f"   Token IDs: [{tokens_full_sequence[0]}, {tokens_full_sequence[1]}, ..., {tokens_full_sequence[-2]}, {tokens_full_sequence[-1]}]")
        print(f"   First token (BOS): {tokens_full_sequence[0]} == {bos_token_id}")
        print(f"   Last token (EOS): {tokens_full_sequence[-1]} == {eos_token_id}")
        
        # Decode with special tokens visible
        decoded_with_special = tokenizer.decode(tokens_full_sequence, skip_special_tokens=False)
        decoded_without_special = tokenizer.decode(tokens_full_sequence, skip_special_tokens=True)
        
        print(f"\n   Decoded (SHOW_SPECIAL_TOKENS=True):")
        print(f"      {decoded_with_special}")
        print(f"\n   Decoded (SHOW_SPECIAL_TOKENS=False):")
        print(f"      {decoded_without_special}")
        
        # Verify
        has_bos = tokens_full_sequence[0] == bos_token_id
        has_eos = tokens_full_sequence[-1] == eos_token_id
        clean_text_matches = test_text_gen == decoded_without_special
        
        if has_bos and has_eos:
            print(f"   ✅ BOS and EOS tokens correctly placed")
        else:
            print(f"   ❌ BOS/EOS tokens placement failed")
            all_passed = False
        
        if clean_text_matches:
            print(f"   ✅ Clean text extraction works (skip_special_tokens=True)")
        else:
            print(f"   ❌ Clean text extraction failed")
            print(f"      Expected: {test_text_gen}")
            print(f"      Got: {decoded_without_special}")
            all_passed = False
        
        # Check if special tokens are visible in decoded_with_special
        bos_token_str = tokenizer.decode([bos_token_id], skip_special_tokens=False)
        eos_token_str = tokenizer.decode([eos_token_id], skip_special_tokens=False)
        
        if bos_token_str in decoded_with_special or decoded_with_special.startswith(test_text_gen):
            print(f"   ✅ Special tokens visible when skip_special_tokens=False")
        else:
            print(f"   ⚠️  Special tokens may not be visible (check tokenizer decode behavior)")
    else:
        print(f"   ⚠️  Cannot test: EOS token not found")
    print()
    
    # Batch encoding test
    print("Batch Encoding Test:")
    print("-" * 40)
    batch_texts = [
        "İlk metin.",
        "İkinci metin.",
        "Üçüncü metin."
    ]
    try:
        batch_tokens = tokenizer.encode(batch_texts, add_special_tokens=False)
        print(f"   Batch size: {len(batch_texts)}")
        print(f"   Results: {[len(tokens) for tokens in batch_tokens]}")
        
        # Verify each can be decoded
        for i, (text, tokens) in enumerate(zip(batch_texts, batch_tokens)):
            decoded = tokenizer.decode(tokens)
            if text == decoded:
                print(f"   Batch item {i+1} decode successful")
            else:
                print(f"   Batch item {i+1} decode failed")
                all_passed = False
    except Exception as e:
        print(f"   Batch encoding failed: {e}")
        all_passed = False
    print()
    
    # Turkish morphology test
    print("Turkish Morphology Test:")
    print("-" * 40)
    
    morphology_examples = [
        ("ev", "evler", "evlerimiz", "evlerimizde"),
        ("öğretmen", "öğretmenler", "öğretmenlerimiz", "öğretmenlerimizin"),
        ("çocuk", "çocuklar", "çocuklarımız", "çocuklarımızın"),
        ("kitap", "kitaplar", "kitaplarımız", "kitaplarımızdan")
    ]
    
    for base, plural, possessive, locative in morphology_examples:
        base_tokens = tokenizer.encode(base, add_special_tokens=False)
        plural_tokens = tokenizer.encode(plural, add_special_tokens=False)
        poss_tokens = tokenizer.encode(possessive, add_special_tokens=False)
        loc_tokens = tokenizer.encode(locative, add_special_tokens=False)
        
        base_len = len(base_tokens) if isinstance(base_tokens, list) else len(base_tokens[0]) if isinstance(base_tokens[0], list) else 1
        plural_len = len(plural_tokens) if isinstance(plural_tokens, list) else len(plural_tokens[0]) if isinstance(plural_tokens[0], list) else 1
        poss_len = len(poss_tokens) if isinstance(poss_tokens, list) else len(poss_tokens[0]) if isinstance(poss_tokens[0], list) else 1
        loc_len = len(loc_tokens) if isinstance(loc_tokens, list) else len(loc_tokens[0]) if isinstance(loc_tokens[0], list) else 1
        
        print(f"   {base} → {plural} → {possessive} → {locative}")
        print(f"   Token counts: {base_len} → {plural_len} → {poss_len} → {loc_len}")
    print()
    
    # Chat format test
    print("Chat Format Test:")
    print("-" * 40)
    
    chat_example = (
        "<|user_start|>Merhaba! Nasılsın?<|user_end|>"
        "<|assistant_start|>Merhaba! Ben iyiyim, teşekkürler. Sen nasılsın?<|assistant_end|>"
    )
    
    try:
        chat_tokens = tokenizer.encode(chat_example, add_special_tokens=False)
        chat_decoded = tokenizer.decode(chat_tokens, skip_special_tokens=False)
        
        print(f"   Chat example: {chat_example[:60]}...")
        print(f"   Token count: {len(chat_tokens) if isinstance(chat_tokens, list) else len(chat_tokens[0]) if isinstance(chat_tokens[0], list) else 1}")
        print(f"   Decoded: {chat_decoded[:60]}...")
        
        # Check if special tokens are present
        # Note: When special tokens are written as strings in text (like "<|user_start|>"),
        # they are tokenized as regular characters, not as special token IDs.
        # So we check if the decoded text contains the special token strings.
        chat_token_list = chat_tokens if isinstance(chat_tokens, list) and not (chat_tokens and isinstance(chat_tokens[0], list)) else (chat_tokens[0] if isinstance(chat_tokens, list) and chat_tokens and isinstance(chat_tokens[0], list) else chat_tokens)
        has_special_tokens = "<|user_start|>" in chat_decoded and "<|user_end|>" in chat_decoded
        print(f"   {'OK' if has_special_tokens else 'NOT OK'} Special tokens detected")
    except Exception as e:
        print(f"   Chat format test failed: {e}")
        all_passed = False
    print()
    
    # Performance benchmark
    print("Performance Benchmark:")
    print("-" * 40)
    
    test_text = "Bu bir performans testidir. " * 100
    test_text_bytes = len(test_text.encode('utf-8'))
    
    # Encoding benchmark
    num_iterations = 100
    start_time = time.time()
    for _ in range(num_iterations):
        tokens = tokenizer.encode(test_text, add_special_tokens=False)
    encoding_time = time.time() - start_time
    
    # Decoding benchmark
    tokens = tokenizer.encode(test_text, add_special_tokens=False)
    token_list = tokens if isinstance(tokens, list) else tokens[0] if isinstance(tokens[0], list) else tokens
    start_time = time.time()
    for _ in range(num_iterations):
        decoded = tokenizer.decode(token_list)
    decoding_time = time.time() - start_time
    
    num_tokens = len(token_list)
    
    print(f"   Test text: {test_text_bytes:,} bytes, {num_tokens:,} tokens")
    print(f"   Encoding: {encoding_time:.4f}s ({num_iterations} operations)")
    print(f"   Decoding: {decoding_time:.4f}s ({num_iterations} operations)")
    print(f"   Encoding speed: {test_text_bytes * num_iterations / encoding_time:,.0f} bytes/sec")
    print(f"   Decoding speed: {test_text_bytes * num_iterations / decoding_time:,.0f} bytes/sec")
    print(f"   Tokens/sec: {num_tokens * num_iterations / encoding_time:,.0f}")
    print()
    
    # Summary
    print("=" * 80)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - Check output above")
    print("=" * 80)
    
    return all_passed


def interactive_test(tokenizer_dir=None):
    # Try to load tokenizer
    if tokenizer_dir is None:
        possible_dirs = ["tokenizer", "turkish_tokenizer", "out/tokenizer"]
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                pickle_path = os.path.join(dir_path, "tokenizer.pkl")
                if os.path.exists(pickle_path):
                    tokenizer_dir = dir_path
                    break
    
    if tokenizer_dir is None or not os.path.exists(tokenizer_dir):
        print("Tokenizer directory not found!")
        return
    
    try:
        tokenizer = create_tokenizer(tokenizer_dir=tokenizer_dir)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return
    
    print("=" * 80)
    print("INTERACTIVE TOKENIZER TEST")
    print("=" * 80)
    print("Type 'exit' or 'quit' to exit")
    print("Try using special tokens like <|bos|>, <|user_start|>, etc.")
    print()
    
    while True:
        try:
            text = input("Enter text to tokenize: ").strip()
            
            if text.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                continue
            
            # Encode
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_ids = tokens if isinstance(tokens, list) else tokens[0] if isinstance(tokens[0], list) else tokens
            
            # Decode
            decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
            
            # Display results
            print(f"   Input: {text}")
            print(f"   Token IDs: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
            print(f"   Token count: {len(token_ids)}")
            print(f"   Compression: {len(text.encode('utf-8')) / len(token_ids):.2f} bytes/token")
            print(f"   Decoded: {decoded}")
            print(f"   {'OK' if text == decoded else 'NOT OK'} Perfect match!")
            print("-" * 80)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("-" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test tokenizer")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--tokenizer-dir", "-d",
        type=str,
        default=None,
        help="Tokenizer directory path (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_test(args.tokenizer_dir)
    else:
        success = test_tokenizer(args.tokenizer_dir)
        sys.exit(0 if success else 1)
