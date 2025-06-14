import os
import sentencepiece as spm
from pathlib import Path

def test_turkish_tokenizer():
    """Türkçe tokenizer'ı kapsamlı test et"""
    
    # Model dosyasını bul
    model_file = Path('tokenizer_train/turkish_tokenizer/turkish_tokenizer.model')
    
    if not model_file.exists():
        print("❌ Model dosyası bulunamadı!")
        print(f"   Beklenen konum: {model_file}")
        print("   Önce tokenizer'ı eğitin: python train_turkish_tokenizer.py")
        return
    
    # Model yükle
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))
    
    print("🇹🇷 Türkçe SentencePiece Tokenizer Test")
    print("=" * 60)
    print(f"📁 Model: {model_file}")
    print(f"📝 Vocabulary boyutu: {sp.vocab_size()}")
    print(f"🔤 UNK token ID: {sp.unk_id()}")
    print(f"🏁 BOS token ID: {sp.bos_id()}")
    print(f"🔚 EOS token ID: {sp.eos_id()}")
    print(f"📄 PAD token ID: {sp.pad_id()}")
    print()
    
    # Özel tokenları kontrol et
    special_tokens = [
        '<s>', '</s>', '<pad>', '<mask>', '<cls>', '<sep>',
        '<turkish>', '<TR>', '<question>', '<answer>',
        '<thinking>', '</thinking>', '<reasoning>', '</reasoning>',
        '<|system|>', '<|user|>', '<|assistant|>',
        '<instruction>', '</instruction>', '<çeviri>', '</çeviri>'
    ]
    
    print("⭐ Özel Token Kontrol:")
    print("-" * 30)
    for token in special_tokens:
        token_id = sp.piece_to_id(token)
        if token_id != sp.unk_id():
            print(f"   ✅ {token} -> ID: {token_id}")
        else:
            print(f"   ❌ {token} -> Bulunamadı")
    print()
    
    # Temel test cümleleri
    print("🔍 Temel Tokenizasyon Testleri:")
    print("-" * 40)
    
    test_sentences = [
        "Merhaba dünya! Türkçe tokenizer nasıl çalışıyor?",
        "Türkiye'nin başkenti Ankara'dır ve en büyük şehri İstanbul'dur.",
        "Öğrencilerimiz sınavlarında çok başarılı oldular.",
        "Evlerimizde, okullarımızda, iş yerlerimizde hep birlikte yaşıyoruz.",
        "Teknolojik gelişmeler hayatımızı kolaylaştırıyor.",
        "Arkadaşlarımızla birlikte güzel anılar oluşturuyoruz."
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"Test {i}: {sentence}")
        
        # Encode (tokenize)
        tokens = sp.encode(sentence, out_type=str)
        token_ids = sp.encode(sentence, out_type=int)
        
        print(f"   📋 Tokenlar: {tokens[:8]}{'...' if len(tokens) > 8 else ''}")
        print(f"   🔢 Token sayısı: {len(tokens)}")
        print(f"   📊 Sıkıştırma oranı: {len(sentence) / len(tokens):.2f} karakter/token")
        
        # Decode (detokenize)
        decoded = sp.decode(token_ids)
        is_correct = sentence == decoded
        print(f"   ✅ Decode {'başarılı' if is_correct else 'başarısız'}")
        if not is_correct:
            print(f"   ⚠️  Orijinal: {sentence}")
            print(f"   ⚠️  Decode:   {decoded}")
        print()
    
    # Türkçe morfoloji testi
    print("🔍 Türkçe Morfoloji Testi:")
    print("-" * 30)
    
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
        
        print(f"📚 {base} -> {plural} -> {possessive} -> {locative}")
        print(f"   Token sayıları: {len(base_tokens)} -> {len(plural_tokens)} -> {len(poss_tokens)} -> {len(loc_tokens)}")
        print(f"   {base}: {base_tokens}")
        print(f"   {plural}: {plural_tokens}")
        print(f"   {possessive}: {poss_tokens}")
        print(f"   {locative}: {loc_tokens}")
        print()
    
    # Instruction Tuning token testi
    print("🤖 Instruction Tuning Token Testi:")
    print("-" * 40)
    
    instruction_example = """<|system|>Sen yardımcı bir Türkçe asistansın.</|system|>
<|user|><instruction>Bu metni özetle:</instruction>
<input>Türkiye Cumhuriyeti Anadolu ve Doğu Trakya'da kurulmuş bir ülkedir.</input></|user|>
<|assistant|><thinking>Kullanıcı özet istiyor.</thinking>
<özet>Türkiye, Anadolu ve Doğu Trakya'da kurulmuş cumhuriyettir.</özet></|assistant|>"""
    
    print("Instruction örneği:")
    print(instruction_example)
    print()
    
    inst_tokens = sp.encode(instruction_example, out_type=str)
    inst_ids = sp.encode(instruction_example, out_type=int)
    
    print(f"📋 Toplam token sayısı: {len(inst_tokens)}")
    print(f"📊 Ortalama token uzunluğu: {len(instruction_example) / len(inst_tokens):.2f} karakter/token")
    
    # Decode kontrolü
    decoded_inst = sp.decode(inst_ids)
    print(f"✅ Instruction decode {'başarılı' if instruction_example == decoded_inst else 'başarısız'}")
    print()
    
    # Performans testi
    print("⚡ Performans Testi:")
    print("-" * 20)
    
    import time
    
    test_text = "Bu bir performans testidir. " * 100  # 100 kez tekrarla
    
    # Encoding performance
    start_time = time.time()
    for _ in range(100):
        tokens = sp.encode(test_text, out_type=int)
    encoding_time = time.time() - start_time
    
    # Decoding performance
    start_time = time.time()
    for _ in range(100):
        decoded = sp.decode(tokens)
    decoding_time = time.time() - start_time
    
    print(f"📝 Test metni uzunluğu: {len(test_text)} karakter")
    print(f"🔢 Token sayısı: {len(tokens)}")
    print(f"⚡ Encoding: {encoding_time:.4f}s (100 işlem)")
    print(f"⚡ Decoding: {decoding_time:.4f}s (100 işlem)")
    print(f"📊 Encoding hızı: {len(test_text) * 100 / encoding_time:.0f} karakter/saniye")
    print()
    
    print("🎉 Tüm testler tamamlandı!")
    print("✅ Tokenizer başarıyla çalışıyor!")

def interactive_test():
    """İnteraktif test modu"""
    
    model_file = Path('turkish_tokenizer/turkish_tokenizer.model')
    
    if not model_file.exists():
        print("❌ Model dosyası bulunamadı!")
        return
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))
    
    print("🔧 İnteraktif Tokenizer Test Modu")
    print("=" * 40)
    print("Çıkmak için 'exit' yazın")
    print()
    
    while True:
        try:
            text = input("🇹🇷 Test metni girin: ").strip()
            
            if text.lower() in ['exit', 'çık', 'quit']:
                print("👋 Görüşürüz!")
                break
            
            if not text:
                continue
            
            # Tokenize
            tokens = sp.encode(text, out_type=str)
            token_ids = sp.encode(text, out_type=int)
            
            print(f"📋 Tokenlar: {tokens}")
            print(f"🔢 Token IDs: {token_ids}")
            print(f"📊 Token sayısı: {len(tokens)}")
            print(f"📏 Sıkıştırma oranı: {len(text) / len(tokens):.2f}")
            
            # Decode
            decoded = sp.decode(token_ids)
            print(f"✅ Decode: {decoded}")
            print(f"🎯 Doğru: {'Evet' if text == decoded else 'Hayır'}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Görüşürüz!")
            break
        except Exception as e:
            print(f"❌ Hata: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()
    else:
        test_turkish_tokenizer() 