import PyPDF2
import re
import string
import os

def extract_text_from_pdf(pdf_path):
    """PDF dosyasından text çıkarır."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            print(f"PDF'de toplam {len(pdf_reader.pages)} sayfa bulundu.")
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += page_text + "\n"
                
                if (page_num + 1) % 10 == 0:
                    print(f"{page_num + 1} sayfa işlendi...")
                    
    except Exception as e:
        print(f"PDF okuma hatası: {e}")
        return ""
    
    return text

def clean_text(text):
    """Text'i modeli eğitmek için temizler."""
    
    # 1. Boş satırları kaldır
    text = text.strip()
    
    # 2. Birden fazla boşluğu tek boşluğa çevir
    text = re.sub(r'\s+', ' ', text)
    
    # 3. Sayfa numaralarını ve header/footer'ları kaldır
    # Sayfa başında/sonunda olan sayıları kaldır
    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'^\d+\s*', '', text, flags=re.MULTILINE)
    
    # 4. Çok kısa satırları kaldır (muhtemelen header/footer)
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # En az 20 karakter olan satırları tut
        if len(line) >= 20:
            cleaned_lines.append(line)
    
    # 5. Satırları birleştir
    text = ' '.join(cleaned_lines)
    
    # 6. Çoklu noktalama işaretlerini düzelt
    text = re.sub(r'\.{2,}', '.', text)  # ... -> .
    text = re.sub(r',{2,}', ',', text)   # ,, -> ,
    text = re.sub(r';{2,}', ';', text)   # ;; -> ;
    
    # 7. Garip karakterleri temizle (optional - Türkçe karakterleri koru)
    # Sadece ASCII olmayan ama Türkçe olmayan karakterleri kaldır
    allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + 
                       'çğıöşüÇĞIİÖŞÜ' + ' \n\t')
    text = ''.join(char for char in text if char in allowed_chars)
    
    # 8. Son temizlik - birden fazla boşluğu tekrar düzelt
    text = re.sub(r'\s+', ' ', text)
    
    # 9. Paragraf yapısını koru - nokta sonrası büyük harfle başlayan yerlerde satır sonu ekle
    text = re.sub(r'\. ([A-ZÇĞIÖŞÜ])', r'.\n\n\1', text)
    
    return text.strip()

def save_to_txt(text, output_path):
    """Temizlenmiş text'i txt dosyasına kaydet."""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Text başarıyla {output_path} dosyasına kaydedildi.")
        
        # İstatistikleri yazdır
        print(f"\nText İstatistikleri:")
        print(f"- Toplam karakter sayısı: {len(text):,}")
        print(f"- Toplam kelime sayısı: {len(text.split()):,}")
        print(f"- Toplam satır sayısı: {text.count(chr(10)) + 1:,}")
        
        # Unique karakter sayısını hesapla
        unique_chars = set(text)
        print(f"- Unique karakter sayısı: {len(unique_chars)}")
        print(f"- Vocabulary: {''.join(sorted(unique_chars))}")
        
    except Exception as e:
        print(f"Dosya kaydetme hatası: {e}")

def main():
    """Ana fonksiyon."""
    pdf_path = "Hakan-Gunday-Kinyas-ve-Kayra.pdf"
    output_path = "kinyas_kayra_clean.txt"
    
    print("🔍 PDF'den text çıkarılıyor...")
    raw_text = extract_text_from_pdf(pdf_path)
    
    if not raw_text:
        print("❌ PDF'den text çıkarılamadı!")
        return
    
    print(f"✅ {len(raw_text):,} karakter çıkarıldı.")
    
    print("\n🧹 Text temizleniyor...")
    clean_text_result = clean_text(raw_text)
    
    print(f"✅ Text temizlendi. {len(clean_text_result):,} karakter kaldı.")
    
    print("\n💾 Dosyaya kaydediliyor...")
    save_to_txt(clean_text_result, output_path)
    
    print("\n🎉 İşlem tamamlandı!")
    print(f"📄 Temizlenmiş text: {output_path}")
    print("🤖 Artık bu dosyayla modelinizi eğitebilirsiniz!")

if __name__ == "__main__":
    main()
