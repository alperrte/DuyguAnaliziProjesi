# Karşılaştırmalı Duygu Analizi Kullanım Kılavuzu

Komutları Windows PowerShell’de, `src`, `data` ve `requirements.txt` dosyasının bulunduğu proje kökünden çalıştırın.

## Hazır sonuçları inceleme

Teslim paketindeki sonuçları incelemek için kurulum veya API anahtarı gerekmez.

| Dosya | İçerik |
| --- | --- |
| [Word raporu](teslim/Proje_Raporu.docx) | Veri kaynakları, yöntemler ve bulgular |
| [Yöntem özeti](result/yontem_ozeti.csv) | İki dilde doğruluk, makro F1 ve hata adetleri |
| [Sınıf bazında hatalar](result/sinif_bazinda_hatalar.csv) | Pozitif, negatif ve nötr sınıflar için hata, kesinlik, duyarlılık ve F1 |
| [Karışıklık matrisleri](result/karisiklik_matrisleri.png) | Gerçek etiketlerin hangi sınıflarla karıştığını gösteren altı matris |

## Kurulum

Terminalin başında `(.venv)` görünüyorsa sanal ortam zaten etkin. Temiz kurulum için Python 3.12 ile aşağıdaki adımları uygulayın.

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip check
```

Yukarıdaki PyTorch komutu NVIDIA GPU için CUDA 12.8 paketini kurar. CPU ile çalışacaksanız kurulum adresindeki `cu128` bölümünü `cpu` olarak değiştirin.

Aşağıdaki `python -m ...` komutlarını `(.venv)` görünen terminalde çalıştırın. PowerShell etkinleştirme betiğini engelliyorsa her komutta `python` yerine `& .\.venv\Scripts\python.exe` kullanın.

## VADER’ı çalıştırma

Sanal ortam etkin ve veri dosyaları hazırsa şu komutu çalıştırın:

```powershell
python -m src.vader
```

Sanal ortamı etkinleştirmeden de `& .\.venv\Scripts\python.exe -m src.vader` komutuyla çalıştırabilirsiniz. VADER özgün İngilizce metinleri ve Türkçe kümenin İngilizce çevirilerini değerlendirir. Tahminleri `result/vader/` altına yazar. Çeviri dosyasını yeniden üretmek için VADER’dan önce `python -m src.translate` komutunu çalıştırın.

## VADER, BERT ve Claude’u karşılaştırma

Teslim paketinde üç yöntemin tahminleri hazır. Altı deneyin tablolarını ve grafiklerini yeniden üretmek için şu komutu çalıştırın:

```powershell
python -m src.comparison
```

Komut `result/vader/`, `result/bert/` ve `result/claude/` altındaki İngilizce ve Türkçe tahminleri özgün etiketlerle eşleştirir. 18 sınıf bazlı hata sayısını hesaplar ve tabloları `result/` içine yazar. Bir tahmin dosyası eksikse komut dosya adını bildirir.

## Deneyleri baştan çalıştırma

Claude için proje kökünde `.env` dosyası yoksa şablondan oluşturun. Ardından `.env` dosyasındaki `ANTHROPIC_API_KEY=` satırına kendi anahtarınızı ekleyin. `.env.example` paylaşılabilir bir şablondur.

```powershell
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
```

Komutları sırayla çalıştırın. Veri kaynaklarına ve model ağırlıklarına ilk erişimde internet gerekir.

| Komut | Ne yapar? |
| --- | --- |
| `python -m src.prepare` | Hazır kaynak etiketlerini kullanarak her dilde 1.000 pozitif, 1.000 negatif ve 1.000 nötr metin seçer. |
| `python -m src.translate` | Türkçe kayıtları OPUS-MT ile İngilizceye çevirir. Sonuç `data/turkce_ingilizce_ceviri.csv` dosyasına yazılır. |
| `python -m src.vader` | Özgün İngilizce metinleri ve Türkçe kümenin İngilizce çevirilerini değerlendirir. |
| `python -m src.bert` | Tek çalıştırmada özgün İngilizce metinleri RoBERTa, ardından özgün Türkçe metinleri BERTurk ile değerlendirir. |
| `python -m src.claude --wait` | İki dilin özgün metinleri için Claude tahminlerini hazırlar ve toplu işlemin tamamlanmasını bekler. |
| `python -m src.comparison` | Üç yöntemin hazır tahminlerini karşılaştırır. |

Claude, doğrulanmış tahminler varsa onları yeniden kullanır. Devam eden bir toplu işlem varsa kayıtlı işlem üzerinden sürdürür. Yeni bir işlem gönderildiğinde Anthropic API ücreti uygulanır.

`python -m src.claude` komutu `--wait` olmadan gerekirse toplu işlemi başlatır ve durumunu bir kez sorgular.
