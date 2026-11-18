# Karşılaştırmalı Duygu Analizi

Veri Madenciliği dersi projesi. Özgün İngilizce ve özgün Türkçe tweetler, pozitif, negatif ve nötr sınıflarında VADER, BERT tabanlı modeller ve Claude Haiku 5.5 ile karşılaştırılır.

## Veri dosyaları

| Dosya | İçerik |
| --- | --- |
| `data/ingilizce.csv` | TweetEval Sentiment test bölümünden 3.000 özgün İngilizce örnek |
| `data/turkce.csv` | Sercan Külcü’nün hazır etiketli veri kümesinden 3.000 özgün Türkçe örnek |
| `data/turkce_ingilizce_ceviri.csv` | Aynı Türkçe örneklerin İngilizce çevirileri, `text_en` sütununda |
| `data/kaynak_manifestosu.json` | Sabit kaynak sürümleri, SHA-256 özetleri ve örnekleme sayıları |
| `data/ceviri_manifestosu.json` | OPUS-MT sürümü, parametreleri ve çıktı özeti |

Her özgün dosyada gerçek sınıf başına 1.000 örnek vardır. `id`, `source_file` ve `source_line` kayıtları kaynağa kadar izlemeyi sağlar. Çeviri sırasında Türkçe satırlar, etiketler ve kimlikler korunur. Örnekleme model çıktılarından önce gerçekleştirilir.

- [TweetEval Sentiment](https://github.com/cardiffnlp/tweeteval/tree/4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66/datasets/sentiment)
- [Türkçe tweet verisi](https://github.com/sercankulcu/sentiment-analysis-of-tweets-in-Turkish/tree/9647a08a9f0f6ac117d6f79a65578315b27682d9/training%20data)

## Kurulum ve çalıştırma

Python 3.12 ve PyTorch 2.9.1 kullanıldı. Windows ve NVIDIA GPU için aşağıdaki örnek CUDA 12.8 paketini kurar. CPU ortamında PyTorch adresinin sonundaki `cu128` yerine `cpu` kullanılabilir.

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip check
```

Sanal ortam etkinken projenin kök klasöründen çalıştırın. Etkinleştirme mümkün değilse aşağıdaki komutlarda `python` yerine `& .\.venv\Scripts\python.exe` kullanın. İlk çalıştırmada GitHub’dan veri ve Hugging Face’ten model ağırlıkları indirilir.

```powershell
python -m src.prepare
python -m src.translate
python -m src.vader
python -m src.bert
python -m src.claude --wait
python -m src.comparison
```

Çeviri kodu `src/translate/` altında, sınıflandırıcılar `src/vader/`, `src/bert/` ve `src/claude/` altında, değerlendirme kodu `src/comparison/` altındadır. VADER özgün İngilizce metinleri ve Türkçe kümenin İngilizce çevirilerini işler. BERT ile Claude her iki dilin özgün metinlerini işler. `src.bert` komutu İngilizce RoBERTa ve Türkçe BERTurk analizlerini tek çalıştırmada tamamlar.

Claude çağrıları ücretli Anthropic Message Batches API üzerinden yapılır. Anahtarı `ANTHROPIC_API_KEY` ortam değişkeniyle veya Git tarafından yok sayılan yerel `.env` dosyasıyla verin. Bir grup zaten gönderildiyse aynı komut kayıtlı grup kimliğiyle durum sorgusuna devam eder. `--wait` olmadan komut bir kez durum sorgulayıp döner. Geçici durum bilgileri `.cache/` altında tutulur.

## Sonuçlar

Tüm çıktılar `result/` altında yer alır. Yöntem özeti, sınıf bazında hata ve F1 değerleri, karışıklık matrisleri, grafikler ve örnek hatalar klasörün içindedir.

- `result/vader/`, `result/bert/` ve `result/claude/` altındaki CSV’ler satır kimliğine göre gerçek etiket ve yöntem tahminini tutar.
- `result/claude/claude_manifestosu.json` kullanılan istemi ve tamamlanan tahmin dosyalarının SHA-256 özetlerini içerir. Claude’un etiket üretmeyi reddettiği örnekler `abstain` ile gösterilir ve ilgili gerçek sınıfın hata sayısına dahil edilir.

Dengeli örnekleme nedeniyle değerlendirme her sınıfa eşit ağırlık verir. Türkçe VADER sonucunda çevirinin etkisi de vardır. Diller arasındaki farklar yorumlanırken kaynak, konu, etiketleme ve BERT modelinin de değiştiği göz önüne alınmalıdır.
