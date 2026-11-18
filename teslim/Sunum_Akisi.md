# Karşılaştırmalı Duygu Analizi Sunum Akışı

Hedef süre 8 dakika 45 saniye. Üç grup üyesi de kendi bölümünü anlatır ve ekrandaki proje çıktıları üzerinden bir örnek gösterir. Kayda başlamadan önce `teslim/Proje_Raporu.docx`, `result/karisiklik_matrisleri.png` ve `result/hata_ornekleri.csv` dosyalarını açın.

## 1. Grup üyesi · Veri ve amaç · 0:00–2:30

- Proje sorusunu söyleyin. Kural tabanlı bir yöntem, iki dil için BERT tabanlı sınıflandırıcılar ve Claude aynı üç duygu sınıfında nasıl performans gösterdi?
- Raporun veri tablosunu gösterin. TweetEval test bölümünden 3.000 İngilizce, Türkçe tweet kaynağından 3.000 Türkçe kayıt seçildi. Her dilde 1.000 pozitif, 1.000 negatif ve 1.000 nötr örnek var.
- Türkçe özgün dosya ile çeviri dosyasından tek bir satırı gösterin. Aynı `id` ve kaynak etiketi çeviride korunur. Üç CSV dosyası toplam 6.000 özgün metin içerir.

## 2. Grup üyesi · Yöntemler · 2:30–5:15

- VADER’ın sözlük ve kural kullanımını açıklayın. Pozitif eşik 0,05, negatif eşik −0,05. Türkçe VADER adımı OPUS-MT çevirileriyle çalışır.
- İngilizce RoBERTa ve Türkçe BERTurk’ün özgün metinlerde üç olasılık üretip en yüksek olasılıklı etiketi seçtiğini gösterin.
- Claude Haiku 5.5’in her istekte bir metin aldığını, ortak istem ve JSON şemasıyla üç etiketten birini döndürdüğünü anlatın. Ekranda API anahtarı bulunmasın.

## 3. Grup üyesi · Sonuçlar ve örnek hata · 5:15–8:45

- Rapor sonuç tablosunu açın. Her dilde makro F1’i en yüksek yöntemi ve doğruluk değerini söyleyin. Gerçek sınıf başına 1.000 kayıt üzerinden hata sayısının nasıl hesaplandığını açıklayın.
- Karışıklık matrisinde farklı iki sınıfı birleştiren en büyük hücrelerden birini gösterin. Satır gerçek etikettir, sütun modelin tahminidir.
- Rapordaki hata örneklerinden birini özgün metin, kaynak etiketi ve model tahmini üzerinden anlatın. Türkçe VADER sonucunun çeviriyi de ölçtüğünü, veri kaynaklarının ve BERT modellerinin diller arasında değiştiğini belirtin.
- Ortak sonuç cümlesiyle bitirin. Sınıf başına hata sayısı, toplam doğruluğun hangi duygulardaki karışıklıkları gizlediğini gösterir.
