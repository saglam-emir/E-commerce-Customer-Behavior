# E-Ticaret Platformlarında Doğal Dil İşleme ve Makine Öğrenmesi Tabanlı Müşteri Kayıp (Churn) Tahmini

Bu proje, Veri Madenciliği dersi dönemsel projesi kapsamında geliştirilmiştir. E-ticaret sektöründe müşteri kaybının (churn) erken tespiti amacıyla, yapılandırılmış demografik veriler ile yapılandırılmamış müşteri yorumlarını (NLP) birleştiren çok modlu (multi-modal) bir makine öğrenmesi boru hattı (pipeline) sunmaktadır.

## 🎯 Projenin Amacı
Geleneksel churn analizi çalışmaları genellikle yalnızca sayısal verilere (yaş, harcama tutarı vb.) dayanmaktadır. Bu projenin amacı, müşterinin asıl memnuniyetsizliğinin gizli olduğu **ürün yorumlarını (metin verisi)** doğal dil işleme teknikleriyle sayısallaştırarak klasik veri madenciliği modellerine entegre etmek ve tahmin başarısını artırmaktır.

Ayrıca, e-ticaret verilerinde sıklıkla karşılaşılan aşırı sınıf dengesizliği (imbalanced data) problemine karşı **SMOTE** algoritması uygulanarak modellerin azınlık sınıfını (churn) tespit etme duyarlılığı (recall) optimize edilmiştir.

## 📁 Klasör Yapısı
Proje, modüler ve tekrarlanabilir bir mimaride tasarlanmıştır:

* `data/`: Kaggle'dan alınan ham veri seti (`Womens Clothing E-Commerce Reviews.csv`) ve ön işlemlerden geçmiş temizlenmiş nihai veri setleri.
* `notebooks/`: Veri keşfi (EDA), NLP işlemleri, SMOTE uygulaması ve model eğitim süreçlerini barındıran Jupyter Notebook / Colab dosyaları.
* `src/`: Notebook'lar dışında kullanılan yardımcı Python betikleri (scriptler) ve dışa aktarılmış model dosyaları (`tfidf_vectorizer.pkl`).
* `visuals/`: Karmaşıklık matrisleri (Confusion Matrix), sınıf dağılım grafikleri ve Kelime Bulutu (Word Cloud) görselleri.
* `reports/`: Projenin tüm süreçlerini kapsayan akademik formatlı Final Raporu.
* `docs/`: Referans makaleler.

## 🛠️ Kullanılan Teknolojiler ve Yöntemler
* **Veri İşleme & Analiz:** `pandas`, `numpy`
* **Doğal Dil İşleme (NLP):** `nltk` (Stop-words, Lemmatization, VADER Sentiment Analysis), `scikit-learn` (TF-IDF Vectorization)
* **Makine Öğrenmesi:** `scikit-learn` (Logistic Regression, Random Forest), `imbalanced-learn` (SMOTE)
* **Görselleştirme:** `matplotlib`, `seaborn`, `wordcloud`

## 🚀 Proje İş Akışı
1. **Veri Ön İşleme:** Eksik verilerin temizlenmesi, veri sızıntısının (data leakage) önlenmesi için hedef değişkene benzeyen sütunların (`Rating`) veri setinden çıkarılması.
2. **Metin Madenciliği:** Müşteri yorumlarının küçük harfe dönüştürülmesi, noktalama işaretlerinin temizlenmesi ve Lemmatization ile köklerine indirgenmesi.
3. **Özellik Mühendisliği:** Metinlerden `Sentiment_Score` (Duygu Skoru) ve `Word_Count` (Kelime Sayısı) türetilmesi.
4. **Vektörizasyon:** Temizlenmiş kelimelerin TF-IDF ile 1000 sütunluk sayısal matrislere dönüştürülmesi.
5. **Modelleme & SMOTE:** Lojistik Regresyon ve Random Forest algoritmalarının eğitilmesi. Aşırı sınıf dengesizliğine karşı SMOTE uygulanarak modelin azınlık sınıfını bulma başarısının (Recall) maksimize edilmesi.

## 📊 Deneysel Sonuçlar
Veri setindeki azınlık sınıfını (Churn) tahmin etme üzerine kurulan senaryoda, doğrusal yapıdaki modellerin yüksek boyutlu metin verilerinde çok daha başarılı olduğu görülmüştür.
* **SMOTE Öncesi:** Temel modeller çoğunluk sınıfını ezberleme eğilimi göstermiş, Churn yakalama oranı %33 - %54 bandında kalmıştır.
* **SMOTE Sonrası (Lojistik Regresyon):** Sentetik veri artırımı ile dengelenen eğitim setinde modelin Churn duyarlılığı (Recall) **%82** seviyesine çıkmış ve e-ticaret firmaları için kullanılabilir bir karar destek sistemi performansı elde edilmiştir.