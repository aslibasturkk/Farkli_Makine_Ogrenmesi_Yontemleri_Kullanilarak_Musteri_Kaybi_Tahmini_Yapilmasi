# Farkli_Makine_Ogrenmesi_Yontemleri_Kullanilarak_Musteri_Kaybi_Tahmini_Yapilmasi
Bu proje, bir dizi-film akış platformundaki müşterilerin aboneliklerini iptal edip etmeyeceğini (churn) makine öğrenmesi modelleriyle tahmin etmek amacıyla geliştirilmiştir.

---

## 📌 Proje Hakkında

Müşteri elde tutma, abonelik tabanlı hizmet veren platformlar için kritik bir süreçtir. Bu çalışmada, kullanıcıların izleme alışkanlıkları, ödeme yöntemleri ve hesap aktiviteleri analiz edilerek potansiyel terk durumları önceden tespit edilmeye çalışılmıştır.

* **Veri Seti:** [Kaggle - Netflix Customer Churn Dataset](https://www.kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset)
* **Veri Boyutu:** 5,000 Kayıt, 14 Öznitelik (Feature)
* **Problem Tipi:** İkili Sınıflandırma (Binary Classification)

---

##Kullanılan Kütüphaneler

* **Dil:** Python
* **Veri Analizi & İşleme:** Pandas, NumPy
* **Veri Görselleştirme:** Matplotlib, Seaborn
* **Makine Öğrenmesi:** Scikit-learn
* **Veri İndirme:** Kagglehub, Glob

---

## Öznitelikler

* `Age`: Kullanıcı yaşı
* `Gender`: Cinsiyet
* `Subscription_type`: Abonelik türü
* `watch_hours`: Toplam izleme saati
* `last_login_days`: Son girişten bu yana geçen gün sayısı
* `region`: Bölge/Ülke
* `device`: Kullanılan cihaz
* `monthly_fee`: Aylık ücret
* `payment_method`: Ödeme yöntemi
* `number_of_profiles`: Profil sayısı
* `avg_watch_time_per_day`: Günlük ortalama izleme süresi
* `favorite_genre`: Favori tür
* `churned` **(Hedef Değişken):** Abonelik iptal durumu (1: İptal Etti, 0: Devam Ediyor)

---

## Uygulama Adımları

1. **Veri Yükleme ve Ön İşleme:**
   * `kagglehub` ve `glob` kullanılarak veri seti dinamik olarak içe aktarıldı.
   * Eksik ve tutarsız değer kontrolleri yapıldı.
   * Modellerde kullanılmayan `Customer_id` sütunu çıkarıldı.
   * Kategorik değişkenler `pd.get_dummies()` yöntemiyle (One-Hot Encoding) sayısal formata dönüştürüldü.

2. **Model Eğitimi:**
   * Veri seti %80 Eğitim (Train) ve %20 Test olarak ayrıldı (`train_test_split`).
   * **Lojistik Regresyon (Logistic Regression)**, **Karar Ağacı (Decision Tree)** ve **Rassal Orman (Random Forest)** algoritmaları eğitildi.

3. **Değerlendirme:**
   * Modellerin başarıları Accuracy (Doğruluk) skorları ve Karmaşıklık Matrisleri (Confusion Matrix) ile karşılaştırıldı.

## Öne Çıkan Bulgular

* **Lojistik Regresyon**, sınıflar arasındaki doğrusal ilişkileri en iyi yakalayan model olarak %88.8 doğruluk oranıyla en yüksek başarıyı göstermiştir.
* Platformdaki müşterilerin yaklaşık **%49.7'si** churn riski taşımaktadır. Başarılı sınıflandırma modeli sayesinde terk etme eğilimi gösteren müşteriler önceden tespit edilerek proaktif promosyon ve indirim kampanyaları kurgulanabilir.

---
