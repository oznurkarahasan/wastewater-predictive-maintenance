# Aşırı Öğrenme (Overfitting) Problemi - Çözüm Raporu

## 📋 Tespit Edilen Kritik Sorunlar

### 1. Model Hiç Arıza Yakalayamıyor (Recall = 0.00)
**Sorun:** Notebook 4'te tüm threshold değerlerinde (0.1-0.3) model **hiç pozitif tahmin üretmiyor**
- Test setindeki 1441 arıza sinyalinden **SIFIR** tanesini yakalıyor
- Bu durum modelin tamamen başarısız olduğunu gösteriyor

**Neden:**
- Class imbalance düzgün yönetilmemiş
- Model her zaman negatif sınıfı tahmin ediyor
- Threshold optimizasyonu yapılmamış

### 2. Veri Sızıntısı (Data Leakage)
**Sorun:** Zaman serisi için kritik hatalar

**Notebook 4 - Son hücre (cell 11):**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=True, stratify=y, random_state=42  # ❌ YANLIŞ!
)
```

**Neden yanlış:**
- `shuffle=True` → Zaman sırasını bozuyor
- Gelecekteki veriler train setine karışıyor
- Model gerçekte olmayan bilgiyi öğreniyor
- Test performansı yanıltıcı yüksek gözükebilir

**Doğru yaklaşım:**
- Temporal split kullanılmalı (tarih bazlı)
- `shuffle=False` olmalı
- TimeSeriesSplit ile validation yapılmalı

### 3. Aşırı Feature Engineering
**Sorun:** 52 sensörden **357 yeni özellik** türetilmiş (toplam 409 sütun)

**Detay:**
- 3 pencere (3h, 12h, 24h) × 2 metrik (mean, std) × 52 sensör = 312 özellik
- 52 sensör × diff = 52 özellik
- Ham sensörler = 52 özellik
- **TOPLAM: 409 özellik**

**Sorunlar:**
- Çok fazla özellik → Model karmaşıklaşıyor
- Gereksiz özellikler → Overfitting artıyor
- Computational cost yüksek
- Bazı sensörlerin arıza ile korelasyonu çok düşük

### 4. Temporal Validation Eksikliği
**Sorun:** TimeSeriesSplit kullanılmamış

**Mevcut durum:**
- Optuna optimizasyonunda basit %80-%20 split
- Temporal order göz ardı edilmiş
- Overfitting tespiti yapılamamış

**Doğru yaklaşım:**
- TimeSeriesSplit (5-fold)
- Her fold'da gelecek tahmin edilmeli
- Cross-validation skorları raporlanmalı

### 5. Class Imbalance Yönetimi Yetersiz
**Sorun:** SMOTE veya undersampling denenmemiş

**Mevcut durum:**
- Sadece `scale_pos_weight` kullanılmış (80-150 aralığı)
- Bu tek başına yeterli olmamış
- Model pozitif sınıfı öğrenememiş

**Doğru yaklaşım:**
- SMOTE ile minority class oversampling
- Undersampling ile majority class azaltma
- Balanced dataset ile eğitim

### 6. Rolling Window Parametreleri
**Sorun:** `min_periods=1` kullanımı

**Notebook 3:**
```python
roll_mean = df_eng[col].rolling(window=w_size, min_periods=1).mean()  # ❌ Riskli
```

**Neden sorunlu:**
- min_periods=1 → İlk değerde bile hesaplama yapılıyor
- Yeterli veri olmadan özellik türetiliyor
- Veri sızıntısı riski

**Doğru yaklaşım:**
```python
min_periods=int(w_size * 0.5)  # En az %50 veri olmalı
```

---

## ✅ Uygulanan Çözümler

### Notebook 3: Feature Engineering İyileştirmeleri

#### 1. Feature Sayısı Azaltıldı
**Önce:** 357 özellik
**Sonra:** ~90 özellik (Azalma: %75)

**Nasıl:**
- En yüksek korelasyonlu 30 sensör seçildi
- Pencere sayısı: 3 → 2 (6h, 12h)
- Metrik sayısı: 2 → 1 (sadece mean, std kaldırıldı)

**Kod:**
```python
# Korelasyon analizi
correlations = {}
for col in sensor_cols:
    correlations[col] = abs(df[col].corr(temp_target))

# En önemli 30 sensör
top_sensors = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:30]
selected_sensors = [s[0] for s in top_sensors]
```

#### 2. Veri Sızıntısı Önlendi
**Önce:**
```python
min_periods=1  # Riskli
fillna(method='ffill')  # Gelecek bilgisi kullanılabilir
```

**Sonra:**
```python
min_periods=int(w_size * 0.5)  # En az %50 veri
dropna()  # Sadece baştan ve sondan kes
```

#### 3. Gereksiz Metrikler Kaldırıldı
**Kaldırılanlar:**
- Rolling std (3 pencere × 52 sensör = 156 özellik kaldırıldı)
- 24 saatlik pencere (çok uzun, arıza sinyallerini kaçırabilir)
- 3 saatlik pencere (çok kısa, gürültülü)

**Korunanlar:**
- Rolling mean (6h, 12h)
- Diff (1h)

### Notebook 4: Model Optimizasyonu İyileştirmeleri

#### 1. SMOTE + Undersampling
**Strateji:**
```python
SMOTE(sampling_strategy=0.3)  # Minority class'ı %30'a çıkar
RandomUnderSampler(sampling_strategy=0.5)  # 1:2 oranı
```

**Sonuç:**
- Balanced dataset
- Model artık pozitif sınıfı öğrenebilir

#### 2. Threshold Optimizasyonu
**Yöntem:**
- Precision-Recall Curve analizi
- F2 Score kullanımı (Recall'a 2x ağırlık)
- 0.05-0.95 aralığında optimal threshold arama

**Kod:**
```python
for thresh in np.arange(0.05, 0.95, 0.05):
    y_pred_temp = (y_prob > thresh).astype(int)
    f2 = fbeta_score(y_test, y_pred_temp, beta=2)

best_threshold = f2_scores[np.argmax(f2_scores[:, 1]), 0]
```

#### 3. TimeSeriesSplit Validation
**Yöntem:**
```python
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X_train):
    # Temporal split
    # Her fold'da gelecek tahmin edilir
    # Overfitting tespiti
```

**Faydalar:**
- Temporal integrity korunur
- Overfitting tespit edilir
- Daha güvenilir metrikler

#### 4. Temporal Split (Shuffle Kaldırıldı)
**Önce:**
```python
shuffle=True, stratify=y  # ❌ Zaman serisini bozuyor
```

**Sonra:**
```python
X_train = X.loc[X.index < split_date]  # ✅ Tarih bazlı
X_test = X.loc[X.index >= split_date]
```

#### 5. Regularization Eklendi
**Parametreler:**
```python
reg_alpha=0.1,   # L1 regularization
reg_lambda=0.1,  # L2 regularization
```

**Fayda:**
- Overfitting azalır
- Model daha genelleşebilir

---

## 📊 Beklenen İyileşmeler

### Metrik Karşılaştırması

| Metrik | Önceki | Beklenen Yeni | İyileşme |
|--------|--------|---------------|----------|
| **Recall** | 0.00 | > 0.70 | +%70 |
| **F1 Score** | 0.00 | > 0.50 | +%50 |
| **Precision** | N/A | 0.30-0.50 | - |
| **AUC-ROC** | ~0.50 | > 0.80 | +%30 |

### Neden Bu Hedefler?

**Recall > 0.70:**
- Arıza tespiti için en önemli metrik
- %70+ arıza yakalanmalı (1441'den en az 1000+)
- Kritik başarısızlıklar önlenmeli

**Precision 0.30-0.50:**
- False alarm kabul edilebilir
- Bir arızayı kaçırmak > 2-3 yanlış alarm
- Predictive maintenance doğası gereği

**F1 Score > 0.50:**
- Precision-Recall dengesi
- Makul bir performans göstergesi

---

## 🚀 Kullanım Talimatları

### 1. Optimize Edilmiş Notebook'ları Çalıştırma

#### Adım 1: Feature Engineering
```bash
# Notebook 3 - Optimize Versiyon
jupyter notebook notebooks/03_FeatureEngineering_Optimized.ipynb
```

**Beklenen çıktılar:**
- ~90 yeni özellik (357 yerine)
- `sensor_enriched_optimized.csv` oluşturulacak
- Korelasyon analizi sonuçları
- Trend görselleştirmeleri

#### Adım 2: Model Optimization
```bash
# Notebook 4 - Optimize Versiyon
jupyter notebook notebooks/04_ModelOptimization_Optimized.ipynb
```

**Beklenen çıktılar:**
- SMOTE + Undersampling sonuçları
- Threshold optimizasyonu
- TimeSeriesSplit CV skorları
- Feature importance analizi
- Final model performansı

### 2. Model Karşılaştırması

#### Eski Model:
```python
# Eski modeli yükle (isterseniz)
old_model = joblib.load('models/final_lgbm_model.pkl')
# Recall: 0.00
```

#### Yeni Model:
```python
# Yeni modeli yükle
new_model = joblib.load('models/final_lgbm_optimized.pkl')
config = joblib.load('models/model_config_optimized.pkl')

# Kullanım
threshold = config['best_threshold']
y_prob = new_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > threshold).astype(int)
```

---

## 🔍 Detaylı Teknik Analiz

### Neden Önceki Model Başarısız Oldu?

#### 1. Class Imbalance Dominant Oldu
**Veri dağılımı:**
- Normal: ~205,000 örnek (98%)
- Arıza: ~1,500 örnek (2%)

**Model davranışı:**
- "Her zaman 0 tahmin et" stratejisi
- Accuracy: %98 (yanıltıcı yüksek)
- Recall: 0.00 (tamamen başarısız)

**Neden:**
- LightGBM varsayılan loss function: binary cross-entropy
- Dengesiz veri için optimize değil
- Pozitif örnekleri görmezden geliyor

#### 2. Veri Sızıntısı → Yanlış Güven
**Shuffle kullanımı:**
```python
# Zaman: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Shuffle sonrası: [7, 2, 9, 1, 5, 3, 10, 4, 8, 6]
# Train: [7, 2, 9, 1, 5]  # Gelecek bilgisi içeriyor!
# Test: [3, 10, 4, 8, 6]
```

**Sonuç:**
- Model validation'da iyi görünebilir
- Ama gerçek dünyada başarısız olur
- Temporal pattern öğrenilemez

#### 3. Aşırı Karmaşıklık
**409 özellik problemi:**
- Model capacity: Yüksek
- Data size: 205K örnek
- Features: 409
- **Risk:** Model ezber yapar, genelleştiremez

**Hughes Phenomenon (Curse of Dimensionality):**
- Özellik sayısı artarken veri yetersiz kalır
- Model noise'i pattern olarak öğrenir
- Test performansı düşer

---

## 📈 İyileştirme Stratejisi Özeti

### 1. Veri Seviyesi
- ✅ Feature selection (korelasyon bazlı)
- ✅ Feature reduction (%75 azalma)
- ✅ Veri sızıntısı önlendi
- ✅ Temporal integrity korundu

### 2. Model Seviyesi
- ✅ SMOTE + Undersampling
- ✅ Threshold optimization
- ✅ Regularization (L1 + L2)
- ✅ Hiperparametre tuning

### 3. Validation Seviyesi
- ✅ TimeSeriesSplit
- ✅ Temporal split (shuffle yok)
- ✅ Multiple metrics (Recall, F1, F2, AUC)
- ✅ Cross-validation reporting

---

## ⚠️ Önemli Notlar

### 1. Precision vs Recall Trade-off
**Bu projede Recall öncelikli çünkü:**
- Bir arızayı kaçırmak maliyetli
- False alarm kabul edilebilir (bakım ekibi kontrol eder)
- Predictive maintenance doğası gereği

**Threshold'u düşürürseniz:**
- Recall artar (daha fazla arıza yakalar)
- Precision düşer (daha fazla false alarm)
- İş gereksinimlerine göre ayarlayın

### 2. Temporal Validation Şart
**Zaman serisi projelerinde:**
- Asla `shuffle=True` kullanmayın
- TimeSeriesSplit kullanın
- Gelecek tahmin edilmeli, geçmiş değil

### 3. Feature Engineering Denge İstiyor
**Fazla özellik:**
- Overfitting riski
- Computational cost
- Interpretability azalır

**Az özellik:**
- Underfitting riski
- Önemli pattern'ler kaçar

**Optimal yaklaşım:**
- Domain knowledge + Data-driven selection
- Iterative experimentation

---

## 🎯 Sonraki Adımlar

### Öncelikli (Bu Rapor Sonrası)
1. ✅ Optimize notebook'ları çalıştırın
2. ✅ Yeni model performansını test edin
3. ✅ Threshold'u iş gereksinimlerine göre ayarlayın

### Orta Vadeli
1. Feature importance'a göre daha fazla özellik temizliği
2. Ensemble methods (XGBoost, CatBoost kombinasyonu)
3. Anomaly detection eklemek (Isolation Forest, Autoencoder)

### Uzun Vadeli
1. Online learning (model güncelleme)
2. Real-time prediction API
3. Monitoring ve alerting sistemi
4. A/B testing framework

---

## 📚 Referanslar ve Kaynaklar

### Kullanılan Teknikler
1. **SMOTE:** Synthetic Minority Over-sampling Technique
2. **TimeSeriesSplit:** Sklearn temporal validation
3. **LightGBM:** Microsoft Gradient Boosting framework
4. **Precision-Recall Curve:** Threshold optimization
5. **F-beta Score:** Recall-weighted F-measure

### İlgili Makaleler
- Chawla et al. (2002): SMOTE - Synthetic Minority Over-sampling
- Bergmeir & Benítez (2012): On the use of cross-validation for time series
- Chen & Guestrin (2016): XGBoost - A Scalable Tree Boosting System

---

## ✅ Checklist

### Uygulama Öncesi
- [x] Mevcut notebook'ları inceledim
- [x] Sorunları tespit ettim
- [x] Çözüm stratejisi hazırladım
- [x] Optimize notebook'ları oluşturdum

### Uygulama Sırasında
- [ ] Notebook 3 Optimized çalıştırıldı
- [ ] Yeni feature set oluşturuldu
- [ ] Notebook 4 Optimized çalıştırıldı
- [ ] Yeni model eğitildi

### Uygulama Sonrası
- [ ] Recall > 0.70 sağlandı mı?
- [ ] CV skorları stabilmi?
- [ ] Feature importance incelendi mi?
- [ ] Threshold optimize edildi mi?
- [ ] Model kaydedildi mi?

---

**Rapor Tarihi:** 2026-01-01
**Hazırlayan:** Claude Code
**Versiyon:** 1.0
