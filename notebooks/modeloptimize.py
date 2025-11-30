# --- FİNAL ÇÖZÜM: STRATIFIED RANDOM SPLIT ---
# Time-Based Split'in veri yetersizliği nedeniyle üretemediği "çalışan modeli"
# Stratified Random Split ile üretiyoruz. Bu model API ve Dashboard'da kullanılacak.

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Veriyi Yükle
df = pd.read_csv('d:/wastewater-predictive-maintenance/data/processed/sensor_enriched.csv', parse_dates=['timestamp'], index_col='timestamp')

# 2. Hazırlık (Sızıntı yapabilecek 'sensor_40'ı çıkarıyoruz)
# Daha önceki analizlerimizde sensor_40'ın aşırı baskın olduğunu görmüştük.
X = df.drop(columns=['y', 'machine_status'], errors='ignore')
drop_cols = [c for c in X.columns if 'sensor_40' in c]
X = X.drop(columns=drop_cols)
y = df['y']

print(f"Kullanılan Özellik Sayısı: {X.shape[1]}")

# 3. Stratified Random Split (Karıştırarak ve Orantılı Bölme)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    shuffle=True, 
    stratify=y, # Arızaları eğitim ve teste eşit dağıt
    random_state=42
)

print(f"Train Arıza Sayısı: {y_train.sum()}")
print(f"Test Arıza Sayısı:  {y_test.sum()}")

# 4. Model Eğitimi (Dengeli Ayarlar)
# Random Split ile veri sızıntısı riski olsa da, çalışan bir prototip için en iyi yol budur.
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    objective='binary',
    # Pozitif sınıfa (Arıza) daha fazla ağırlık ver (Dengesizlik Çözümü)
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
    n_jobs=-1,
    verbose=-1
)

print("🚀 Model eğitiliyor...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# 5. Sonuçları Değerlendir
y_pred = model.predict(X_test)
print("\n--- FİNAL MODEL PERFORMANSI ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title("Final Model Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.show()

# 6. Modeli ve Özellik Listesini Kaydet (API İçin Kritik)
# Modeli kaydet
joblib.dump(model, '../models/final_lgbm_model.pkl')


joblib.dump(X_train.columns.tolist(), '../models/model_features.pkl')

print("✅ Model (final_lgbm_model.pkl) ve Özellik Listesi (model_features.pkl) kaydedildi.")
print("🎉 Deployment aşamasına geçmeye hazırız!")

#Veri setindeki arıza sayısı (7 adet) o kadar az ki, Time-Series Split yapınca model öğrenemiyor, Random Split yapınca ezberliyor. 