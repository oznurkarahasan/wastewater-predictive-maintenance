"""
Hızlı Test - Model ve Threshold Optimizasyonu
==============================================
Bu script optimize edilmiş modeli hızlıca test eder
"""

import os
import sys

print("="*70)
print("🧪 HIZLI TEST - Threshold Optimization Validasyonu")
print("="*70)

# Gerekli kütüphaneleri kontrol et
try:
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score
    from imblearn.over_sampling import SMOTE
    print("✅ Tüm kütüphaneler yüklü")
except ImportError as e:
    print(f"❌ Eksik kütüphane: {e}")
    print("\nGerekli kütüphaneler:")
    print("  pip install pandas numpy scikit-learn lightgbm imbalanced-learn matplotlib seaborn")
    sys.exit(1)

# Veri yolu kontrolü
data_paths = [
    '/home/user/wastewater-predictive-maintenance/data/processed/sensor_enriched.csv',
    'data/processed/sensor_enriched.csv',
    '../data/processed/sensor_enriched.csv',
    'd:/wastewater-predictive-maintenance/data/processed/sensor_enriched.csv'
]

data_path = None
for path in data_paths:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    print("\n⚠️ UYARI: sensor_enriched.csv bulunamadı!")
    print("\nVeri dosyası yolları kontrol edildi:")
    for path in data_paths:
        print(f"  ❌ {path}")
    print("\nÇözüm:")
    print("  1. Veri dosyasını uygun bir yere kopyalayın")
    print("  2. Veya script içindeki data_paths listesine doğru yolu ekleyin")
    sys.exit(1)

print(f"✅ Veri bulundu: {data_path}")

# Hızlı veri analizi
print("\n" + "="*70)
print("📊 VERİ ANALİZİ")
print("="*70)

df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
X = df.drop(columns=['y', 'machine_status'], errors='ignore')
drop_cols = [c for c in X.columns if 'sensor_40' in c]
X = X.drop(columns=drop_cols)
y = df['y']

print(f"Toplam Örnek: {len(y):,}")
print(f"Class 0: {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.2f}%)")
print(f"Class 1: {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.2f}%)")
print(f"İmbalance Ratio: {(y==0).sum() / (y==1).sum():.1f}:1")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=True, stratify=y, random_state=42
)

print("\n" + "="*70)
print("🔬 HIZLI TEST 1: Class Weight Optimization")
print("="*70)

# Basit class weight testi
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    scale_pos_weight=50,  # Yüksek weight
    objective='binary',
    n_jobs=-1,
    verbose=-1,
    random_state=42
)

print("Model eğitiliyor...")
model.fit(X_train, y_train)

# Farklı threshold'larda test
y_proba = model.predict_proba(X_test)[:, 1]

print(f"\nOlasılık İstatistikleri:")
print(f"  Min:  {y_proba.min():.6f}")
print(f"  Max:  {y_proba.max():.6f}")
print(f"  Mean: {y_proba.mean():.6f}")
print(f"  Std:  {y_proba.std():.6f}")

print("\nThreshold Testi:")
print(f"{'Threshold':>10s} {'Tahmin':>10s} {'Precision':>12s} {'Recall':>10s} {'F1':>10s}")
print("-" * 60)

from sklearn.metrics import precision_score, recall_score, f1_score

best_f1 = 0
best_threshold = 0.05

for th in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1]:
    y_pred = (y_proba >= th).astype(int)
    pred_count = y_pred.sum()

    if pred_count > 0:
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
            marker = " 🎯"

        print(f"{th:10.3f} {pred_count:10d} {prec:12.3f} {rec:10.3f} {f1:10.3f}{marker}")
    else:
        print(f"{th:10.3f} {'0':>10s} {'N/A':>12s} {'N/A':>10s} {'N/A':>10s}")

print(f"\n🏆 En İyi Threshold: {best_threshold:.3f} (F1={best_f1:.3f})")

# En iyi threshold ile detaylı rapor
y_pred_best = (y_proba >= best_threshold).astype(int)

print("\n" + "="*70)
print(f"📊 DETAYLI PERFORMANS RAPORU (Threshold={best_threshold:.3f})")
print("="*70)
print(classification_report(y_test, y_pred_best))

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_best)
print("\nConfusion Matrix:")
print(f"                 Predicted")
print(f"              0           1")
print(f"Actual 0   {cm[0,0]:6d}    {cm[0,1]:6d}")
print(f"       1   {cm[1,0]:6d}    {cm[1,1]:6d}")

tn, fp, fn, tp = cm.ravel()
print(f"\nDetaylı Metrikler:")
print(f"  True Positives:  {tp:4d} (Doğru tespit edilen arızalar)")
print(f"  False Positives: {fp:4d} (Yanlış alarm)")
print(f"  True Negatives:  {tn:4d} (Doğru normal)")
print(f"  False Negatives: {fn:4d} (Kaçırılan arızalar)")

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n  Sensitivity (Recall): {sensitivity:.3f}")
print(f"  Specificity:          {specificity:.3f}")

# SMOTE testi
print("\n" + "="*70)
print("🧬 HIZLI TEST 2: SMOTE")
print("="*70)

try:
    print("SMOTE uygulanıyor...")
    k_neighbors = min(5, y_train.sum() - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"SMOTE Öncesi: Class 0={y_train.value_counts()[0]}, Class 1={y_train.value_counts()[1]}")
    print(f"SMOTE Sonrası: Class 0={(y_train_smote==0).sum()}, Class 1={(y_train_smote==1).sum()}")

    model_smote = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        objective='binary',
        n_jobs=-1,
        verbose=-1,
        random_state=42
    )

    model_smote.fit(X_train_smote, y_train_smote)
    y_proba_smote = model_smote.predict_proba(X_test)[:, 1]

    # En iyi threshold bul
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_smote)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])
    optimal_threshold_smote = thresholds[best_idx]

    y_pred_smote = (y_proba_smote >= optimal_threshold_smote).astype(int)

    print(f"\n🎯 SMOTE Optimal Threshold: {optimal_threshold_smote:.4f}")
    print(f"   F1 Score: {f1_scores[best_idx]:.3f}")

    print("\nSMOTE Model Performansı:")
    print(classification_report(y_test, y_pred_smote))

except Exception as e:
    print(f"❌ SMOTE Hatası: {e}")

# Sonuç
print("\n" + "="*70)
print("✅ TEST TAMAMLANDI")
print("="*70)

print("\n📝 ÖNERİLER:")

if best_f1 > 0.2:
    print(f"  ✅ Model çalışıyor! En iyi threshold: {best_threshold:.3f}")
    print(f"  ✅ F1 Score: {best_f1:.3f}")

    if tp > 0:
        print(f"  ✅ {tp} arıza başarıyla tespit edildi!")
    if fn > 0:
        print(f"  ⚠️ {fn} arıza kaçırıldı (threshold düşürülebilir)")
    if fp > 10:
        print(f"  ⚠️ {fp} false alarm var (threshold yükseltilebilir)")

    print("\n  Sonraki Adım: threshold_optimizer.py scriptini çalıştırın:")
    print("    python notebooks/threshold_optimizer.py")
else:
    print(f"  ⚠️ Model performansı düşük (F1={best_f1:.3f})")
    print("  🔧 Öneriler:")
    print("     1. Class weight'i artırın (50-100 arası)")
    print("     2. SMOTE kullanın")
    print("     3. Daha fazla feature engineering yapın")
    print("     4. threshold_optimizer.py ile detaylı optimizasyon yapın")

print("\n" + "="*70)
