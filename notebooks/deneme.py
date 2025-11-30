import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi yükle
df = pd.read_csv('d:/wastewater-predictive-maintenance/data/processed/sensor_enriched.csv', parse_dates=['timestamp'], index_col='timestamp')


X = df.drop(columns=['y', 'machine_status'], errors='ignore')
y = df['y']

print("📊 VERİ KONTROLÜ")
print("="*50)
print(f"Toplam satır: {len(df):,}")
print(f"Toplam feature: {X.shape[1]}")
print(f"\nSınıf dağılımı:")
print(y.value_counts())
print(f"Pozitif oran: {y.mean():.6f} ({y.mean()*100:.4f}%)")

# ✅ Feature'ların değer aralıklarını kontrol et
print("\n📈 FEATURE İSTATİSTİKLERİ (İlk 20):")
print(X.describe().T.head(20)[['mean', 'std', 'min', 'max']])

# ✅ Eksik değer kontrolü
missing = X.isnull().sum()
if missing.sum() > 0:
    print(f"\n⚠️ UYARI: {missing.sum():,} eksik değer var!")
    print(missing[missing > 0].head(10))
else:
    print("\n✅ Eksik değer yok")

# ✅ Sonsuz değer kontrolü
inf_counts = np.isinf(X).sum()
if inf_counts.sum() > 0:
    print(f"\n⚠️ UYARI: {inf_counts.sum():,} sonsuz değer var!")
    print(inf_counts[inf_counts > 0].head(10))
else:
    print("\n✅ Sonsuz değer yok")

# ✅ Sabit feature'ları bul
constant_features = X.columns[X.nunique() <= 1]
if len(constant_features) > 0:
    print(f"\n⚠️ {len(constant_features)} sabit feature var (silinmeli):")
    print(constant_features.tolist())
else:
    print("\n✅ Sabit feature yok")

# ✅ Arıza öncesi/sonrası feature'ları karşılaştır
failure_indices = y[y == 1].index
if len(failure_indices) > 0:
    print(f"\n🔍 ARIZA ANALİZİ ({len(failure_indices)} arıza var)")
    
    # Her arıza için önceki 100 ve sonraki 100 satırı al
    before_failure = []
    during_failure = []
    
    for idx in failure_indices[:5]:  # İlk 5 arıza
        try:
            loc = df.index.get_loc(idx)
            if loc > 100:
                before_failure.append(X.iloc[loc-100:loc].mean())
            during_failure.append(X.loc[idx])
        except:
            continue
    
    if before_failure and during_failure:
        before_df = pd.DataFrame(before_failure).mean()
        during_df = pd.DataFrame(during_failure).mean()
        
        # Farkı hesapla
        diff = (during_df - before_df).abs()
        diff = diff.sort_values(ascending=False).head(20)
        
        print("\n🔥 ARIZA SIRASINDA EN ÇOK DEĞIŞEN 20 FEATURE:")
        print(diff)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Train/Test split
last_failure_date = y[y==1].index.max()
split_date = last_failure_date - pd.Timedelta(days=5)

X_train = X.loc[X.index < split_date]
y_train = y.loc[y.index < split_date]
X_test = X.loc[X.index >= split_date]
y_test = y.loc[y.index >= split_date]

print("\n🌲 RANDOM FOREST ile TEST")
print("="*50)

# Basit Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print(f"Olasılık dağılımı:")
print(f"  Min: {y_proba_rf.min():.6f}")
print(f"  Max: {y_proba_rf.max():.6f}")
print(f"  Mean: {y_proba_rf.mean():.6f}")

# Farklı threshold'lar dene
for th in [0.01, 0.05, 0.1, 0.3, 0.5]:
    y_pred = (y_proba_rf >= th).astype(int)
    if y_pred.sum() > 0:
        from sklearn.metrics import precision_score, recall_score, f1_score
        print(f"\nThreshold {th}:")
        print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
        print(f"  Recall: {recall_score(y_test, y_pred):.3f}")
        print(f"  F1: {f1_score(y_test, y_pred, zero_division=0):.3f}")
        print(f"  Tahminler: {y_pred.sum()}")

# Feature importance
feature_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).head(20)

print("\n🔝 EN ÖNEMLİ 20 FEATURE:")
print(feature_imp.to_string(index=False))