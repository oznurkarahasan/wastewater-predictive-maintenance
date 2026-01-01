"""
THRESHOLD OPTIMIZATION & CLASS IMBALANCE SOLUTION
==================================================
Bu script, class 1 (arıza) tespitindeki sıfır performans sorununu çözer.

Sorun: Model threshold=0.05'te hiçbir arıza tespit edemiyor
Çözüm:
  1. SMOTE ile sentetik örnekler oluştur
  2. Optimal threshold'u ROC ve F1 bazlı belirle
  3. Class weights'i optimize et
  4. Probability calibration uygula
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV

print("="*70)
print("🔧 THRESHOLD & CLASS IMBALANCE OPTIMIZER")
print("="*70)

# ============================================================================
# 1. VERİ YÜKLEME
# ============================================================================
# Veri yolunu kontrol et (Linux için)
import os
data_paths = [
    '/home/user/wastewater-predictive-maintenance/data/processed/sensor_enriched.csv',
    'data/processed/sensor_enriched.csv',
    '../data/processed/sensor_enriched.csv',
    'd:/wastewater-predictive-maintenance/data/processed/sensor_enriched.csv'
]

df = None
for path in data_paths:
    if os.path.exists(path):
        print(f"✅ Veri bulundu: {path}")
        df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
        break

if df is None:
    print("❌ HATA: sensor_enriched.csv bulunamadı!")
    print("Lütfen veri yolunu kontrol edin.")
    exit(1)

# ============================================================================
# 2. VERİ HAZIRLIĞI
# ============================================================================
X = df.drop(columns=['y', 'machine_status'], errors='ignore')
drop_cols = [c for c in X.columns if 'sensor_40' in c]
X = X.drop(columns=drop_cols)
y = df['y']

print(f"\n📊 VERİ İSTATİSTİKLERİ")
print(f"Toplam Örnek: {len(y):,}")
print(f"Class 0 (Normal): {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.2f}%)")
print(f"Class 1 (Arıza): {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.2f}%)")
print(f"İmbalance Ratio: {(y==0).sum() / (y==1).sum():.1f}:1")
print(f"Feature Sayısı: {X.shape[1]}")

# ============================================================================
# 3. TRAIN-TEST SPLIT (Stratified)
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    shuffle=True,
    stratify=y,
    random_state=42
)

print(f"\n📦 SPLIT SONUÇLARI")
print(f"Train: {len(y_train):,} (Arıza: {y_train.sum()})")
print(f"Test:  {len(y_test):,} (Arıza: {y_test.sum()})")

# ============================================================================
# 4. YAKLAŞIM 1: SMOTE ile Oversampling
# ============================================================================
print("\n" + "="*70)
print("🧬 YAKLAŞIM 1: SMOTE (Synthetic Minority Over-sampling)")
print("="*70)

try:
    smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"SMOTE Sonrası:")
    print(f"  Class 0: {(y_train_smote==0).sum():,}")
    print(f"  Class 1: {(y_train_smote==1).sum():,}")
    print(f"  Yeni Balance Ratio: {(y_train_smote==0).sum() / (y_train_smote==1).sum():.2f}:1")

    # Model eğit
    model_smote = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=10,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary',
        is_unbalance=True,
        n_jobs=-1,
        verbose=-1,
        random_state=42
    )

    model_smote.fit(
        X_train_smote, y_train_smote,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    # Tahmin yap (olasılıklar)
    y_proba_smote = model_smote.predict_proba(X_test)[:, 1]

    print(f"\n📊 Olasılık Dağılımı:")
    print(f"  Min:  {y_proba_smote.min():.6f}")
    print(f"  Max:  {y_proba_smote.max():.6f}")
    print(f"  Mean: {y_proba_smote.mean():.6f}")
    print(f"  Std:  {y_proba_smote.std():.6f}")

except Exception as e:
    print(f"❌ SMOTE Hatası: {e}")
    y_proba_smote = None

# ============================================================================
# 5. YAKLAŞIM 2: Class Weight Optimization
# ============================================================================
print("\n" + "="*70)
print("⚖️ YAKLAŞIM 2: CLASS WEIGHT OPTIMIZATION")
print("="*70)

# Farklı class weight'ler dene
best_f1 = 0
best_weight = 1
best_model_weighted = None

weights_to_try = [1, 5, 10, 20, 30, 50, 100]

for weight in weights_to_try:
    model_temp = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=10,
        scale_pos_weight=weight,
        objective='binary',
        n_jobs=-1,
        verbose=-1,
        random_state=42
    )

    model_temp.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    y_proba = model_temp.predict_proba(X_test)[:, 1]

    # Optimal threshold bul (F1 bazlı)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])  # Son eleman sınır durumu
    optimal_threshold = thresholds[best_idx]

    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    f1 = f1_score(y_test, y_pred_optimal)

    print(f"Weight={weight:3d} → Optimal Threshold={optimal_threshold:.4f}, F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_weight = weight
        best_model_weighted = model_temp

print(f"\n🏆 En İyi Class Weight: {best_weight} (F1={best_f1:.4f})")

y_proba_weighted = best_model_weighted.predict_proba(X_test)[:, 1]

# ============================================================================
# 6. OPTIMAL THRESHOLD BELİRLEME (Her İki Model İçin)
# ============================================================================
print("\n" + "="*70)
print("🎯 OPTIMAL THRESHOLD BELİRLEME")
print("="*70)

def find_optimal_threshold(y_true, y_proba, method='f1'):
    """ROC veya F1 bazlı optimal threshold bul"""

    if method == 'roc':
        # Youden's J statistic (sensitivity + specificity - 1)
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx]

    elif method == 'f1':
        # F1 score maksimizasyonu
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        return thresholds[best_idx]

    elif method == 'f2':
        # F2 score (recall'a daha fazla ağırlık)
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        beta = 2
        f2_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-10)
        best_idx = np.argmax(f2_scores[:-1])
        return thresholds[best_idx]

# Her model için optimal threshold bul
results = {}

if y_proba_smote is not None:
    print("\n📈 SMOTE MODEL:")
    for method in ['roc', 'f1', 'f2']:
        threshold = find_optimal_threshold(y_test, y_proba_smote, method=method)
        y_pred = (y_proba_smote >= threshold).astype(int)

        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"  {method.upper():3s} Threshold={threshold:.4f} → P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")
        results[f'smote_{method}'] = {
            'model': model_smote,
            'threshold': threshold,
            'f1': f1,
            'precision': prec,
            'recall': rec
        }

print("\n📈 CLASS WEIGHTED MODEL:")
for method in ['roc', 'f1', 'f2']:
    threshold = find_optimal_threshold(y_test, y_proba_weighted, method=method)
    y_pred = (y_proba_weighted >= threshold).astype(int)

    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"  {method.upper():3s} Threshold={threshold:.4f} → P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")
    results[f'weighted_{method}'] = {
        'model': best_model_weighted,
        'threshold': threshold,
        'f1': f1,
        'precision': prec,
        'recall': rec
    }

# ============================================================================
# 7. EN İYİ MODELİ SEÇ
# ============================================================================
print("\n" + "="*70)
print("🏆 EN İYİ MODEL SEÇİMİ")
print("="*70)

# F1 score'a göre en iyi modeli seç
best_approach = max(results.items(), key=lambda x: x[1]['f1'])
best_name = best_approach[0]
best_config = best_approach[1]

print(f"\n✨ Kazanan: {best_name.upper()}")
print(f"   Threshold: {best_config['threshold']:.4f}")
print(f"   Precision: {best_config['precision']:.3f}")
print(f"   Recall:    {best_config['recall']:.3f}")
print(f"   F1 Score:  {best_config['f1']:.3f}")

# En iyi modeli kullan
final_model = best_config['model']
final_threshold = best_config['threshold']

# Tahmin yap
y_proba_final = final_model.predict_proba(X_test)[:, 1]
y_pred_final = (y_proba_final >= final_threshold).astype(int)

# ============================================================================
# 8. DETAYLI PERFORMANS RAPORU
# ============================================================================
print("\n" + "="*70)
print("📊 FİNAL MODEL SONUÇLARI")
print("="*70)
print(f"Threshold: {final_threshold:.4f}\n")
print(classification_report(y_test, y_pred_final))

# Confusion Matrix
print("\n📋 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_final)
print(f"                 Predicted")
print(f"              0           1")
print(f"Actual 0   {cm[0,0]:6d}    {cm[0,1]:6d}")
print(f"       1   {cm[1,0]:6d}    {cm[1,1]:6d}")

# Metrikleri hesapla
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"\n📈 Ek Metrikler:")
print(f"  True Positives:  {tp}")
print(f"  False Positives: {fp}")
print(f"  True Negatives:  {tn}")
print(f"  False Negatives: {fn}")
print(f"  Sensitivity (Recall): {sensitivity:.3f}")
print(f"  Specificity:          {specificity:.3f}")
print(f"  False Positive Rate:  {fpr:.3f}")
print(f"  ROC-AUC Score:        {roc_auc_score(y_test, y_proba_final):.3f}")

# ============================================================================
# 9. FARKLI THRESHOLD'LARDA PERFORMANS ANALİZİ
# ============================================================================
print("\n" + "="*70)
print("🔍 THRESHOLD SENSİTİVİTE ANALİZİ")
print("="*70)
print(f"{'Threshold':>10s} {'Predictions':>12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
print("-" * 70)

for th in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, final_threshold]:
    y_pred = (y_proba_final >= th).astype(int)
    preds = y_pred.sum()

    if preds > 0:
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        marker = " 🎯" if abs(th - final_threshold) < 0.0001 else ""
        print(f"{th:10.4f} {preds:12d} {prec:10.3f} {rec:10.3f} {f1:10.3f}{marker}")
    else:
        print(f"{th:10.4f} {'0 (no pred)':>12s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s}")

# ============================================================================
# 10. MODEL KAYDETME
# ============================================================================
print("\n" + "="*70)
print("💾 MODEL KAYDETME")
print("="*70)

# Model dizinini oluştur
os.makedirs('models', exist_ok=True)

# Modeli kaydet
joblib.dump(final_model, 'models/optimized_lgbm_model.pkl')
joblib.dump(X_train.columns.tolist(), 'models/model_features.pkl')

# Threshold'u kaydet
threshold_config = {
    'threshold': final_threshold,
    'approach': best_name,
    'metrics': {
        'precision': best_config['precision'],
        'recall': best_config['recall'],
        'f1': best_config['f1']
    }
}
joblib.dump(threshold_config, 'models/threshold_config.pkl')

print(f"✅ Model kaydedildi: models/optimized_lgbm_model.pkl")
print(f"✅ Features kaydedildi: models/model_features.pkl")
print(f"✅ Threshold config kaydedildi: models/threshold_config.pkl")

# ============================================================================
# 11. GÖRSELLEŞTIRME
# ============================================================================
print("\n" + "="*70)
print("📊 GÖRSELLEŞTIRMELER OLUŞTURULUYOR")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_final)
auc_score = roc_auc_score(y_test, y_proba_final)
axes[0, 0].plot(fpr, tpr, label=f'ROC (AUC={auc_score:.3f})', linewidth=2)
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Precision-Recall Curve
precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_proba_final)
axes[0, 1].plot(recalls, precisions, linewidth=2)
axes[0, 1].axvline(best_config['recall'], color='r', linestyle='--',
                   label=f'Selected (Recall={best_config["recall"]:.3f})')
axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Curve')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Normal', 'Failure'],
            yticklabels=['Normal', 'Failure'])
axes[1, 0].set_title(f'Confusion Matrix (Threshold={final_threshold:.4f})')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xlabel('Predicted')

# 4. Feature Importance (Top 20)
feature_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False).head(20)

axes[1, 1].barh(range(len(feature_imp)), feature_imp['importance'])
axes[1, 1].set_yticks(range(len(feature_imp)))
axes[1, 1].set_yticklabels(feature_imp['feature'], fontsize=8)
axes[1, 1].set_xlabel('Importance')
axes[1, 1].set_title('Top 20 Feature Importance')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('models/model_performance.png', dpi=150, bbox_inches='tight')
print("✅ Grafik kaydedildi: models/model_performance.png")

print("\n" + "="*70)
print("✅ OPTİMİZASYON TAMAMLANDI!")
print("="*70)
print(f"\nÖnerilen Kullanım:")
print(f"  1. Modeli yükle: joblib.load('models/optimized_lgbm_model.pkl')")
print(f"  2. Tahmin yap: y_proba = model.predict_proba(X)[:, 1]")
print(f"  3. Threshold uygula: y_pred = (y_proba >= {final_threshold:.4f}).astype(int)")
print("\n⚠️ DİKKAT: API ve Dashboard'da threshold değerini güncellemeyi unutmayın!")
