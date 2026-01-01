"""
Model Utilities - API ve Dashboard için yardımcı fonksiyonlar
==============================================================
Optimize edilmiş modeli yükle ve tahmin yap
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import os


class OptimizedPredictor:
    """
    Optimize edilmiş tahmin sınıfı

    Kullanım:
        predictor = OptimizedPredictor()
        predictor.load_model('models/optimized_lgbm_model.pkl')
        result = predictor.predict(sensor_data)
    """

    def __init__(self):
        self.model = None
        self.features = None
        self.threshold = 0.05  # Varsayılan threshold
        self.threshold_config = None

    def load_model(self,
                   model_path: str = 'models/optimized_lgbm_model.pkl',
                   features_path: str = 'models/model_features.pkl',
                   threshold_path: str = 'models/threshold_config.pkl'):
        """
        Modeli, feature listesini ve threshold config'i yükle

        Args:
            model_path: Model dosya yolu
            features_path: Feature listesi dosya yolu
            threshold_path: Threshold config dosya yolu
        """
        try:
            # Modeli yükle
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model bulunamadı: {model_path}")
            self.model = joblib.load(model_path)
            print(f"✅ Model yüklendi: {model_path}")

            # Feature listesini yükle
            if os.path.exists(features_path):
                self.features = joblib.load(features_path)
                print(f"✅ Features yüklendi: {len(self.features)} adet")
            else:
                print(f"⚠️ Feature listesi bulunamadı: {features_path}")

            # Threshold config'i yükle
            if os.path.exists(threshold_path):
                self.threshold_config = joblib.load(threshold_path)
                self.threshold = self.threshold_config['threshold']
                print(f"✅ Optimal threshold: {self.threshold:.4f}")
                print(f"   Approach: {self.threshold_config['approach']}")
                print(f"   F1 Score: {self.threshold_config['metrics']['f1']:.3f}")
            else:
                print(f"⚠️ Threshold config bulunamadı, varsayılan kullanılıyor: {self.threshold}")

        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            raise

    def predict(self,
                X: pd.DataFrame,
                return_proba: bool = False,
                custom_threshold: Optional[float] = None) -> Dict:
        """
        Tahmin yap

        Args:
            X: Sensor verisi (DataFrame)
            return_proba: Olasılık değerlerini de döndür
            custom_threshold: Özel threshold kullan (None ise optimal threshold kullanılır)

        Returns:
            Dict: {
                'predictions': np.array,  # 0 veya 1
                'probabilities': np.array (opsiyonel),
                'threshold': float,
                'failure_count': int,
                'risk_level': str  # 'Low', 'Medium', 'High'
            }
        """
        if self.model is None:
            raise ValueError("Model yüklenmemiş! Önce load_model() çağrısı yapın.")

        # Feature kontrolü
        if self.features is not None:
            missing_features = set(self.features) - set(X.columns)
            if missing_features:
                print(f"⚠️ Eksik features: {missing_features}")
            X = X[self.features]

        # Tahmin yap
        y_proba = self.model.predict_proba(X)[:, 1]

        # Threshold uygula
        threshold = custom_threshold if custom_threshold is not None else self.threshold
        y_pred = (y_proba >= threshold).astype(int)

        # Failure count
        failure_count = y_pred.sum()

        # Risk level
        failure_ratio = failure_count / len(y_pred)
        if failure_ratio < 0.01:
            risk_level = 'Low'
        elif failure_ratio < 0.05:
            risk_level = 'Medium'
        else:
            risk_level = 'High'

        result = {
            'predictions': y_pred,
            'threshold': threshold,
            'failure_count': int(failure_count),
            'total_samples': len(y_pred),
            'failure_ratio': float(failure_ratio),
            'risk_level': risk_level
        }

        if return_proba:
            result['probabilities'] = y_proba
            result['max_probability'] = float(y_proba.max())
            result['mean_probability'] = float(y_proba.mean())

        return result

    def predict_single(self,
                      sensor_values: Dict[str, float],
                      return_proba: bool = True) -> Dict:
        """
        Tek bir sensor okuması için tahmin yap

        Args:
            sensor_values: {'sensor_1': 23.5, 'sensor_2': 45.1, ...}
            return_proba: Olasılık döndür

        Returns:
            Dict: {
                'is_failure': bool,
                'probability': float,
                'risk_level': str,
                'threshold': float
            }
        """
        # DataFrame'e çevir
        X = pd.DataFrame([sensor_values])

        # Tahmin yap
        result = self.predict(X, return_proba=True)

        return {
            'is_failure': bool(result['predictions'][0]),
            'probability': float(result['probabilities'][0]),
            'risk_level': 'High' if result['probabilities'][0] >= self.threshold else 'Low',
            'threshold': self.threshold,
            'confidence': float(abs(result['probabilities'][0] - self.threshold))
        }

    def evaluate_threshold(self,
                          X: pd.DataFrame,
                          y_true: np.array,
                          thresholds: Optional[list] = None) -> pd.DataFrame:
        """
        Farklı threshold değerlerinde performansı değerlendir

        Args:
            X: Test verisi
            y_true: Gerçek etiketler
            thresholds: Test edilecek threshold listesi

        Returns:
            DataFrame: Threshold başına metrikler
        """
        from sklearn.metrics import precision_score, recall_score, f1_score

        if thresholds is None:
            thresholds = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]

        y_proba = self.model.predict_proba(X)[:, 1]

        results = []
        for th in thresholds:
            y_pred = (y_proba >= th).astype(int)

            if y_pred.sum() > 0:
                results.append({
                    'threshold': th,
                    'predictions': y_pred.sum(),
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'f1': f1_score(y_true, y_pred, zero_division=0)
                })
            else:
                results.append({
                    'threshold': th,
                    'predictions': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1': 0
                })

        return pd.DataFrame(results)


def load_legacy_model(model_path: str = 'models/final_lgbm_model.pkl') -> Tuple:
    """
    Eski model formatını yükle (backwards compatibility)

    Returns:
        (model, features)
    """
    model = joblib.load(model_path)

    features_path = model_path.replace('model.pkl', 'features.pkl')
    if os.path.exists(features_path):
        features = joblib.load(features_path)
    else:
        features = None

    return model, features


def compare_models(X_test: pd.DataFrame,
                  y_test: np.array,
                  model_paths: Dict[str, str]) -> pd.DataFrame:
    """
    Farklı modelleri karşılaştır

    Args:
        X_test: Test verisi
        y_test: Test etiketleri
        model_paths: {'model_name': 'path/to/model.pkl'}

    Returns:
        DataFrame: Model karşılaştırma tablosu
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score
    )

    results = []

    for name, path in model_paths.items():
        try:
            predictor = OptimizedPredictor()
            predictor.load_model(path)

            result = predictor.predict(X_test, return_proba=True)

            results.append({
                'Model': name,
                'Threshold': result['threshold'],
                'Accuracy': accuracy_score(y_test, result['predictions']),
                'Precision': precision_score(y_test, result['predictions'], zero_division=0),
                'Recall': recall_score(y_test, result['predictions'], zero_division=0),
                'F1': f1_score(y_test, result['predictions'], zero_division=0),
                'ROC-AUC': roc_auc_score(y_test, result['probabilities']),
                'Failures Detected': result['failure_count']
            })
        except Exception as e:
            print(f"❌ {name} yüklenemedi: {e}")

    return pd.DataFrame(results)


# Kullanım örneği
if __name__ == "__main__":
    print("="*70)
    print("MODEL UTILS - KULLANIM ÖRNEĞİ")
    print("="*70)

    # Predictor oluştur
    predictor = OptimizedPredictor()

    # Modeli yükle
    try:
        predictor.load_model()

        # Örnek tahmin (eğer veri varsa)
        print("\n✅ Model yüklendi, tahmin için hazır!")
        print(f"\nÖrnek kullanım:")
        print(f"  result = predictor.predict(X_test, return_proba=True)")
        print(f"  print(result)")

    except FileNotFoundError:
        print("\n⚠️ Model dosyaları henüz oluşturulmamış.")
        print("   Önce threshold_optimizer.py scriptini çalıştırın.")
