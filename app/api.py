import pandas as pd
import joblib
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import numpy as np

# --- AYARLAR ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/final_lgbm_model.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "../models/model_features.pkl")

app = FastAPI(
    title="Wastewater Pump Failure Prediction API",
    description="Atık su pompaları için anomali tespiti ve arıza tahmini servisi.",
    version="1.0"
)

# --- MODEL YÜKLEME ---
model = None
model_features = None

@app.on_event("startup")
def load_model():
    global model, model_features
    try:
        # Modeli ve özellik listesini yükle
        model = joblib.load(MODEL_PATH)
        model_features = joblib.load(FEATURES_PATH)
        print("✅ Model ve özellik listesi başarıyla yüklendi.")
    except Exception as e:
        print(f"❌ Kritik Hata: Model yüklenemedi! {e}")

# --- VERİ ŞEMASI ---
class PredictionRequest(BaseModel):
    # Kullanıcıdan (veya Dashboard'dan) bir sözlük (JSON) bekliyoruz
    data: dict

# --- ENDPOINTLER ---

@app.get("/")
def home():
    return {"status": "active", "message": "Wastewater AI System Ready 🚀"}

@app.post("/predict")
def predict(request: PredictionRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model henüz yüklenmedi.")
    
    try:
        # 1. Gelen veriyi DataFrame'e çevir
        input_data = request.data
        df = pd.DataFrame([input_data])
        
        # 2. Modelin beklediği sütun sırasını garantiye al (Eksik varsa 0 doldur)
        # (Model eğitimindeki feature sırası ile tahmin sırası AYNI olmalı)
        df_reordered = df.reindex(columns=model_features, fill_value=0)
        
        # 3. Tahmin Yap
        # LightGBM 0 (Normal) veya 1 (Arıza) döner
        prediction = model.predict(df_reordered)[0]
        
        # Olasılık (Risk Skoru)
        probability = model.predict_proba(df_reordered)[0][1]
        
        # 4. Sonuç Dön
        result = {
            "prediction": int(prediction),
            "risk_score": float(probability),
            "status": "CRITICAL FAILURE" if prediction == 1 else "NORMAL",
            "confidence": f"{probability * 100:.2f}%"
        }
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tahmin hatası: {str(e)}")