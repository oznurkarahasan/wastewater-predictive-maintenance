import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import json
import os
import math  # Matematiksel işlemler (NaN/Inf kontrolü) için

# --- AYARLAR ---
API_URL = "http://127.0.0.1:8000/predict"
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/sensor_enriched.csv")

# Sayfa Yapısı
st.set_page_config(
    page_title="Pump Guardian AI",
    page_icon="🌊",
    layout="wide"
)

# --- BAŞLIK ---
st.title("🌊 Wastewater Treatment - Predictive Maintenance Dashboard")
st.markdown("Atık su pompaları için yapay zeka destekli erken uyarı sistemi.")

# --- YAN MENÜ (SİMÜLASYON) ---
st.sidebar.header("🔧 Simülasyon Paneli")
st.sidebar.info("Modeli test etmek için geçmiş verilerden bir an seçin.")

# Veriyi Önbellekli Yükle (Hız için)
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        # Veriyi oku
        df = pd.read_csv(DATA_PATH)
        # Timestamp indeksini düzelt
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        return df
    else:
        return None

df = load_data()

if df is not None:
    # Kullanıcıya tarih seçtir (Slider veya Selectbox)
    # Rastgele 50 örnek alalım ki liste şişmesin
    sample_indices = df.sample(50).index.sort_values()
    selected_date = st.sidebar.selectbox("Tarih Seçin (Simülasyon):", sample_indices)
    
    # Seçilen satırı al
    selected_row = df.loc[selected_date]
    
    # Gerçek Durumu (Label) göster (Eğer y sütunu varsa)
    real_status = "Bilinmiyor"
    if 'y' in df.columns:
        real_label = selected_row['y']
        real_status = "🔴 ARIZA (Gerçek)" if real_label == 1 else "🟢 NORMAL (Gerçek)"
        
    st.sidebar.markdown(f"**Seçilen Anın Gerçek Durumu:**")
    st.sidebar.markdown(f"### {real_status}")
    
    # --- ANA EKRAN ---
    
    # 1. Sensör Verileri (Özet)
    st.subheader(f"📊 Sensör Durumu - {selected_date}")
    
    col1, col2, col3, col4 = st.columns(4)
    # Önemli sensörleri göster (Örnek: sensor_00, sensor_04...)
    sensor_keys = ['sensor_00', 'sensor_04', 'sensor_10', 'sensor_50']
    
    metrics_cols = [col1, col2, col3, col4]
    
    for i, sensor in enumerate(sensor_keys):
        if sensor in selected_row:
            val = selected_row[sensor]
            metrics_cols[i].metric(label=sensor, value=f"{val:.2f}")
            
    # 2. API'ye Gönder ve Tahmin Al
    st.divider()
    
    if st.button("🔍 Yapay Zeka Analizi Başlat", type="primary"):
        with st.spinner('AI Modeli Verileri İnceliyor...'):
            
            # --- VERİ TEMİZLİĞİ (JSON UYUMLULUĞU İÇİN) ---
            # 1. Hedef sütunu çıkar
            row_data = selected_row.drop(['y'], errors='ignore')
            
            # 2. Pandas Series'i sözlüğe çevir
            raw_payload = row_data.to_dict()

            # 3. NaN ve Infinite değerlerini temizle (JSON hatası almamak için)
            clean_payload = {}
            for key, value in raw_payload.items():
                # Sayısal değerleri kontrol et
                if isinstance(value, (float, int)):
                    if pd.isna(value) or math.isinf(value):
                        clean_payload[key] = 0.0  # Hatalı değerleri 0 yap
                    else:
                        clean_payload[key] = value
                else:
                    clean_payload[key] = value
            # ----------------------------------------------
            
            try:
                # DİKKAT: Burada 'clean_payload' değişkenini kullanıyoruz!
                response = requests.post(API_URL, json={"data": clean_payload})
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Sonuç Gösterimi
                    risk_score = result["risk_score"]
                    status = result["status"]
                    
                    c1, c2 = st.columns([1, 2])
                    
                    with c1:
                        # Gauge Chart (İbre)
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = risk_score * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Arıza Riski (%)"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkred" if risk_score > 0.5 else "green"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgreen"},
                                    {'range': [50, 80], 'color': "orange"},
                                    {'range': [80, 100], 'color': "red"}],
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with c2:
                        st.markdown("### AI Kararı:")
                        if result['prediction'] == 1:
                            st.error(f"⚠️ {status} - DİKKAT! Sistem arıza riski tespit etti.")
                            st.markdown(f"Bu pompanın önümüzdeki 24 saat içinde bozulma ihtimali çok yüksek.")
                        else:
                            st.success(f"✅ {status} - Sistem stabil görünüyor.")
                            
                else:
                    st.error(f"API Hatası: {response.status_code} - {response.text}")
                    
            except Exception as e:
                st.error(f"Bağlantı Hatası: API çalışıyor mu? ({e})")

else:
    st.warning("Veri dosyası bulunamadı! Lütfen 'data/processed/sensor_enriched.csv' dosyasını kontrol edin.")