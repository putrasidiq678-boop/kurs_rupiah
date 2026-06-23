import pandas as pd
import numpy as np
from datetime import timedelta
import joblib
import os

def jalankan_backtest(path_data="data/data_gabungan_total.csv", path_model="model/prediksi_rupiah_rf.pkl"):
    # 1. Validasi file
    if not os.path.exists(path_data) or not os.path.exists(path_model):
        return None, f"File tidak ditemukan. Data: {os.path.exists(path_data)}, Model: {os.path.exists(path_model)}"
        
    # 2. Muat data pasar asli
    df = pd.read_csv(path_data)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df = df.sort_values('Tanggal').reset_index(drop=True)
    
    # 3. Ambil data 3 bulan terakhir
    batas_waktu = df['Tanggal'].max() - timedelta(days=90)
    data_backtest = df[df['Tanggal'] >= batas_waktu].copy()
    
    if data_backtest.empty:
        return None, "Data 3 bulan terakhir kosong."
    
    # 4. Definisi Fitur (Pastikan nama kolom ini ada di CSV Anda)
    fitur = ['DXY_Index', 'Harga_Emas', 'IHSG_Index', 'BI_Rate', 'Sentimen_Berita']
    
    X_test = data_backtest[fitur]
    y_aktual = data_backtest['Kurs_USDIDR']
    
    # 5. Muat Model AI
    model = joblib.load(path_model)
    
    # 6. Prediksi
    data_backtest['Prediksi_AI'] = model.predict(X_test)
    
    # 7. Hitung Akurasi (MAPE)
    mape = np.mean(np.abs((y_aktual - data_backtest['Prediksi_AI']) / y_aktual)) * 100
    skor_akurasi = 100 - mape
    
    # Ubah format tanggal ke string untuk sumbu grafik Streamlit
    data_backtest['Tanggal'] = data_backtest['Tanggal'].dt.strftime('%Y-%m-%d')
    
    # SINKRONISASI: Kembalikan kolom 'Kurs_USDIDR' agar app.py tidak bingung mencari 'Kurs_Rupiah'
    return data_backtest[['Tanggal', 'Kurs_USDIDR', 'Prediksi_AI']], skor_akurasi