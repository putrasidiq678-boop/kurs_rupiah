import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import pickle
import os

def latih_dan_simpan_model():
    path_data = "data/data_gabungan_total.csv"
    
    if not os.path.exists(path_data):
        print(f"Error: File {path_data} tidak ditemukan! Jalankan 'proses_berita.py' terlebih dahulu.")
        return

    # 1. Muat Dataset 4 Pilar
    df = pd.read_csv(path_data)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    # Pastikan data terurut rapi dari tanggal terlama ke terbaru
    df = df.sort_values('Tanggal').reset_index(drop=True)
    
    print("Memulai proses training AI dengan metode Time-Series Split...")

    # 2. Tentukan batasan waktu (Isolasi data 90 hari terakhir untuk backtesting)
    batas_waktu = df['Tanggal'].max() - timedelta(days=90)
    
    # Memisahkan baris data secara kronologis
    data_train = df[df['Tanggal'] < batas_waktu].copy()
    data_test = df[df['Tanggal'] >= batas_waktu].copy() # Data murni untuk simulasi backtest

    # 3. Tentukan Fitur (X) dan Target (Y)
    fitur = ['DXY_Index', 'Harga_Emas', 'IHSG_Index', 'BI_Rate', 'Sentimen_Berita']
    
    # Bersihkan data latihan dari nilai kosong
    data_train_clean = data_train.dropna(subset=fitur + ['Kurs_USDIDR'])
    data_test_clean = data_test.dropna(subset=fitur + ['Kurs_USDIDR'])
    
    X_train = data_train_clean[fitur]
    y_train = data_train_clean['Kurs_USDIDR']
    
    X_test = data_test_clean[fitur]
    y_test = data_test_clean['Kurs_USDIDR']

    # 4. Gunakan Algoritma Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Hitung akurasi evaluasi internal (R2 Score) terhadap data yang belum pernah dilihat
    if not X_test.empty:
        akurasi = model.score(X_test, y_test) * 100
        print(f"🔥 Model Berhasil Dilatih! Tingkat Validasi (R2 Score) pada data baru: {akurasi:.2f}%")
    else:
        print("🔥 Model Berhasil Dilatih! (Data uji kosong)")

    # 5. Simpan Model AI ke dalam file binary (.pkl)
    os.makedirs("model", exist_ok=True)
    with open("model/prediksi_rupiah_rf.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model AI sukses disimpan di folder 'model/prediksi_rupiah_rf.pkl'")

if __name__ == "__main__":
    latih_dan_simpan_model()