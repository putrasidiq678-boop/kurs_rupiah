import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def latih_dan_simpan_model():
    path_data = "data/data_gabungan_total.csv"
    
    if not os.path.exists(path_data):
        print(f"Error: File {path_data} tidak ditemukan! Jalankan 'proses_berita.py' terlebih dahulu.")
        return

    # 1. Muat Dataset 4 Pilar
    df = pd.read_csv(path_data)
    print("Memulai proses training AI dengan data 4 Pilar...")

    # 2. Tentukan Fitur (X) dan Target (Y)
    # Kita gunakan indikator makro, BI rate, dan sentimen berita untuk menebak Kurs Rupiah
    fitur = ['DXY_Index', 'Harga_Emas', 'IHSG_Index', 'BI_Rate', 'Sentimen_Berita']
    
    # Bersihkan baris yang kosong jika ada data yang corrupt
    df_clean = df.dropna(subset=fitur + ['Kurs_USDIDR'])
    
    X = df_clean[fitur]
    y = df_clean['Kurs_USDIDR']

    # 3. Split Data untuk Testing & Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Gunakan Algoritma Random Forest Regressor untuk Menangkap Hubungan Non-Linear pasar
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Hitung akurasi model
    akurasi = model.score(X_test, y_test) * 100
    print(f"🔥 Model Berhasil Dilatih! Tingkat Akurasi (R2 Score): {akurasi:.2f}%")

    # 5. Simpan Model AI ke dalam file binary (.pkl) agar bisa dipanggil oleh app.py
    os.makedirs("model", exist_ok=True)
    with open("model/prediksi_rupiah_rf.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model AI sukses disimpan di folder 'model/prediksi_rupiah_rf.pkl'")

if __name__ == "__main__":
    latih_dan_simpan_model()