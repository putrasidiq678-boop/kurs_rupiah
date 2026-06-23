import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

def hitung_sentimen_berita(path_input="data/berita_mentah.csv"):
    print("Memulai analisis sentimen berita...")
    
    if not os.path.exists(path_input):
        print(f"File {path_input} tidak ditemukan! Membuat data simulasi sentimen...")
        # Jika belum ada file, kita buat dummy data sentimen netral (0.0) agar coding tidak error
        df_pasar = pd.read_csv("data/data_gabungan.csv")
        df_sentimen = pd.DataFrame({'Tanggal': df_pasar['Tanggal'], 'Sentimen_Berita': 0.0})
        return df_sentimen

    df_berita = pd.read_csv(path_input)
    # Pastikan ada kolom Tanggal dan Berita/Judul
    df_berita['Tanggal'] = pd.to_datetime(df_berita['Tanggal']).dt.strftime('%Y-%m-%d')
    
    analyzer = SentimentIntensityAnalyzer()
    
    skor_sentimen = []
    for teks in df_berita['Judul']: # sesuaikan dengan nama kolom teks di csv Anda
        # VADER menghasilkan compound score antara -1 (sangat negatif) hingga +1 (sangat positif)
        skor = analyzer.polarity_scores(str(teks))['compound']
        skor_sentimen.append(skor)
        
    df_berita['Skor_Sentimen'] = skor_sentimen
    
    # Agregasikan (cari rata-rata skor sentimen) per hari
    df_agregat = df_berita.groupby('Tanggal')['Skor_Sentimen'].mean().reset_index()
    df_agregat.rename(columns={'Skor_Sentimen': 'Sentimen_Berita'}, inplace=True)
    
    print("Analisis sentimen selesai!")
    return df_agregat

def gabungkan_semua_pilar():
    df_pasar_bi = pd.read_csv("data/data_gabungan.csv")
    df_sentimen = hitung_sentimen_berita()
    
    # Gabungkan pilar kuantitatif + pilar sentimen berita
    df_pasar_bi['Tanggal'] = df_pasar_bi['Tanggal'].astype(str)
    df_sentimen['Tanggal'] = df_sentimen['Tanggal'].astype(str)
    
    df_total = pd.merge(df_pasar_bi, df_sentimen, on='Tanggal', how='left')
    # Isi hari tanpa berita dengan nilai 0 (Netral)
    df_total['Sentimen_Berita'] = df_total['Sentimen_Berita'].fillna(0.0)
    
    df_total.to_csv("data/data_gabungan_total.csv", index=False)
    print("Sukses! 4 Pilar Data terintegrasi penuh di 'data/data_gabungan_total.csv'")

if __name__ == "__main__":
    gabungkan_semua_pilar()