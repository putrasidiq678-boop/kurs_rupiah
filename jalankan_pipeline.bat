@echo off
cd /d D:\Project\kurs_rupiah
echo === MEMULAI AUTOMATED PIPELINE JAM 5 SORE ===

echo 1. Mengambil Data Pasar & BI Rate Live...
call .venv\Scripts\activate
python model/ambil_data.py

echo 2. Memproses Analisis Sentimen Berita...
python model/proses_berita.py

echo 3. Melatih Ulang Model AI Prediksi...
python model/latih_model.py

echo === PIPELINE SELESAI DIEKSEKUSI ===
deactivate