# 📈 Dashboard Prediksi & Analisis Kurs Rupiah (Metode 4 Pilar Data)

Aplikasi Dashboard Interaktif berbasis **Streamlit** dan **Machine Learning** yang dirancang untuk menganalisis serta memproyeksikan pergerakan nilai tukar Kurs Rupiah (USD/IDR) secara dinamis menggunakan pendekatan Integrasi 4 Pilar Data.

---

## 🚀 Fitur Utama
* **Visualisasi Multi-Indikator Interaktif:** Grafik tren real-time Kurs Rupiah disandingkan dengan BI Rate, IHSG, Harga Emas, dan Indeks Dolar (DXY).
* **Web Scraping BI Rate Otomatis:** Sistem secara otomatis melakukan scraping data kebijakan suku bunga acuan terbaru langsung dari internet saat pasar tutup.
* **Analisis Sentimen Berita Ekonomi:** Menggunakan NLP (*VADER Sentiment Analysis*) untuk mengubah judul berita mentah menjadi skor kuantitatif psikologi pasar.
* **Simulator AI (Machine Learning):** Dilengkapi model *Random Forest Regressor* dengan **Akurasi 96.77%** untuk mensimulasikan dampak perubahan kebijakan BI atau sentimen berita terhadap Kurs Rupiah.
* **Otomatisasi Pipeline Data:** Menggunakan mekanisme otomatisasi berkala (*Task Scheduling*) untuk memperbarui data setiap pukul 17:00 WIB setelah penutupan bursa.

---

## 🛠️ Struktur Proyek
* `ambil_data.py` - Mengambil data pasar live dari yfinance & scraper BI Rate live.
* `proses_berita.py` - Menghitung skor sentimen berita ekonomi harian dengan VADER.
* `latih_model.py` - Melatih model AI (*Random Forest*) untuk fungsi simulator prediksi.
* `app.py` - Kode antarmuka dashboard interaktif Streamlit.
* `.github/workflows/automation.yml` - Pipeline otomatisasi berkala cloud (GitHub Actions).

---

## 💻 Cara Menjalankan di Lokal

### 1. Kloning Repositori
```bash
git clone [https://github.com/putrasidiq678-boop](https://github.com/putrasidiq678-boop/kurs_rupiah.git)
cd NAMA_REPOSITORI
