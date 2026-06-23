import os
import pickle

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# ==============================================================================
# Konfigurasi & Konstanta
# ==============================================================================

PATH_DATA = "data/data_gabungan_total.csv"
PATH_MODEL = "model/prediksi_rupiah_rf.pkl"

WARNA_INDIKATOR = {
    "DXY_Index": "blue",
    "Harga_Emas": "gold",
    "IHSG_Index": "orange",
    "BI_Rate": "red",
    "Sentimen_Berita": "purple",
}

DAFTAR_INDIKATOR = list(WARNA_INDIKATOR.keys())


# ==============================================================================
# Fungsi-fungsi Pembantu (Helper)
# ==============================================================================

def muat_data(path: str) -> pd.DataFrame:
    """Memuat dan memproses data CSV gabungan 4 pilar."""
    df = pd.read_csv(path)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df.sort_values("Tanggal", inplace=True)
    return df


def label_sentimen(nilai: float) -> str:
    """Mengubah nilai sentimen numerik menjadi label teks."""
    if nilai > 0.05:
        return "Positif"
    elif nilai < -0.05:
        return "Negatif"
    return "Netral"


def muat_model(path: str):
    """Memuat model machine learning dari file pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ==============================================================================
# Komponen UI: Metrik Utama
# ==============================================================================

def tampilkan_metrik_utama(data_terakhir: pd.Series, data_sebelumnya: pd.Series):
    """Menampilkan 4 kartu metrik utama di bagian atas dashboard."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selisih_kurs = data_terakhir["Kurs_USDIDR"] - data_sebelumnya["Kurs_USDIDR"]
        st.metric(
            label="Kurs USD/IDR",
            value=f"Rp {data_terakhir['Kurs_USDIDR']:,.2f}",
            delta=f"{selisih_kurs:,.2f}",
        )

    with col2:
        selisih_dxy = data_terakhir["DXY_Index"] - data_sebelumnya["DXY_Index"]
        st.metric(
            label="US Dollar Index (DXY)",
            value=f"{data_terakhir['DXY_Index']:.2f}",
            delta=f"{selisih_dxy:.2f}",
        )

    with col3:
        st.metric(
            label="BI Rate Terkini",
            value=f"{data_terakhir['BI_Rate']:.2f} %",
        )

    with col4:
        sentimen = data_terakhir["Sentimen_Berita"]
        st.metric(
            label="Sentimen Berita Hari Ini",
            value=label_sentimen(sentimen),
            delta=f"{sentimen:.2f}",
        )


# ==============================================================================
# Halaman: Visualisasi Tren 4 Pilar
# ==============================================================================

def halaman_visualisasi(df: pd.DataFrame):
    """Halaman grafik tren multi-indikator interaktif."""
    st.subheader("📊 Grafik Tren Multi-Indikator Interaktif")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Sumbu kiri: Kurs Rupiah
    fig.add_trace(
        go.Scatter(
            x=df["Tanggal"],
            y=df["Kurs_USDIDR"],
            name="Kurs USD/IDR",
            line=dict(color="green", width=3),
        ),
        secondary_y=False,
    )

    # Sumbu kanan: Pilihan user
    opsi_pembanding = st.selectbox(
        "Pilih Indikator Pembanding (Sumbu Kanan):",
        DAFTAR_INDIKATOR,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Tanggal"],
            y=df[opsi_pembanding],
            name=opsi_pembanding,
            line=dict(
                color=WARNA_INDIKATOR[opsi_pembanding],
                width=2,
                dash="dash",
            ),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title_text=f"Analisis Pola Hubungan: Kurs Rupiah vs {opsi_pembanding}",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Kurs Rupiah (Rp)", secondary_y=False)
    fig.update_yaxes(title_text=opsi_pembanding, secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Tampilkan Tabel Data Mentah 4 Pilar"):
        st.dataframe(df.tail(50))


# ==============================================================================
# Halaman: Simulasi & Prediksi AI
# ==============================================================================

def halaman_simulasi(data_terakhir: pd.Series):
    """Halaman simulasi kebijakan menggunakan model ML."""
    st.subheader("🤖 Simulasi Kebijakan AI (Machine Learning)")
    st.info(
        "Simulasikan perubahan BI Rate atau Sentimen Berita untuk melihat "
        "estimasi pergerakan nilai tukar Rupiah."
    )

    # Input simulasi
    bi_input = st.slider(
        "Simulasi Jika BI Rate Berubah Ke (%):",
        4.0, 7.0,
        float(data_terakhir["BI_Rate"]),
        0.25,
    )
    sentimen_input = st.slider(
        "Simulasi Jika Sentimen Berita Berubah Ke:",
        -1.0, 1.0,
        float(data_terakhir["Sentimen_Berita"]),
        0.1,
    )

    if not os.path.exists(PATH_MODEL):
        st.warning("⚠️ File model belum tersedia. Silakan latih model terlebih dahulu.")
        return

    model_ai = muat_model(PATH_MODEL)

    if st.button("Hitung Prediksi Dampak AI"):
        input_simulasi = pd.DataFrame([{
            "DXY_Index": data_terakhir["DXY_Index"],
            "Harga_Emas": data_terakhir["Harga_Emas"],
            "IHSG_Index": data_terakhir["IHSG_Index"],
            "BI_Rate": bi_input,
            "Sentimen_Berita": sentimen_input,
        }])

        hasil_prediksi = model_ai.predict(input_simulasi)[0]
        selisih = hasil_prediksi - data_terakhir["Kurs_USDIDR"]

        st.success(f"### 🎯 Hasil Proyeksi AI: Rp {hasil_prediksi:,.2f}")
        st.metric(
            label="Estimasi Perubahan Nilai Kurs",
            value=f"Rp {hasil_prediksi:,.2f}",
            delta=f"{selisih:,.2f}",
        )


# ==============================================================================
# Halaman: Backtesting & Evaluasi Model
# ==============================================================================

def halaman_backtest():
    """Halaman evaluasi performa model AI terhadap data historis."""
    st.subheader("🧪 Backtesting Performa Model AI")
    st.write(
        "Halaman ini membandingkan hasil tebakan model AI di masa lalu "
        "dengan data riil yang sebenarnya terjadi di lapangan selama "
        "3 bulan terakhir."
    )

    from backtest import jalankan_backtest

    df_res, akurasi = jalankan_backtest(path_data=PATH_DATA)

    if df_res is None:
        st.error(f"Gagal memuat sistem pengujian: {akurasi}")
        return

    st.metric(
        label="Rata-rata Akurasi Backtest (90 Hari Terakhir)",
        value=f"{akurasi:.2f} %",
    )

    # 🎯 PERBAIKAN UTAMA: Mengubah 'Kurs_Rupiah' menjadi 'Kurs_USDIDR'
    df_chart = df_res.set_index("Tanggal")[["Kurs_USDIDR", "Prediksi_AI"]]
    df_chart.columns = ["Data Riil Lapangan (Aktual)", "Tebakan Model AI (Prediksi)"]

    st.write("### Grafik Perbandingan Historis")
    st.line_chart(df_chart)

    st.success(
        "💡 **Analisis Kesimpulan:** Jika kedua grafik di atas bergerak "
        "beriringan dan tidak saling menjauh, berarti model Random Forest "
        "Anda memiliki tingkat adaptasi real-time yang sangat sehat terhadap "
        "dinamika pasar finansial!"
    )


# ==============================================================================
# Fungsi Utama (Main)
# ==============================================================================

def main():
    """Entry point utama dashboard Streamlit."""
    st.set_page_config(page_title="Dashboard Ekonomi 4 Pilar", layout="wide")
    st.title("📈 Dashboard Interaktif Pergerakan Kurs Rupiah")
    st.markdown(
        "Analisis Kuantitatif Pasar, Kebijakan BI, Psikologi Investor, "
        "dan Sentimen Berita"
    )

    if not os.path.exists(PATH_DATA):
        st.error(f"File data `{PATH_DATA}` tidak ditemukan!")
        return

    df = muat_data(PATH_DATA)

    # Metrik utama
    data_terakhir = df.iloc[-1]
    data_sebelumnya = df.iloc[-2] if len(df) > 1 else data_terakhir
    tampilkan_metrik_utama(data_terakhir, data_sebelumnya)

    st.markdown("---")

    # Sidebar navigasi
    st.sidebar.header("🤖 Menu Model Prediksi")
    menu_pilihan = st.sidebar.radio(
        "Pilih Halaman:",
        ["Visualisasi Tren 4 Pilar", "Simulasi & Prediksi AI", "Backtesting & Evaluasi Model"],
    )

    # Routing halaman
    if menu_pilihan == "Visualisasi Tren 4 Pilar":
        halaman_visualisasi(df)
    elif menu_pilihan == "Simulasi & Prediksi AI":
        halaman_simulasi(data_terakhir)
    elif menu_pilihan == "Backtesting & Evaluasi Model":
        halaman_backtest()


if __name__ == "__main__":
    main()