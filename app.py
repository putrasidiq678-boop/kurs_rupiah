import streamlit as st
import pandas as pd
import plotly.graph_objects as dict_plot
from plotly.subplots import make_subplots
import os

# Set konfigurasi halaman dashboard
st.set_page_config(page_title="Dashboard Ekonomi 4 Pilar", layout="wide")

st.title("📈 Dashboard Interaktif Pergerakan Kurs Rupiah")
st.markdown("Analisis Kuantitatif Pasar, Kebijakan BI, Psikologi Investor, dan Sentimen Berita")

# Menggunakan data 4 pilar yang sudah digabungkan dengan sentimen
path_data = "data/data_gabungan_total.csv"

if os.path.exists(path_data):
    df = pd.read_csv(path_data)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df.sort_values('Tanggal', inplace=True)
    
    # --- METRIK UTAMA HARI INI ---
    data_terakhir = df.iloc[-1]
    data_sebelumnya = df.iloc[-2] if len(df) > 1 else data_terakhir
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selisih_kurs = data_terakhir['Kurs_USDIDR'] - data_sebelumnya['Kurs_USDIDR']
        st.metric(label="Kurs USD/IDR", value=f"Rp {data_terakhir['Kurs_USDIDR']:,.2f}", delta=f"{selisih_kurs:,.2f}")
    with col2:
        selisih_dxy = data_terakhir['DXY_Index'] - data_sebelumnya['DXY_Index']
        st.metric(label="US Dollar Index (DXY)", value=f"{data_terakhir['DXY_Index']:.2f}", delta=f"{selisih_dxy:.2f}")
    with col3:
        st.metric(label="BI Rate Terkini", value=f"{data_terakhir['BI_Rate']:.2f} %")
    with col4:
        # Menampilkan indikator rata-rata sentimen hari ini
        sentimen_hari_ini = data_terakhir['Sentimen_Berita']
        status_sentimen = "Positif" if sentimen_hari_ini > 0.05 else ("Negatif" if sentimen_hari_ini < -0.05 else "Netral")
        st.metric(label="Sentimen Berita Hari Ini", value=status_sentimen, delta=f"{sentimen_hari_ini:.2f}")

    st.markdown("---")

    # --- SIDEBAR UNTUK MENU MACHINE LEARNING ---
    st.sidebar.header("🤖 Menu Model Prediksi")
    menu_pilihan = st.sidebar.radio("Pilih Halaman:", ["Visualisasi Tren 4 Pilar", "Simulasi & Prediksi AI"])

    if menu_pilihan == "Visualisasi Tren 4 Pilar":
        st.subheader("📊 Grafik Tren Multi-Indikator Interaktif")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Sumbu Kiri: Kurs Rupiah
        fig.add_trace(
            dict_plot.Scatter(x=df['Tanggal'], y=df['Kurs_USDIDR'], name="Kurs USD/IDR", line=dict(color='green', width=3)),
            secondary_y=False,
        )
        
        # Sumbu Kanan: Pilihan user termasuk Sentimen Berita
        opsi_pembanding = st.selectbox(
            "Pilih Indikator Pembanding (Sumbu Kanan):",
            ["DXY_Index", "Harga_Emas", "IHSG_Index", "BI_Rate", "Sentimen_Berita"]
        )
        
        warna_pilihan = {'DXY_Index': 'blue', 'Harga_Emas': 'gold', 'IHSG_Index': 'orange', 'BI_Rate': 'red', 'Sentimen_Berita': 'purple'}
        
        fig.add_trace(
            dict_plot.Scatter(x=df['Tanggal'], y=df[opsi_pembanding], name=opsi_pembanding,
                              line=dict(color=warna_pilihan[opsi_pembanding], width=2, dash='dash')),
            secondary_y=True,
        )
        
        fig.update_layout(title_text=f"Analisis Pola Hubungan: Kurs Rupiah vs {opsi_pembanding}", hovermode="x unified")
        fig.update_yaxes(title_text="Kurs Rupiah (Rp)", secondary_y=False)
        fig.update_yaxes(title_text=opsi_pembanding, secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        if st.checkbox("Tampilkan Tabel Data Mentah 4 Pilar"):
            st.dataframe(df.tail(50))
            
    elif menu_pilihan == "Simulasi & Prediksi AI":
        st.subheader("🤖 Simulasi Kebijakan AI (Machine Learning)")
        st.info("Simulasikan perubahan BI Rate atau Sentimen Berita untuk melihat estimasi pergerakan nilai tukar Rupiah.")
        
        # Input simulasi untuk user
        bi_input = st.slider("Simulasi Jika BI Rate Berubah Ke (%):", 4.0, 7.0, float(data_terakhir['BI_Rate']), 0.25)
        sentimen_input = st.slider("Simulasi Jika Sentimen Berita Berubah Ke:", -1.0, 1.0, float(data_terakhir['Sentimen_Berita']), 0.1)
        
        path_model = "model/prediksi_rupiah_rf.pkl"
        if os.path.exists(path_model):
            import pickle
            with open(path_model, "rb") as f:
                model_ai = pickle.load(f)
                
            if st.button("Hitung Prediksi Dampak AI"):
                # Siapkan dataframe input data sesuai format training model
                input_simulasi = pd.DataFrame([{
                    'DXY_Index': data_terakhir['DXY_Index'],
                    'Harga_Emas': data_terakhir['Harga_Emas'],
                    'IHSG_Index': data_terakhir['IHSG_Index'],
                    'BI_Rate': bi_input,
                    'Sentimen_Berita': sentimen_input
                }])
                
                hasil_prediksi = model_ai.predict(input_simulasi)[0]
                selisih_dari_sekarang = hasil_prediksi - data_terakhir['Kurs_USDIDR']
                
                st.success(f"### 🎯 Hasil Proyeksi AI: Rp {hasil_prediksi:,.2f}")
                st.metric(label="Estimasi Perubahan Nilai Kurs", value=f"Rp {hasil_prediksi:,.2f}", delta=f"{selisih_dari_sekarang:,.2f}")
        else:
            st.error("File model AI belum ditemukan. Silakan jalankan 'py latih_model.py' terlebih dahulu di terminal Anda!")