import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# 1. Konfigurasi Halaman Utama
st.set_page_config(page_title="IoT & ANN Energy Management", page_icon="⚡", layout="wide")

st.title("⚡ Smart Microgrid Energy Management System")
st.markdown("### Optimalisasi Self-Consumption PLTS Atap & Prediksi Beban Listrik Berbasis Machine Learning")
st.markdown("---")

# 2. Model Machine Learning (ANN)
@st.cache_resource
def get_model():
    np.random.seed(42)
    X = np.random.rand(100, 3) * 50
    y = X[:, 0] * 1.5 + X[:, 1] * 0.8 + X[:, 2] * 1.2
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=200, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = get_model()

# Inisialisasi Session State untuk Riwayat
if 'riwayat' not in st.session_state:
    st.session_state['riwayat'] = []

# 3. Sidebar: Kontrol Sensor IoT & Parameter Grid
st.sidebar.header("🎛️ Parameter Sensor IoT")
suhu = st.sidebar.slider("Suhu Ruangan (°C)", 20.0, 35.0, 28.0)
kelembapan = st.sidebar.slider("Kelembapan (%)", 40.0, 80.0, 60.0)
daya_perangkat = st.sidebar.slider("Beban Terpasang (Watt)", 100.0, 500.0, 250.0)

st.sidebar.markdown("---")
st.sidebar.header("☀️ Pasokan Energi Terbarukan")
daya_plts = st.sidebar.slider("Generasi Daya PLTS Atap (Watt)", 0.0, 400.0, 180.0)
tarif_pln = st.sidebar.number_input("Tarif PLN (Rp/kWh)", value=1444.70, step=50.0)

# 4. Layout 2 Kolom Utama
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Telemetri & Kontrol")
    st.info(
        f"• **Suhu:** {suhu} °C\n\n"
        f"• **Kelembapan:** {kelembapan} %\n\n"
        f"• **Beban Terpasang:** {daya_perangkat} Watt\n\n"
        f"• **Estimasi Suplai PLTS:** {daya_plts} Watt"
    )
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        jalankan = st.button("🚀 Eksekusi Inference ANN", type="primary", use_container_width=True)
    with col_btn2:
        reset = st.button("🔄 Reset Riwayat", use_container_width=True)
        if reset:
            st.session_state['riwayat'] = []
            st.rerun()

with col2:
    st.subheader("📊 Hasil Prediksi & Manajemen Daya")
    
    if jalankan:
        # Preprocessing & Inference
        input_data = scaler.transform([[suhu, kelembapan, daya_perangkat]])
        hasil_ann = model.predict(input_data)[0]
        
        # Kalkulasi Self-Consumption & Grid Imbalance
        sisa_beban_pln = max(0, hasil_ann - daya_plts)
        surplus_plts = max(0, daya_plts - hasil_ann)
        self_consumption_ratio = min(100.0, (daya_plts / hasil_ann) * 100.0) if hasil_ann > 0 else 0
        
        # Analisis Biaya
        biaya_tanpa_plts = (hasil_ann / 1000) * 24 * tarif_pln
        biaya_dengan_plts = (sisa_beban_pln / 1000) * 24 * tarif_pln
        penghematan_rp = biaya_tanpa_plts - biaya_dengan_plts
        
        st.success("✅ Output Model Neural Network Berhasil Dihitung")
        
        # Grid Metrik
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediksi Beban (ANN)", f"{hasil_ann:.2f} W")
        m2.metric("Daya Impor (PLN)", f"{sisa_beban_pln:.2f} W")
        m3.metric("Hemat Harian", f"Rp {penghematan_rp:,.0f}")
        
        # Progress Bar Self-Consumption
        st.write(f"**Tingkat Self-Consumption PLTS:** {self_consumption_ratio:.1f}%")
        st.progress(int(min(100, self_consumption_ratio)))
        
        # Indikator Status Sistem
        if surplus_plts > 0:
            st.success(f"✨ **Surplus Daya Terdeteksi!** Tersisa {surplus_plts:.2f} W untuk pengisian baterai / diekspor ke grid.")
        elif sisa_beban_pln == 0:
            st.success("🌱 **100% Mandiri Energi!** Seluruh beban disuplai oleh PLTS Atap.")
        else:
            st.warning(f"⚠️ **Kekurangan Suplai!** Membutuhkan tambahan {sisa_beban_pln:.2f} W dari jaringan PLN.")
            
        # Simpan ke Riwayat Session State
        st.session_state['riwayat'].append({
            "Suhu (°C)": suhu,
            "Kelembapan (%)": kelembapan,
            "Beban (W)": daya_perangkat,
            "PLTS (W)": daya_plts,
            "Prediksi ANN (W)": round(hasil_ann, 2),
            "Daya PLN (W)": round(sisa_beban_pln, 2),
            "Self-Cons (%)": round(self_consumption_ratio, 1),
            "Hemat Harian (Rp)": round(penghematan_rp, 0)
        })
    else:
        st.info("👈 Klik tombol **'Eksekusi Inference ANN'** untuk memulai analisis.")

# 5. Bagian Bawah: Data Logging & Visualisasi Grafik
st.markdown("---")
st.subheader("📋 Logging Pengujian & Visualisasi Real-Time")

if len(st.session_state['riwayat']) > 0:
    df_riwayat = pd.DataFrame(st.session_state['riwayat'])
    
    col_tb, col_gf = st.columns([1.2, 1])
    
    with col_tb:
        st.write("**Tabel Riwayat Eksperimen:**")
        st.dataframe(df_riwayat, use_container_width=True)
        
        # Fitur Ekspor Data CSV
        csv_data = df_riwayat.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data Riwayat (CSV)",
            data=csv_data,
            file_name="riwayat_prediksi_energy_ann.csv",
            mime="text/csv"
        )
        
    with col_gf:
        st.write("**Grafik Perbandingan Beban vs Suplai PLN:**")
        st.line_chart(df_riwayat[["Prediksi ANN (W)", "Daya PLN (W)"]])
else:
    # Grafik dummy sebelum ada transaksi
    st.caption("Belum ada data pengujian tersimpan. Grafik simulasi akan muncul otomatis di bawah ini.")
    chart_dummy = pd.DataFrame({
        "Estimasi Beban ANN (W)": np.random.randn(10) * 15 + 140,
        "Impor PLN (W)": np.random.randn(10) * 10 + 60
    })
    st.line_chart(chart_dummy)