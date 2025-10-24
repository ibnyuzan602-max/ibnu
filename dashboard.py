import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import time
import io
import os
import json
from streamlit_lottie import st_lottie
import base64
import re

# =========================
# KONFIGURASI DASAR
# =========================
st.set_page_config(
    page_title="AI Vision Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CSS DARK FUTURISTIK
# =========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 10% 20%, #0b0b17, #1b1b2a 80%);
    color: white;
}
[data-testid="stSidebar"] {
    background: rgba(15, 15, 25, 0.95);
    backdrop-filter: blur(10px);
    border-right: 1px solid #333;
    padding-bottom: 80px; 
}
[data-testid="stSidebar"] * { color: white !important; }

h1, h2, h3 {
    text-align: center;
    font-family: 'Poppins', sans-serif;
}
.lottie-center {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 30px;
}
.result-card {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    text-align: center;
    box-shadow: 0 4px 25px rgba(0,0,0,0.25);
}
.detection-summary {
    background: rgba(40, 40, 60, 0.6);
    border-radius: 10px;
    padding: 15px;
    margin-top: 15px;
    text-align: left;
    border: 1px solid #555;
    color: #f0f0f0; 
}
.warning-box {
    background-color: rgba(255, 193, 7, 0.1);
    border-left: 5px solid #ffc107;
    color: #ffc107;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    width: 90%;
    margin: 15px auto;
}
</style>
""", unsafe_allow_html=True)

# =========================
# FUNGSI LOAD LOTTIE
# =========================
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

# =========================
# ANIMASI LOTTIE
# =========================
LOTTIE_WELCOME = "https://assets10.lottiefiles.com/packages/lf20_pwohahvd.json"
LOTTIE_DASHBOARD = "https://assets10.lottiefiles.com/packages/lf20_t24tpvcu.json"
LOTTIE_TRANSITION = "https://assets2.lottiefiles.com/packages/lf20_touohxv0.json"

# =========================
# SISTEM HALAMAN
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

# =========================
# SISTEM MUSIK (Final Fix: Memastikan Ganti Lagu DAN Loop Bekerja)
# =========================
music_folder = "music"

# Ambil semua file musik mp3 di folder /music
if os.path.exists(music_folder):
    music_files = [f for f in os.listdir(music_folder) if f.endswith(".mp3")]

    if len(music_files) == 0:
        st.sidebar.warning("⚠ Tidak ada file musik di folder 'music/'.")
    else:
        st.sidebar.markdown("#### 🎧 Player Musik")

        # --- INISIALISASI SESSION STATE YANG AMAN ---
        if "current_music" not in st.session_state:
            st.session_state.current_music = music_files[0]
        # ---------------------------------------------
        
        # Selectbox untuk memilih lagu
        selected_music = st.sidebar.selectbox(
            "Pilih Lagu:",
            options=music_files,
            index=music_files.index(st.session_state.current_music),
            key="music_selector"
        )
        
        # Perbarui state dan panggil rerun HANYA jika lagu benar-benar berubah
        if selected_music != st.session_state.current_music:
            st.session_state.current_music = selected_music
            # PENTING: Memaksa Streamlit me-render ulang seluruh halaman
            st.rerun() 

        music_path = os.path.join(music_folder, st.session_state.current_music)

        audio_b64 = ""
        try:
            # Mengubah data audio menjadi base64
            with open(music_path, "rb") as f:
                audio_data = f.read()
                audio_b64 = base64.b64encode(audio_data).decode()
        except FileNotFoundError:
            st.sidebar.error(f"File musik tidak ditemukan: {st.session_state.current_music}")
            
        # Menggunakan st.markdown dengan tag <audio> yang memiliki 'loop' dan 'autoplay'
        audio_html = f"""
        <p style="font-size: 14px; margin-top: 10px;">Sedang Memutar: <b>{st.session_state.current_music}</b></p>
        <audio controls loop autoplay style="width:100%">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            Browser Anda tidak mendukung audio.
        </audio>
        """
        st.sidebar.markdown(
            audio_html, 
            unsafe_allow_html=True
        )

else:
    st.sidebar.warning("⚠ Folder 'music/' tidak ditemukan.")
# =========================
# AKHIR SISTEM MUSIK
# =========================


# =========================
# HALAMAN 1: WELCOME
# =========================
if st.session_state.page == "home":
    st.markdown("<h1 style='text-align:center;'>🤖 Selamat Datang di AI Vision Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Sistem Cerdas untuk Deteksi Objek dan Klasifikasi Gambar</p>", unsafe_allow_html=True)
    
    lottie = load_lottie_url(LOTTIE_WELCOME)
    if lottie:
        st.markdown("<div class='lottie-center'>", unsafe_allow_html=True)
        st_lottie(lottie, height=300, key="welcome_anim")
        st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Masuk ke Website", use_container_width=True):
            st.session_state.page = "dashboard"
            with st.spinner("🔄 Memuat halaman..."):
                anim = load_lottie_url(LOTTIE_TRANSITION)
                if anim:
                    st_lottie(anim, height=200, key="transition_anim")
                time.sleep(1.5)
            st.rerun()

# =========================
# HALAMAN 2: DASHBOARD
# =========================
elif st.session_state.page == "dashboard":
    st.title("🤖 AI Vision Pro Dashboard")
    st.markdown("### Sistem Deteksi dan Klasifikasi Gambar Cerdas")

    lottie_ai = load_lottie_url(LOTTIE_DASHBOARD)
    if lottie_ai:
        st.markdown("<div class='lottie-center'>", unsafe_allow_html=True)
        st_lottie(lottie_ai, height=250, key="ai_anim")
        st.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.header("🧠 Mode AI")
    mode = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar", "AI Insight"])
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Unggah gambar, lalu biarkan AI menganalisis secara otomatis.")
    st.sidebar.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # DEFINISI NAMA KELAS KLASIFIKASI (Keras)
    CLASS_NAMES = ["Kucing 🐈", "Anjing 🐕", "Manusia 👤"]

    @st.cache_resource
    def load_models():
        try:
            yolo_model = YOLO(os.path.join("model", "Ibnu Hawari Yuzan_Laporan 4.pt"))
            classifier = tf.keras.models.load_model(os.path.join("model", "Ibnu Hawari Yuzan_Laporan 2.h5"))
            
            yolo_names = yolo_model.names 
            
            return yolo_model, classifier, yolo_names
        except Exception as e:
            st.warning(f"⚠ Gagal memuat model. Pastikan file model ada di folder 'model/'. Error: {e}")
            return None, None, {}

    # Tangkap model dan nama kelas YOLO
    yolo_model, classifier, YOLO_CLASS_NAMES = load_models()

    uploaded_file = st.file_uploader("📤 Unggah Gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file and yolo_model and classifier:
        img = Image.open(uploaded_file)
        
        # Perbaikan Warning Streamlit: use_container_width
        st.image(img, caption="🖼 Gambar yang Diupload", use_container_width=True)
        
        with st.spinner("🤖 AI sedang menganalisis gambar..."):
            time.sleep(1.5)

        if mode == "Deteksi Objek (YOLO)":
            st.info("🚀 Menjalankan deteksi objek...")
            img_cv2 = np.array(img)
            
            # Lakukan prediksi
            results = yolo_model.predict(source=img_cv2, verbose=False)
            
            # Visualisasikan hasil
            result_img = results[0].plot()
            # Perbaikan Warning Streamlit: use_container_width
            st.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)
            
            # MENGHITUNG DAN MENAMPILKAN RINGKASAN TEKSTUAL (DISEMPURNAKAN)
            detection_counts = {}
            
            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    
                    raw_class_name = YOLO_CLASS_NAMES.get(class_id, "Kelas Tidak Dikenal") 
                    
                    # Pembersihan nama kelas dari tanda baca/Markdown/angka
                    clean_name = raw_class_name.strip().replace('**', '')
                    final_class_name = re.sub(r'[^\w\s]+$', '', clean_name).strip()
                    
                    if final_class_name in detection_counts:
                        detection_counts[final_class_name] += 1
                    else:
                        detection_counts[final_class_name] = 1
                
                # Membuat ringkasan HTML/Markdown dengan format sederhana
                summary_list = []
                for name, count in detection_counts.items():
                    # Format final: Hanya nama objek (tanpa bold, tanpa angka count)
                    summary_list.append(f"- {name}") 
                
                summary_html = f"""
                <div class="detection-summary">
                    <h4>🔍 Ringkasan Objek Terdeteksi</h4>
                    <p>Objek yang terdeteksi adalah:</p>
                    <p>
                        {'<br>'.join(summary_list)}
                    </p>
                    <p>Total Objek Terdeteksi: <b>{len(results[0].boxes)}</b></p>
                </div>
                """
                st.markdown(summary_html, unsafe_allow_html=True)
            else:
                st.info("Tidak ada objek yang terdeteksi dalam gambar ini.")
            
            # Tombol Download
            img_bytes = io.BytesIO()
            Image.fromarray(result_img).save(img_bytes, format="PNG")
            img_bytes.seek(0)
            st.download_button("📥 Download Hasil Deteksi", data=img_bytes, file_name="hasil_deteksi_yolo.png", mime="image/png")

        elif mode == "Klasifikasi Gambar":
            st.info("🧠 Menjalankan klasifikasi gambar...")
            img_resized = img.resize((128, 128))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            prediction = classifier.predict(img_array, verbose=0)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # MENDAPATKAN NAMA KELAS DARI INDEKS
            try:
                predicted_class_name = CLASS_NAMES[class_index]
            except IndexError:
                predicted_class_name = f"Kelas Tidak Dikenal (Indeks: {class_index})"
                
            st.markdown(f"""
            <div class="result-card">
                <h3>🧾 Hasil Prediksi</h3>
                <p><b>Kelas:</b> {predicted_class_name}</p>
                <p><b>Akurasi:</b> {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

        elif mode == "AI Insight":
            st.info("🔍 Mode Insight Aktif")
            st.markdown("""
            <div class="result-card">
                <h3>💬 Insight Otomatis</h3>
                <p>AI menganalisis pola visual, bentuk, dan warna utama.</p>
                <p>Fitur ini masih dalam tahap pengembangan.</p>
            </div>
            """, unsafe_allow_html=True)

    elif uploaded_file and (yolo_model is None or classifier is None):
        st.markdown("<div class='warning-box'>⚠ Model AI gagal dimuat. Harap periksa path model.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='warning-box'>📂 Silakan unggah gambar terlebih dahulu.</div>", unsafe_allow_html=True)

    # 🔹 TOMBOL KEMBALI
    if st.sidebar.button("⬅ Kembali ke Halaman Awal", key="back_to_home_fixed", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
