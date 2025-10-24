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
# CSS DARK FUTURISTIK (SEMUA PERBAIKAN)
# =========================
st.markdown("""
<style>
/* Keyframes untuk pergerakan latar belakang (TIDAK DIHAPUS) */
@keyframes move-background {
    from {
        background-position: 0 0;
    }
    to {
        background-position: 10000px 10000px; 
    }
}

/* Container utama Streamlit sebagai latar belakang (STARFIELD PERMANEN) */
[data-testid="stAppViewContainer"] {
    background: 
        /* Starfield Layers */
        radial-gradient(2px 2px at 20px 30px, #eee, rgba(0,0,0,0)),
        radial-gradient(2px 2px at 40px 70px, #fff, rgba(0,0,0,0)),
        radial-gradient(2px 2px at 150px 20px, #ddd, rgba(0,0,0,0)),
        radial-gradient(2px 2px at 200px 150px, #eee, rgba(0,0,0,0)),
        radial-gradient(1px 1px at 80px 100px, #ccc, rgba(0,0,0,0)),
        radial-gradient(3px 3px at 300px 120px, #fff, rgba(0,0,0,0)),
        radial-gradient(3px 3px at 450px 80px, #eee, rgba(0,0,0,0)),
        radial-gradient(4px 4px at 500px 50px, #fff, rgba(0,0,0,0)),
        /* Latar belakang dasar gelap */
        radial-gradient(circle at 10% 20%, #0b0b17, #1b1b2a 80%);

    background-size: 500px 500px, 500px 500px, 500px 500px, 500px 500px, 500px 500px, 
                     500px 500px, 500px 500px, 
                     500px 500px,
                     100% 100%; 

    background-repeat: repeat;
    animation: move-background 100s linear infinite; 
    color: white;
}

/* Memaksa warna teks menjadi putih terang untuk SEMUA konten KECUALI yang spesifik */
[data-testid="stAppViewContainer"] p, 
[data-testid="stAppViewContainer"] div:not([data-testid="stFileUploader"] *) , 
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span {
    color: white !important;
}

/* FIX HEADING: Memastikan semua judul putih */
h1, h2, h3 {
    text-align: center;
    font-family: 'Poppins', sans-serif;
    color: white !important; 
}

/* FIX KONTEN PUTIH: Menghilangkan latar belakang putih di semua blok konten Streamlit */
[data-testid="stBlock"] { 
    background-color: transparent !important; 
    color: white !important;
}

[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background-color: transparent !important;
}
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > div:first-child,
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > div:nth-child(2) > div:first-child,
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > div:nth-child(3) {
    background-color: transparent !important;
}

/* 🔥🔥🔥 PERBAIKAN KRITIS 1: Teks pada st.file_uploader (Dropzone) harus HITAM PEKAT */

/* Pastikan area dropzone itu sendiri Putih Murni */
[data-testid="stFileUploaderDropzone"] {
    background-color: #FFFFFF !important; /* Putih Murni */
    border-color: #BBBBBB !important; 
}

/* Menargetkan teks "Drag and drop file here" */
[data-testid="stFileUploaderDropzone"] p { 
    color: #000000 !important; /* HARUS HITAM PEKAT */
}
/* Menargetkan teks "Limit 200MB..." */
[data-testid="stFileUploaderDropzone"] small { 
    color: #000000 !important; /* HARUS HITAM PEKAT */
}
/* Menargetkan ikon upload menjadi HITAM PEKAT */
[data-testid="stFileUploaderDropzone"] svg {
    fill: #000000 !important; /* HARUS HITAM PEKAT */
}

/* 🔥🔥🔥 PERBAIKAN KRITIS 2: Teks & Ikon file yang SUDAH DIUNGGAH harus PUTIH */

/* Memastikan teks yang melaporkan file yang sudah diunggah menjadi PUTIH */
[data-testid="stFileUploader"] > div > div > div > div:nth-child(2) > div > div > div > p {
    color: white !important; /* Nama file: aadidas_ (24).jpg */
}

/* Memastikan ukuran file (KB/MB) yang diunggah menjadi PUTIH */
[data-testid="stFileUploader"] > div > div > div > div:nth-child(2) > div > div > div > small {
    color: white !important; /* Ukuran file: 44.1KB */
}
/* Memastikan ikon file yang sudah diunggah (ikon dokumen) menjadi PUTIH MURNI */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFilename"] svg {
    fill: #FFFFFF !important; /* IKON PUTIH MURNI */
}

/* --- KODE CSS LAINNYA --- */
[data-testid="stSidebar"] {
    background: rgba(15, 15, 25, 0.95);
    backdrop-filter: blur(10px);
    border-right: 1px solid #333;
    padding-bottom: 80px; 
}
[data-testid="stSidebar"] * { color: white !important; }

/* FIX: Target label st.file_uploader (Label "Unggah Gambar" di luar kotak) */
[data-testid="stFileUploader"] label p {
    color: #f0f0f0 !important; /* tetap putih */
    font-size: 1.1em; 
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

/* FIX: Tombol "Browse Files" di dalam st.file_uploader */
[data-testid="stFileUploader"] button {
    background-color: #334466; 
    color: white !important;
    border: 1px solid #556688;
    padding: 8px 12px;
    border-radius: 8px;
    box-shadow: none;
    transition: all 0.2s;
}

[data-testid="stFileUploader"] button:hover {
    background-color: #445577; 
}

/* FIX: Tombol di Halaman Awal & Kembali ke Halaman Awal */
.stButton>button:first-child { 
    background-color: #0077b6; 
    color: white !important;
    border: 1px solid #00b4d8;
    font-size: 1.2em;
    font-weight: bold;
    height: 3.5em; 
    box-shadow: 0 0 15px rgba(0, 119, 182, 0.5); 
}

.stButton>button:first-child:hover {
    background-color: #0096c7;
    box-shadow: 0 0 20px rgba(0, 183, 224, 0.8);
}


/* Perubahan Seleksi Musik: Latar Belakang Gelap dan Border */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child {
    background-color: rgba(40, 40, 60, 0.8) !important; 
    color: white !important;
    border-color: #556688 !important; 
    border-radius: 8px !important; 
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] label p {
    color: white !important;
    font-weight: normal !important;
    display: block !important; 
}

[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: white !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: white !important;
}

/* FIX: Menu Dropdown (List Pilihan) - Container */
div[data-baseweb="popover"] {
    background-color: rgba(15, 15, 25, 1) !important; 
    border: 1px solid #556688 !important;
    border-radius: 8px !important;
}

/* FIX: Menu Dropdown (List Pilihan) - Item Default */
div[role="option"] {
    background-color: transparent !important;
    color: white !important; 
}

/* FIX: Menu Dropdown (List Pilihan) - Item Hover/Pilih Biru Neon */
div[role="option"]:hover {
    background-color: #0077b6 !important; 
    color: white !important; 
    box-shadow: 0 0 10px rgba(0, 119, 182, 0.5) !important; 
    border-radius: 8px !important;
}
div[role="option"][aria-selected="true"] {
    background-color: #0077b6 !important; 
    color: white !important;
    box-shadow: 0 0 10px rgba(0, 119, 182, 0.5) !important; 
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# FUNGSI DAN LOGIKA UTAMA (TIDAK BERUBAH)
# =========================
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

LOTTIE_WELCOME = "https://assets10.lottiefiles.com/packages/lf20_pwohahvd.json"
LOTTIE_DASHBOARD = "https://assets10.lottiefiles.com/packages/lf20_t24tpvcu.json"
LOTTIE_TRANSITION = "https://assets2.lottiefiles.com/packages/lf20_touohxv0.json"

if "page" not in st.session_state:
    st.session_state.page = "home"

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

elif st.session_state.page == "dashboard":
    st.title("🤖 AI Vision Pro Dashboard")
    st.markdown("### Sistem Deteksi dan Klasifikasi Gambar Cerdas")

    # =========================
    # SISTEM MUSIK (DI DALAM SIDEBAR)
    # =========================
    music_folder = "music"

    if os.path.exists(music_folder):
        music_files = [f for f in os.listdir(music_folder) if f.endswith(".mp3")]

        if len(music_files) == 0:
            st.sidebar.warning("⚠ Tidak ada file musik di folder 'music/'.")
        else:
            st.sidebar.markdown("#### 🎧 Player Musik")

            if "current_music" not in st.session_state:
                st.session_state.current_music = music_files[0] if music_files else None
            
            current_index = music_files.index(st.session_state.current_music) if st.session_state.current_music in music_files else 0
            selected_music = st.sidebar.selectbox(
                "Pilih Lagu:", 
                options=music_files,
                index=current_index,
                key="music_selector"
            )
            
            if selected_music != st.session_state.current_music:
                st.session_state.current_music = selected_music
                st.rerun() 

            music_path = os.path.join(music_folder, st.session_state.current_music)

            audio_bytes = None
            try:
                with open(music_path, "rb") as f:
                    audio_bytes = f.read()
            except FileNotFoundError:
                st.sidebar.error(f"File musik tidak ditemukan: {st.session_state.current_music}")
            
            if audio_bytes:
                st.sidebar.audio(
                    audio_bytes,
                    format='audio/mp3',
                )
            
    else:
        st.sidebar.warning("⚠ Folder 'music/' tidak ditemukan.")
    # =========================

    lottie_ai = load_lottie_url(LOTTIE_DASHBOARD)
    if lottie_ai:
        st.markdown("<div class='lottie-center'>", unsafe_allow_html=True)
        st_lottie(lottie_ai, height=250, key="ai_anim")
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # KONTROL MODE AI (DI DALAM SIDEBAR)
    # =========================
    st.sidebar.header("🧠 Mode AI")
    mode = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Unggah gambar, lalu biarkan AI menganalisis secara otomatis.")
    st.sidebar.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
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

    yolo_model, classifier, YOLO_CLASS_NAMES = load_models()

    uploaded_file = st.file_uploader("📤 Unggah Gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file and yolo_model and classifier:
        img = Image.open(uploaded_file)
        
        try:
            if img.mode != 'RGB':
                img = img.convert('RGB')
        except Exception as e:
            st.error(f"Gagal mengonversi gambar ke RGB: {e}. Harap coba dengan file gambar lain.")
            st.stop() 
        
        st.image(img, caption="🖼 Gambar yang Diupload", use_container_width=True)
        
        with st.spinner("🤖 AI sedang menganalisis gambar..."):
            time.sleep(1.5)

        if mode == "Deteksi Objek (YOLO)":
            st.info("🚀 Menjalankan deteksi objek...")
            img_cv2 = np.array(img)
            
            results = yolo_model.predict(source=img_cv2, verbose=False)
            
            result_img = results[0].plot()
            st.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)
            
            detection_counts = {}
            
            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    raw_class_name = YOLO_CLASS_NAMES.get(class_id, "Kelas Tidak Dikenal") 
                    
                    clean_name = raw_class_name.strip().replace('**', '')
                    final_class_name = re.sub(r'[^\w\s]+$', '', clean_name).strip()
                    
                    if final_class_name in detection_counts:
                        detection_counts[final_class_name] += 1
                    else:
                        detection_counts[final_class_name] = 1
                
                summary_list = []
                # Perbaikan: Hanya tampilkan nama objek saja, tanpa hitungan atau format Markdown yang berlebihan
                for name, count in detection_counts.items():
                    summary_list.append(f"- {name}") 
                
                summary_html = f"""
                <div class="detection-summary">
                    <h4>🔍 Ringkasan Objek Terdeteksi</h4>
                    <p>Jenis objek yang terdeteksi:</p>
                    <p style="color: #00b4d8; font-weight: bold;">
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
            
            try:
                predicted_class_name = CLASS_NAMES[class_index]
            except IndexError:
                predicted_class_name = f"Kelas Tidak Dikenal (Indeks: {class_index})"
                
            st.markdown(f"""
            <div class="result-card">
                <h3>🧾 Hasil Prediksi</h3>
                <p><b>Kelas:</b> <span style="color: #00b4d8;">{predicted_class_name}</span></p>
                <p><b>Akurasi:</b> <span style="color: #00b4d8;">{confidence:.2%}</span></p>
            </div>
            """, unsafe_allow_html=True)
        

    elif uploaded_file and (yolo_model is None or classifier is None):
        st.markdown("<div class='warning-box'>⚠ Model AI gagal dimuat. Harap periksa path model.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='warning-box'>📂 Silakan unggah gambar terlebih dahulu.</div>", unsafe_allow_html=True)

    if st.sidebar.button("⬅ Kembali ke Halaman Awal", key="back_to_home_fixed", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
