import streamlit as st
import joblib

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Gender Berdasarkan Cuitan",
    page_icon="⚥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS untuk styling sesuai screenshot
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #e8d5ff 0%, #c8a8ff 100%);
        font-family: 'Arial', sans-serif;
    }
    
    .gender-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        color: #00bcd4;
    }
    
    .title {
        color: #333;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .subtitle {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 1px solid #ddd;
        padding: 0.8rem 1rem;
        font-size: 1rem;
        text-align: center;
        width: 100%;
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background: #ff4444;
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: bold;
        width: 100%;
        margin: 1rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #e63939;
        transform: translateY(-1px);
    }
    
    .result-box {
        background: white;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1.5rem;
        text-align: center;
    }
    
    .result-text {
        color: #333;
        font-size: 1rem;
        font-weight: normal;
    }
</style>
""", unsafe_allow_html=True)

# Memuat model dan TF-IDF vectorizer
best_model = joblib.load('model/naive_bayes_model.joblib')
tfidf = joblib.load('model/tfidf_vectorizer.joblib')

# Inisialisasi session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.result = ""

# Container utama
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Icon gender
st.markdown('<div class="gender-icon">⚥</div>', unsafe_allow_html=True)

# Judul dan subtitle
st.markdown('<h1 class="title">Prediksi Gender Berdasarkan Cuitan</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Masukkan Teks, akan kami prediksi <br>gendernya</p>', unsafe_allow_html=True)

# Input nama
nama_input = st.text_input("", placeholder="Contoh: teks cuitan/tweet untuk diprediksi gendernya", label_visibility="collapsed")

# Tombol prediksi
if st.button("Prediksi"):
    if nama_input.strip():
        # Transformasi input menggunakan TF-IDF
        input_vector = tfidf.transform([nama_input])

        # Prediksi menggunakan model
        prediction = best_model.predict(input_vector)

        # Menyimpan hasil prediksi ke session state
        st.session_state.result = prediction[0]
        st.session_state.prediction_made = True
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")

# Tampilkan hasil jika sudah ada prediksi
if st.session_state.prediction_made:
    st.markdown(f'''
    <div class="result-box">
        <div class="result-text">Hasil Prediksi: {st.session_state.result}</div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Reset button (tersembunyi, untuk development)
if st.session_state.prediction_made:
    if st.button("Reset", key="reset_btn"):
        st.session_state.prediction_made = False
        st.session_state.result = ""
        st.rerun()
