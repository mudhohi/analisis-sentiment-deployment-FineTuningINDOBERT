import re
import os
import streamlit as st
import torch
# Menggunakan class BERT asli agar tidak memicu error tokenization/config
from transformers import BertTokenizer, BertForSequenceClassification
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ============================================================
# KONFIGURASI PATH ABSOLUT (Disesuaikan dengan Struktur Folder Baru)
# ============================================================
# 1. Ambil path folder utama tempat app.py berada
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Arahkan masuk ke dalam sub-folder tempat file model disimpan
MODEL_DIR = os.path.join(CURRENT_DIR, "indobert-mbg-sentiment-final")   

MAX_LENGTH = 128
LABEL_MAP = {0: "Negatif", 1: "Netral", 2: "Positif"}
LABEL_COLOR = {0: "#E85C5C", 1: "#8A8F98", 2: "#4CC38A"}
LABEL_EMOJI = {0: "\U0001F534", 1: "\U000026AA", 2: "\U0001F7E2"}

st.set_page_config(page_title="Sentimen MBG", page_icon="\U0001F37D", layout="centered")

# ============================================================
# CSS — TEMA GELAP, PREMIUM & CINEMATIC
# ============================================================
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at top, #1a1a22 0%, #0b0b0f 65%);
    color: #EDEDED;
}

.app-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #E8C468, #C9A227);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.app-subtitle {
    color: #9A9AA5;
    font-size: 0.95rem;
    margin-top: 0.2rem;
    margin-bottom: 1.6rem;
}

textarea {
    background-color: #16161C !important;
    color: #EDEDED !important;
    border: 1px solid #2A2A33 !important;
    border-radius: 10px !important;
}

.stButton>button {
    background: linear-gradient(90deg, #C9A227, #E8C468);
    color: #0B0B0F;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.4rem;
    transition: 0.2s ease;
}
.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(201, 162, 39, 0.35);
}

.result-card {
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    background: #14141A;
    border: 1px solid #26262F;
    margin-top: 1.2rem;
}

.result-card .result-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
}

.footer-note {
    color: #6E6E78;
    font-size: 0.8rem;
    text-align: center;
    margin-top: 3rem;
}
</style>
''', unsafe_allow_html=True)

# ============================================================
# PIPELINE PREPROCESSING
# ============================================================
SLANG_DICT = {
    'ga':'tidak', 'gak':'tidak', 'gk':'tidak', 'nggak':'tidak',
    'ngga':'tidak', 'enggak':'tidak', 'ndak':'tidak', 'tdk':'tidak', 'bkn':'bukan',
    'gue':'saya', 'gw':'saya', 'aku':'saya', 'sy':'saya',
    'lu':'kamu', 'lo':'kamu', 'km':'kamu', 'mrk':'mereka',
    'yg':'yang', 'dgn':'dengan', 'utk':'untuk',
    'krn':'karena', 'karna':'karena', 'krna':'karena',
    'tp':'tapi', 'tpi':'tapi', 'kl':'kalau', 'klo':'kalau', 'klu':'kalau',
    'dr':'dari', 'pd':'pada', 'jg':'juga', 'lg':'lagi', 'spt':'seperti',
    'spy':'supaya', 'biar':'supaya',
    'udah':'sudah', 'udh':'sudah', 'sdh':'sudah', 'dah':'sudah',
    'blm':'belum', 'bs':'bisa', 'bsa':'bisa', 'hrs':'harus',
    'dpt':'dapat', 'pgn':'ingin', 'pengen':'ingin', 'buat':'untuk', 'bwt':'untuk',
    'bgt':'sangat', 'bngt':'sangat', 'banget':'sangat',
    'emg':'memang', 'emang':'memang', 'aja':'saja', 'aj':'saja',
    'cuma':'hanya', 'cm':'hanya', 'gmn':'bagaimana', 'gimana':'bagaimana',
    'kayak':'seperti', 'kaya':'seperti',
    'mantap':'bagus', 'mantul':'bagus', 'keren':'bagus',
    'parah':'buruk', 'ok':'oke', 'sip':'oke',
    'thn':'tahun', 'bln':'bulan', 'hr':'hari', 'jm':'jam',
    'sblm':'sebelum', 'stlh':'setelah', 'skrg':'sekarang', 'skrang':'sekarang',
    'jt':'juta', 'rb':'ribu', 'rbu':'ribu',
    'tlng':'tolong', 'smg':'semoga', 'moga':'semoga', 'pdhl':'padahal', 'slh':'salah',
}

TOXIC_SLANG_MAP = {
    'tolol':'bodoh', 'goblok':'bodoh', 'bego':'bodoh', 'idiot':'bodoh',
    'dungu':'bodoh', 'oon':'bodoh', 'bloon':'bodoh', 'pekok':'bodoh',
    'ngaco':'salah', 'ngawur':'salah', 'asalasalan':'salah',
    'bobrok':'jelek', 'ampas':'jelek', 'sampah':'jelek',
    'kampangan':'jelek', 'tai':'jelek', 'taik':'jelek',
    'kontol':'buruk', 'ngontol':'buruk', 'ngontolin':'buruk',
    'bangsat':'buruk', 'bajingan':'buruk', 'anjing':'buruk',
    'kampret':'buruk', 'brengsek':'buruk', 'sialan':'buruk',
    'bejat':'buruk', 'biadab':'buruk', 'keparat':'buruk',
    'asu':'buruk', 'jancuk':'buruk', 'jancok':'buruk',
}
SLANG_DICT.update(TOXIC_SLANG_MAP)

POLITICAL_ENTITY_MAP = {
    'wowok':'prabowo',
}
SLANG_DICT.update(POLITICAL_ENTITY_MAP)

POST_STEM_REPLACE = {
    'grati': 'gratis', 'bergizi': 'gizi', 'efisiensi': 'efisien',
    'efektivitas': 'efektif', 'berkualitas': 'kualitas',
}

KEEP_WORDS = {
    'tidak', 'bukan', 'jangan', 'belum', 'tanpa', 'kurang', 'lebih', 'sangat', 'paling', 'anti',
    'bagus', 'baik', 'senang', 'sedih', 'marah', 'puas', 'setuju', 'dukung', 'mendukung', 'bangga',
    'buruk', 'jelek', 'bodoh', 'salah', 'kecewa', 'menolak', 'tolak', 'kritik',
    'curang', 'bohong', 'korupsi', 'gagal', 'rusak', 'hancur', 'sombong',
}
CUSTOM_STOP = {
    'rt', 'cc', 'via', 'yuk', 'ayo', 'nih', 'tuh', 'sih', 'deh', 'dong', 'loh', 'lho',
    'wkwk', 'haha', 'hehe', 'wih', 'wah', 'nah', 'kan', 'ya', 'yah',
    'thread', 'share', 'like', 'follow', 'subscribe', 'baca', 'selengkapnya', 'link', 'klik', 'bio',
}


@st.cache_resource(show_spinner=False)
def load_preprocessing_tools():
    factory_stem = StemmerFactory()
    stemmer = factory_stem.create_stemmer()
    factory_stop = StopWordRemoverFactory()
    stop_words = set(factory_stop.get_stop_words())
    stop_words -= KEEP_WORDS
    stop_words |= CUSTOM_STOP
    return stemmer, stop_words


@st.cache_resource(show_spinner="Memuat Model IndoBERT...")
def load_model():
    # Mengunci pembacaan langsung ke sub-folder model lokal
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR) 
    
    # Memaksa model membaca 3 kelas (Negatif, Netral, Positif) dan bypass perbedaan jumlah label config
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=3, ignore_mismatched_sizes=True)
    model.eval()
    return tokenizer, model


def clean_text(text, stemmer, stop_words):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"http\S+|www\.\S+|t\.co\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF\U0001FA00-\U0001FAFF]+",
        " ", text, flags=re.UNICODE,
    )
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(SLANG_DICT.get(tok, tok) for tok in text.split())
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(t for t in text.split() if t not in stop_words)
    text = re.sub(r"\s+", " ", text).strip()
    text = stemmer.stem(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(POST_STEM_REPLACE.get(tok, tok) for tok in text.split())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict(text, stemmer, stop_words, tokenizer, model):
    cleaned = clean_text(text, stemmer, stop_words)
    enc = tokenizer(cleaned, max_length=MAX_LENGTH, padding="max_length",
                     truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_label = int(torch.tensor(probs).argmax())
    return cleaned, pred_label, probs


# ============================================================
# UI (USER INTERFACE)
# ============================================================
st.markdown('<div class="app-title">Analisis Sentimen MBG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Pengecekan sentimen opini publik tentang Program '
    'Makan Bergizi Gratis &mdash; ditenagai IndoBERT hasil fine-tuning</div>',
    unsafe_allow_html=True,
)

examples = [
    "Alhamdulillah anak saya jadi lebih semangat sekolah karena ada MBG, menunya juga sehat.",
    "Puluhan siswa keracunan lagi habis makan MBG, kok bisa dibiarkan terus tanpa evaluasi.",
    "Program MBG hari ini menu ayam goreng dan sayur, dibagikan jam 9 pagi seperti biasa.",
]

# Menggunakan session state agar text tetap aman saat halaman dimuat ulang[cite: 1]
if "input_text_val" not in st.session_state:
    st.session_state.input_text_val = ""

st.write("Coba contoh cuitan:")
cols = st.columns(3)

# Sinkronisasi tombol contoh dengan session state[cite: 1]
for i, col in enumerate(cols):
    if col.button(f"Contoh {i+1}", use_container_width=True):
        st.session_state.input_text_val = examples[i]

# Menghubungkan text_area langsung ke session state[cite: 1]
text_input = st.text_area(
    "Masukkan teks / cuitan:", 
    key="input_text_val", 
    height=130,
    placeholder="Tulis atau tempel cuitan tentang MBG di sini..."
)

if st.button("Analisis Sentimen", use_container_width=True):
    if not text_input.strip():
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Menganalisis..."):
            stemmer, stop_words = load_preprocessing_tools()
            tokenizer, model = load_model()
            cleaned, pred_label, probs = predict(text_input, stemmer, stop_words, tokenizer, model)

        color = LABEL_COLOR[pred_label]
        emoji = LABEL_EMOJI[pred_label]
        st.markdown(f'''
        <div class="result-card">
            <div class="result-label" style="color:{color};">{emoji} {LABEL_MAP[pred_label]}</div>
        </div>
        ''', unsafe_allow_html=True)

        st.write("")
        st.write("**Distribusi probabilitas tiap kelas:**")
        for k in range(3):
            st.write(f"{LABEL_MAP[k]}")
            st.progress(float(probs[k]))
            st.caption(f"{probs[k]*100:.1f}%")

        with st.expander("Lihat teks setelah preprocessing"):
            st.code(cleaned if cleaned else "(kosong setelah dibersihkan)")

st.markdown(
    '<div class="footer-note">Model: IndoBERT (indobenchmark/indobert-base-p1, fine-tuned) '
    '&mdash; Tugas Analisis Sentimen Program MBG</div>',
    unsafe_allow_html=True,
)
