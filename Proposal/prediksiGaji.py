"""
Prediksi Gaji Tahunan (USD) Profesional Data, AI, dan Teknologi
IT6043 - Machine Learning | Kelompok GodBless

Jalankan:  streamlit run app.py
"""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- konfigurasi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_MODELS = [
    os.path.join(BASE_DIR, "model.pkl"),
    os.path.join(BASE_DIR, "..", "model.pkl"),
    "model.pkl",
]
TARGET_DI_LOG = True       # True jika model dilatih pada log1p(salary_in_usd)

EXPERIENCE = {
    "Entry (pemula)": "EN",
    "Mid (menengah)": "MI",
    "Senior": "SE",
    "Executive": "EX",
}
EMPLOYMENT = {
    "Full-time": "FT",
    "Part-time": "PT",
    "Contract": "CT",
    "Freelance": "FL",
}
COMPANY_SIZE = {"Kecil (S)": "S", "Menengah (M)": "M", "Besar (L)": "L"}
REMOTE = {"On-site (0%)": 0, "Hybrid (50%)": 50, "Remote penuh (100%)": 100}

# Daftar singkat agar dropdown mudah dipakai. Ganti dengan nilai unik dari
# dataset (df["job_title"].unique(), dst.) jika ingin daftar lengkap.
JOB_TITLES = [
    "Data Scientist", "Data Engineer", "Data Analyst", "Software Engineer",
    "Machine Learning Engineer", "AI Engineer", "Research Scientist",
    "Analytics Engineer", "Business Intelligence Analyst", "Data Architect",
    "Engineer", "Manager", "Other",
]
COUNTRIES = {
    "Amerika Serikat": "US", "Inggris": "GB", "Kanada": "CA", "Jerman": "DE",
    "Prancis": "FR", "Spanyol": "ES", "Belanda": "NL", "India": "IN",
    "Australia": "AU", "Singapura": "SG", "Indonesia": "ID", "Jepang": "JP",
    "Brasil": "BR", "Lainnya": "Other",
}

# ----------------------------------------------------------------- tampilan
st.set_page_config(page_title="Prediksi Gaji Data & AI", page_icon="💼")

st.markdown(
    """
    <style>
    .block-container {max-width: 760px; padding-top: 2.5rem;}
    .hasil {
        border-left: 6px solid #1f6f5c; background: #eef6f3;
        padding: 1.2rem 1.5rem; border-radius: 4px; margin-top: 1rem;
    }
    .hasil .angka {font-size: 2.4rem; font-weight: 700; color: #14463a;}
    .hasil .ket {color: #3d5a52; margin: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------- model
@st.cache_resource
def load_model():
    for path in CANDIDATE_MODELS:
        if os.path.exists(path):
            return joblib.load(path)
    return None


def estimasi_demo(row: pd.Series) -> float:
    """Perkiraan kasar saat model.pkl belum tersedia (hanya untuk uji tampilan)."""
    dasar = {"EN": 85_000, "MI": 122_000, "SE": 156_000, "EX": 190_000}
    gaji = dasar[row["experience_level"]]
    gaji *= 1.0 if row["company_location"] == "US" else 0.65
    gaji *= 0.55 if row["remote_ratio"] == 50 else 1.0
    gaji *= 0.9 if row["company_size"] == "S" else 1.0
    gaji *= 1 + 0.03 * (row["work_year"] - 2023)
    return gaji


def prediksi(model, data: pd.DataFrame) -> float:
    if model is None:
        return estimasi_demo(data.iloc[0])
    hasil = float(model.predict(data)[0])
    return float(np.expm1(hasil)) if TARGET_DI_LOG else hasil


model = load_model()

# ------------------------------------------------------------------- header
st.title("Prediksi Gaji Profesional Data & AI")
st.write(
    "Isi profil pekerjaan di bawah ini untuk mendapat estimasi gaji tahunan "
    "dalam USD, berdasarkan model machine learning yang dilatih dari data gaji global."
)

if model is None:
    st.warning(
        "File `model.pkl` belum ditemukan, jadi aplikasi memakai estimasi demo. "
        "Jalankan `python train_model.py` untuk melatih dan mengekspor model."
    )

# ------------------------------------------------------------------- input
with st.form("form_prediksi"):
    st.subheader("Profil pekerjaan")
    c1, c2 = st.columns(2)

    with c1:
        job_title = st.selectbox("Judul pekerjaan", JOB_TITLES)
        exp_label = st.selectbox("Level pengalaman", list(EXPERIENCE), index=1)
        emp_label = st.selectbox("Jenis kontrak", list(EMPLOYMENT))
        work_year = st.slider("Tahun gaji", 2020, 2025, 2025)

    with c2:
        company_label = st.selectbox("Lokasi perusahaan", list(COUNTRIES))
        residence_label = st.selectbox("Negara tempat tinggal", list(COUNTRIES))
        size_label = st.selectbox("Ukuran perusahaan", list(COMPANY_SIZE), index=1)
        remote_label = st.selectbox("Pola kerja", list(REMOTE))

    kirim = st.form_submit_button("Hitung estimasi gaji", type="primary", use_container_width=True)

# ------------------------------------------------------------------- hasil
if kirim:
    company_location = COUNTRIES[company_label]
    employee_residence = COUNTRIES[residence_label]

    data = pd.DataFrame(
        [{
            "work_year": work_year,
            "experience_level": EXPERIENCE[exp_label],
            "employment_type": EMPLOYMENT[emp_label],
            "job_title": job_title,
            "employee_residence": employee_residence,
            "remote_ratio": REMOTE[remote_label],
            "company_location": company_location,
            "company_size": COMPANY_SIZE[size_label],
            # fitur turunan sesuai rencana pre-processing
            "is_us_company": int(company_location == "US"),
            "same_country": int(company_location == employee_residence),
        }]
    )

    try:
        tahunan = prediksi(model, data)
    except Exception as e:
        st.error(
            "Prediksi gagal. Pastikan kolom input sama dengan kolom yang dipakai "
            f"saat training.\n\nDetail: {e}"
        )
    else:
        st.markdown(
            f"""
            <div class="hasil">
                <p class="ket">Estimasi gaji per tahun</p>
                <div class="angka">${tahunan:,.0f}</div>
                <p class="ket">sekitar ${tahunan / 12:,.0f} per bulan</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Ini hanya estimasi dari model, bukan jaminan penawaran gaji sebenarnya. "
            "Hasil bisa berbeda untuk kombinasi yang jarang muncul di data."
        )
        with st.expander("Lihat data yang dikirim ke model"):
            st.dataframe(data, hide_index=True)

st.divider()
st.caption("IT6043 Machine Learning · Kelompok GodBless · UKDW")
