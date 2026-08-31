# -*- coding: utf-8 -*-
"""
Байрны үнэ таамаглах Streamlit веб апп.

Хэрэглэгч талбай, давхар, ашиглалтын он, дүүргээ оруулаад товч дархад
өмнө нь сургасан загвар (train_model.py) ашиглан байрны ойролцоо үнийг
тооцоолж харуулна. Мөн загварын нарийвчлалын мэдээллийг (дундаж алдаа, R2)
хажууд нь харуулна.
"""

import json

import joblib
import streamlit as st

MODEL_FILE = "price_model.joblib"
DISTRICT_MAP_FILE = "district_mapping.json"
METRICS_FILE = "metrics.json"


@st.cache_resource
def load_model():
    """Сургасан загварыг дискнээс уншиж, кэшэлнэ (дахин дахин уншихаас сэргийлнэ)."""
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_district_map():
    """Сургалтын үед хадгалсан дүүрэг-нэр <-> код харгалзааг уншина.

    Энэ файл train_model.py-с үүсдэг бөгөөд загвар яг ижил кодоор сурсан тул
    веб апп дээр өөр харгалзаа ашиглавал таамаглал буруу гарна. Тиймээс
    харгалзааг код дотор дахин бичихгүй, зөвхөн энэ файлаас уншина.
    """
    with open(DISTRICT_MAP_FILE, "r", encoding="utf-8") as f:
        code_to_name = json.load(f)
    # JSON-ий түлхүүр үргэлж string байдаг тул int рүү хөрвүүлнэ
    code_to_name = {int(code): name for code, name in code_to_name.items()}
    name_to_code = {name: code for code, name in code_to_name.items()}
    return code_to_name, name_to_code


@st.cache_data
def load_metrics():
    """Загварын чанарын үзүүлэлтүүдийг (MAE, R2) уншина."""
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


st.set_page_config(page_title="Байрны үнэ таамаглагч", page_icon="🏠")

st.title("🏠 Байрны үнэ таамаглагч")
st.write(
    "Байрныхаа мэдээллийг оруулаад доорх товчийг дарвал ойролцоо зах зээлийн "
    "үнийг таамаглана."
)

model = load_model()
code_to_name, name_to_code = load_district_map()
metrics = load_metrics()

# --- Загварын чанарыг харуулах хэсэг ---
with st.sidebar:
    st.header("Загварын нарийвчлал")
    st.metric("Дундаж андуурал (MAE)", f"{metrics['mae']:,.0f} ₮")
    st.caption(
        "Загвар тест дата дээр дунджаар дээрх дүнгээр бодит үнээс "
        "андуурч таамагладаг гэсэн үг."
    )
    st.metric("Тайлбарлах чадвар (R²)", f"{metrics['r2']:.2f}")
    st.caption(
        "1.0 бол үнийн хэлбэлзлийг бүрэн тайлбарладаг, 0 бол огт "
        "тайлбарладаггүй гэсэн утгатай. Одоогийн загвар үнийн хэлбэлзлийн "
        f"ойролцоогоор {metrics['r2'] * 100:.0f}%-ийг тайлбарлаж чадаж байна."
    )

# --- Оролтын талбарууд ---
col1, col2 = st.columns(2)
with col1:
    area = st.number_input(
        "Талбай (м²)", min_value=10.0, max_value=500.0, value=50.0, step=0.5
    )
    floor = st.number_input("Давхар", min_value=1, max_value=30, value=5, step=1)
with col2:
    year = st.number_input(
        "Ашиглалтын он", min_value=1960, max_value=2026, value=2010, step=1
    )
    district_name = st.selectbox("Дүүрэг", options=list(name_to_code.keys()))

if st.button("Үнэ тооцоолох", type="primary"):
    district_code = name_to_code[district_name]

    # Загварыг сургахад ашигласантай яг ижил дарааллаар оролтын багана үүсгэнэ:
    # ["Талбай", "Давхар", "Ашиглалтын он", "Дүүрэг"]
    features = [[area, floor, year, district_code]]
    predicted_price = model.predict(features)[0]

    st.success(f"Таамагласан үнэ: **{predicted_price:,.0f} ₮**")
    st.caption(
        f"(Загвар дунджаар ±{metrics['mae']:,.0f} ₮ хүрээтэй андуурдгийг анхаарна уу)"
    )
