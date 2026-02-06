# -*- coding: utf-8 -*-
"""
================================================================================
app.py - Interfaccia Web DealHunter (Streamlit)
================================================================================

Dashboard interattiva per la valutazione del prezzo di smartphone usati.
Usa Logistic Regression per classificare in 4 fasce di prezzo.

AVVIO:
    cd src
    streamlit run app.py

================================================================================
"""

import streamlit as st
import os

from predictor import SmartphonePredictor
from config import SUPPORTED_BRANDS, REFERENCE_YEAR, PRICE_RANGES


# ==============================================================================
# CONFIGURAZIONE PAGINA E TEMA
# ==============================================================================

st.set_page_config(
    page_title="DealHunter",
    page_icon="📱",
    layout="centered"
)

# Colori del brand
PRIMARY_COLOR = "#1B6BB2"    # Blu
SECONDARY_COLOR = "#EA7900"  # Arancione

# CSS personalizzato per i colori del tema
st.markdown(f"""
<style>
    /* Colore primario per bottoni e accenti */
    .stButton > button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border: none;
        font-weight: bold;
    }}
    .stButton > button:hover {{
        background-color: {SECONDARY_COLOR};
        color: white;
    }}
    
    /* Titoli */
    h1, h2, h3 {{
        color: {PRIMARY_COLOR};
    }}
    
    /* Link */
    a {{
        color: {SECONDARY_COLOR};
    }}
    
    /* Metriche */
    [data-testid="stMetricValue"] {{
        color: {PRIMARY_COLOR};
    }}
    
    /* Divider colorato */
    hr {{
        border-color: {SECONDARY_COLOR};
    }}
    
    /* Header centrato */
    .header-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 1rem;
    }}
    
    .header-container img {{
        max-width: 200px;
        margin-bottom: 0.5rem;
    }}
    
    /* Box risultato */
    .result-box {{
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }}
    
    .result-budget {{
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
    }}
    
    .result-midrange {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, #1557a0 100%);
        color: white;
    }}
    
    .result-highend {{
        background: linear-gradient(135deg, {SECONDARY_COLOR} 0%, #d16d00 100%);
        color: white;
    }}
    
    .result-premium {{
        background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%);
        color: white;
    }}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HEADER CON LOGO
# ==============================================================================

# Path del logo
logo_path = os.path.join(os.path.dirname(__file__), 'img', 'MainLogo.png')

# Logo centrato con HTML (non zoomabile)
import base64

if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    
    st.markdown(f"""
    <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
        <img src="data:image/png;base64,{logo_base64}" 
             style="width: 60%; max-width: 350px; min-width: 200px; pointer-events: none;" 
             alt="DealHunter Logo">
    </div>
    """, unsafe_allow_html=True)
else:
    st.title("📱 DealHunter")

st.markdown("""
<p style="text-align: center; font-size: 1.1rem; color: #666; margin-top: 0;">
    Scopri se il prezzo del tuo smartphone è giusto!
</p>
""", unsafe_allow_html=True)

st.divider()


# ==============================================================================
# CARICAMENTO MODELLO
# ==============================================================================

@st.cache_resource
def load_predictor():
    """Carica il predictor (cached per performance)."""
    return SmartphonePredictor()

try:
    predictor = load_predictor()
except FileNotFoundError:
    st.error("❌ Modello non trovato! Esegui prima `python pipeline.py`")
    st.stop()


# ==============================================================================
# INPUT FORM
# ==============================================================================

st.subheader("📋 Specifiche Tecniche")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox(
        "🏷️ Marca",
        SUPPORTED_BRANDS,
        help="Seleziona la marca del dispositivo"
    )
    
    ram = st.number_input(
        "💾 RAM (GB)",
        min_value=0.5,
        max_value=24.0,
        value=4.0,
        step=0.5,
        help="Memoria RAM in GB"
    )
    
    memory = st.number_input(
        "📦 Memoria Interna (GB)",
        min_value=8,
        max_value=1024,
        value=64,
        step=8,
        help="Spazio di archiviazione in GB"
    )
    
    rear_cam = st.number_input(
        "📸 Fotocamera Posteriore (MP)",
        min_value=1.0,
        max_value=200.0,
        value=12.0,
        step=1.0,
        help="Megapixel della fotocamera principale"
    )
    
    front_cam = st.number_input(
        "🤳 Fotocamera Anteriore (MP)",
        min_value=1.0,
        max_value=50.0,
        value=8.0,
        step=1.0,
        help="Megapixel della fotocamera selfie"
    )

with col2:
    battery = st.number_input(
        "🔋 Batteria (mAh)",
        min_value=500,
        max_value=10000,
        value=3000,
        step=100,
        help="Capacità della batteria"
    )
    
    screen = st.number_input(
        "📐 Schermo (cm)",
        min_value=5.0,
        max_value=25.0,
        value=15.0,
        step=0.5,
        help="Diagonale dello schermo in cm"
    )
    
    weight = st.number_input(
        "⚖️ Peso (g)",
        min_value=50,
        max_value=500,
        value=160,
        step=5,
        help="Peso del dispositivo in grammi"
    )
    
    year = st.number_input(
        "📅 Anno di Uscita",
        min_value=2010,
        max_value=2026,
        value=2020,
        step=1,
        help="Anno di lancio del modello"
    )
    
    st.markdown("**📶 Connettività**")
    col_4g, col_5g = st.columns(2)
    with col_4g:
        has_4g = st.checkbox("4G", value=True)
    with col_5g:
        has_5g = st.checkbox("5G", value=False)

input_data = {
    'brand': brand,
    'ram': ram,
    'internal_memory': memory,
    'rear_camera_mp': rear_cam,
    'front_camera_mp': front_cam,
    'battery': battery,
    'screen_size': screen,
    'weight': weight,
    'release_year': year,
    '4g': has_4g,
    '5g': has_5g
}


# ==============================================================================
# PREDIZIONE
# ==============================================================================

st.divider()

if st.button("🔮 Calcola Prezzo Giusto", type="primary", use_container_width=True):
    
    warnings = predictor.validate_specs(input_data)
    for warning in warnings:
        st.warning(f"⚠️ {warning}")
    
    prediction, confidence, probas = predictor.predict_with_confidence(input_data)
    price_range = predictor.get_price_range(prediction)
    
    st.divider()
    
    # Box risultato con colori personalizzati
    if prediction == 'Budget':
        st.markdown(f"""
        <div class="result-box result-budget">
            <h2>💰 Fascia: {prediction}</h2>
            <p style="font-size: 1.3rem;">Prezzo stimato: <strong>{price_range}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    elif prediction == 'Mid-Range':
        st.markdown(f"""
        <div class="result-box result-midrange">
            <h2>⚖️ Fascia: {prediction}</h2>
            <p style="font-size: 1.3rem;">Prezzo stimato: <strong>{price_range}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    elif prediction == 'High-End':
        st.markdown(f"""
        <div class="result-box result-highend">
            <h2>🚀 Fascia: {prediction}</h2>
            <p style="font-size: 1.3rem;">Prezzo stimato: <strong>{price_range}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    elif prediction == 'Premium':
        st.markdown(f"""
        <div class="result-box result-premium">
            <h2>💎 Fascia: {prediction}</h2>
            <p style="font-size: 1.3rem;">Prezzo stimato: <strong>{price_range}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Confidenza
    col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
    with col_m2:
        st.metric("🎯 Confidenza", f"{confidence:.1%}")
    
    # Grafico probabilità
    st.subheader("📊 Distribuzione Probabilità")
    st.bar_chart(probas)
    
    with st.expander("🔧 Dettagli tecnici"):
        st.json(input_data)
        st.write("Probabilità:", probas)


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()
st.markdown(f"""
<p style="text-align: center; color: #888; font-size: 0.9rem;">
    DealHunter - Machine Learning per la valutazione di smartphone usati<br>
    <span style="color: {SECONDARY_COLOR};">Modello: Logistic Regression | Accuracy: ~65%</span>
</p>
""", unsafe_allow_html=True)