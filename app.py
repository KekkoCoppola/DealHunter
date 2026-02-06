import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="DealHunter", page_icon="📱")

# --- TITOLO E DESCRIZIONE ---
st.title("📱 DealHunter: Valuta il tuo Smartphone")
st.markdown("""
Questa applicazione utilizza un modello di **Machine Learning (Random Forest)** per stimare la fascia di prezzo corretta di un telefono usato.
Inserisci le specifiche tecniche qui sotto per scoprire se l'annuncio è un affare o una truffa!
I dati sono relativi alle fascie di prezzo dell'anno 2020
futuri aggiornamenti adatteranno tali dati ad anni più recenti
""")

# --- CARICAMENTO MODELLO (CACHE) ---
# Usiamo la cache così non lo ricarica a ogni click
@st.cache_resource
def load_model():
    model = joblib.load('model/random_forest_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    features = joblib.load('model/feature_names.pkl')
    return model, scaler, features

try:
    model, scaler, feature_names = load_model()
except FileNotFoundError:
    st.error("Errore: File del modello non trovati. Assicurati di aver scaricato i file .pkl nella cartella 'model'.")
    st.stop()

# --- SIDEBAR (INPUT UTENTE) ---
st.sidebar.header("Specifiche Tecniche")

def user_input_features():
    # Brand (Menu a tendina)
    brand = st.sidebar.selectbox("Marca", ["Samsung", "Apple", "Huawei", "Xiaomi", "Oppo", "LG", "Lenovo", "Sony", "Others"])
    
    # Specifiche Numeriche (Slider e Input)
    ram = st.sidebar.slider("RAM (GB)", 0.5, 16.0, 4.0)
    memory = st.sidebar.selectbox("Memoria Interna (GB)", [16, 32, 64, 128, 256, 512, 1024])
    rear_cam = st.sidebar.slider("Fotocamera Posteriore (MP)", 2.0, 108.0, 12.0)
    front_cam = st.sidebar.slider("Fotocamera Anteriore (MP)", 2.0, 40.0, 8.0)
    battery = st.sidebar.slider("Batteria (mAh)", 1000, 6000, 3000)
    screen = st.sidebar.slider("Schermo (cm)", 10.0, 18.0, 15.0)
    weight = st.sidebar.slider("Peso (g)", 100.0, 300.0, 160.0)
    
    # Anno (Gestione Data Drift)
    year = st.sidebar.number_input("Anno di Uscita", min_value=2010, max_value=2026, value=2020)
    
    has_4g = st.sidebar.checkbox("Supporto 4G", value=True)
    has_5g = st.sidebar.checkbox("Supporto 5G", value=False)

    # --- PREPROCESSING LIVE ---
    # Creiamo un dizionario con i dati
    data = {
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
    return data

input_data = user_input_features()

# --- BOTTONE PREDIZIONE ---
if st.button("🔮 Calcola Prezzo Giusto"):
    
    # 1. Creiamo il DataFrame vuoto con le colonne giuste
    # Questo serve per garantire che l'ordine delle colonne sia identico al training
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # 2. Riempiamo i valori numerici
    input_df['ram'] = input_data['ram']
    input_df['internal_memory'] = input_data['internal_memory']
    input_df['rear_camera_mp'] = input_data['rear_camera_mp']
    input_df['front_camera_mp'] = input_data['front_camera_mp']
    input_df['battery'] = input_data['battery']
    input_df['screen_size'] = input_data['screen_size']
    input_df['weight'] = input_data['weight']
    input_df['days_used'] = 100 # Valore medio fittizio
    
    # Gestione Anno (Data Drift)
    if input_data['release_year'] > 2020:
        st.warning("⚠️ Nota: Il telefono è più recente del dataset (2020). Verrà valutato come un Top di Gamma del 2020.")
        model_age = 0
    else:
        model_age = 2020 - input_data['release_year']
    input_df['model_age'] = model_age

    # 3. One-Hot Encoding Manuale
    # Brand
    brand_col = f"device_brand_{input_data['brand']}"
    if brand_col in input_df.columns:
        input_df[brand_col] = 1
        
    # 4G/5G
    if input_data['4g'] and '4g_yes' in input_df.columns: input_df['4g_yes'] = 1
    if input_data['5g'] and '5g_yes' in input_df.columns: input_df['5g_yes'] = 1

    # 4. Scaling
    cols_to_scale = ['screen_size', 'rear_camera_mp', 'front_camera_mp', 
                     'internal_memory', 'ram', 'battery', 'weight', 'days_used', 'model_age']
    # Scaliamo solo le colonne che esistono effettivamente
    existing_cols = [c for c in cols_to_scale if c in input_df.columns]
    input_df[existing_cols] = scaler.transform(input_df[existing_cols])

    # 5. Predizione
    prediction = model.predict(input_df)[0]
    
    # --- RISULTATO VISIVO ---
    st.divider()
    st.subheader(f"Risultato: Fascia {prediction}")
    
    if prediction == 'Budget':
        st.success("💰 **Fascia Economica**: Prezzo stimato < 150€")
    elif prediction == 'Mid-Range':
        st.info("⚖️ **Fascia Media**: Prezzo stimato 150€ - 300€")
    elif prediction == 'High-End':
        st.warning("🚀 **Fascia Alta**: Prezzo stimato 300€ - 600€")
    elif prediction == 'Premium':
        st.error("💎 **Top di Gamma**: Prezzo stimato > 600€")
        
    # Debug (facoltativo, rimuovere per la versione finale)
    with st.expander("Vedi dati processati (Debug)"):
        st.write(input_df)