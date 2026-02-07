# -*- coding: utf-8 -*-
"""
================================================================================
config.py - Configurazione Centralizzata per DealHunter
================================================================================

Questo modulo contiene TUTTE le costanti, path e parametri condivisi tra i moduli.
Centralizzare la configurazione offre vantaggi importanti:

1. MANUTENIBILITÀ: Per cambiare una soglia, modifichi un solo file
2. CONSISTENZA: Tutti i moduli usano gli stessi valori
3. DOCUMENTAZIONE: Tutti i parametri sono raccolti e spiegati qui

ORGANIZZAZIONE:
    - PATH DEI FILE: Dove trovare e salvare dati e modelli
    - COLONNE DATASET: Quali colonne processare e come
    - SOGLIE OUTLIER: Parametri per la rimozione dei tablet
    - PARAMETRI MODELLO: Configurazione per il training
    - BRAND SUPPORTATI: Marche riconosciute dall'applicazione

USO:
    from config import DATASET_PATH, RANDOM_STATE, PRICE_CATEGORIES
    
    df = pd.read_csv(DATASET_PATH)

================================================================================
"""

import os


# ==============================================================================
# PATH DEI FILE
# ==============================================================================

# Cartella base del progetto (un livello sopra /src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sottocartelle principali
MODEL_DIR = os.path.join(BASE_DIR, 'model')   # Artefatti salvati (modello, scaler)
DATA_DIR = os.path.join(BASE_DIR, 'data')     # Dataset CSV

# Directory per i grafici
ANALYTICS_DIR = os.path.join(BASE_DIR, 'analytics')
os.makedirs(ANALYTICS_DIR, exist_ok=True)

# --- Path specifici dei file ---

# Modello addestrato (Logistic Regression o modello migliore)
MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')

# StandardScaler con parametri (media, std) del training
# FONDAMENTALE: usare lo stesso scaler per nuovi dati!
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Lista ordinata dei nomi delle feature
# Garantisce che i nuovi dati abbiano le colonne nell'ordine corretto
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')

# Dataset originale (grezzo)
DATASET_PATH = os.path.join(DATA_DIR, 'used_device_data.csv')

# Dataset processato (dopo preprocessing)
# Utile per saltare la pulizia nelle esecuzioni successive
PROCESSED_DATASET_PATH = os.path.join(DATA_DIR, 'used_device_data_processed.csv')


# ==============================================================================
# COLONNE DEL DATASET
# ==============================================================================

# --- Colonne con zeri fittizi (0 = dato mancante, non valore reale) ---
# Un telefono non può avere 0 grammi di peso o 0 MP di fotocamera
COLS_WITH_HIDDEN_MISSING = [
    'rear_camera_mp',    # Megapixel fotocamera posteriore
    'front_camera_mp',   # Megapixel fotocamera anteriore
    'internal_memory',   # Memoria interna (GB)
    'ram',               # RAM (GB)
    'battery',           # Capacità batteria (mAh)
    'weight',            # Peso (grammi)
    'screen_size'        # Dimensione schermo (cm)
]

# --- Colonne numeriche da scalare (StandardScaler) ---
# Queste hanno scale diverse (battery: 1000-5000, ram: 1-12)
# Lo scaling le normalizza tutte alla stessa scala
FEATURES_TO_SCALE = [
    'screen_size',
    'rear_camera_mp',
    'front_camera_mp',
    'internal_memory',
    'ram',
    'battery',
    'weight',
    'days_used',
    'model_age'      # Creata durante feature engineering
]

# --- Colonne categoriche (One-Hot Encoding) ---
# Contengono testo che deve essere convertito in numeri
CATEGORICAL_COLUMNS = [
    'device_brand',  # Samsung, Apple, Xiaomi, ecc.
    'os',            # android, ios, altro
    '4g',            # yes/no
    '5g'             # yes/no
]

# --- Colonne da rimuovere dopo il preprocessing ---
COLUMNS_TO_DROP = [
    'release_year',           # Sostituito da 'model_age'
    'normalized_used_price',  # Sostituito da 'price_category' (target)
    'normalized_new_price'    # RIMOSSO per evitare DATA LEAKAGE!
                              # Il prezzo nuovo è correlato all'usato
]


# ==============================================================================
# SOGLIE PER RIMOZIONE OUTLIER
# ==============================================================================

# Peso massimo in grammi per considerare un dispositivo smartphone
# Soglia basata su conoscenza del dominio:
# - Smartphone pesanti (es. iPhone Pro Max): ~240g
# - Foldable (es. Samsung Fold): ~270g
# - Tablet (es. iPad Mini): ~500g
# Soglia 350g cattura tutti gli smartphone e esclude i tablet
MAX_WEIGHT_THRESHOLD = 350  # grammi

# Dimensione schermo massima in cm
# - Smartphone grandi: 17 cm (6.8 pollici)
# - Tablet piccoli: 20+ cm (8+ pollici)
MAX_SCREEN_THRESHOLD = 20   # cm


# ==============================================================================
# PARAMETRI DEL MODELLO
# ==============================================================================

# Anno di riferimento per calcolare model_age
# model_age = REFERENCE_YEAR - release_year
# Usato per rendere l'età interpretabile: 0 = nuovo, alto = vecchio
REFERENCE_YEAR = 2020

# Seed per riproducibilità
# Usando sempre lo stesso seed, gli esperimenti sono ripetibili
RANDOM_STATE = 42

# --- Categorie di prezzo (Target del modello) ---
# Ordinate dal più economico al più costoso
PRICE_CATEGORIES = ['Budget', 'Mid-Range', 'High-End', 'Premium']

# Fasce di prezzo in Euro (per visualizzazione nell'app)
# Questi valori sono indicativi, basati sul mercato italiano
PRICE_RANGES = {
    'Budget': '< 150€',
    'Mid-Range': '150€ - 300€',
    'High-End': '300€ - 600€',
    'Premium': '> 600€'
}


# ==============================================================================
# BRAND SUPPORTATI
# ==============================================================================

# Lista delle marche riconosciute dall'applicazione
# I brand non in lista vengono trattati come 'Others'
SUPPORTED_BRANDS = [
    'Samsung', 
    'Apple', 
    'Huawei', 
    'Xiaomi', 
    'Oppo', 
    'LG', 
    'Lenovo', 
    'Sony', 
    'Honor', 
    'Others'
]


# ==============================================================================
# GRIGLIA IPERPARAMETRI (GridSearchCV)
# ==============================================================================

# Combinazioni di parametri testate durante l'ottimizzazione del Random Forest
# GridSearchCV prova tutte le combinazioni (36 totali) e seleziona la migliore
RF_PARAM_GRID = {
    # Numero di alberi nella foresta
    # Più alberi = predizioni più stabili ma training più lento
    'n_estimators': [100, 200],
    
    # Profondità massima di ogni albero
    # None = nessun limite (rischio overfitting)
    # 10-20 = limitato (più generalizzabile)
    'max_depth': [10, 20, None],
    
    # Minimo numero di campioni per dividere un nodo
    # Valori alti = alberi più semplici
    'min_samples_split': [2, 5, 10],
    
    # Bilanciamento classi
    # 'balanced' = pesa di più le classi minoritarie
    'class_weight': ['balanced', None]
}
