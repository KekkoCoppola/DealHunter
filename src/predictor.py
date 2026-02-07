# -*- coding: utf-8 -*-
"""
================================================================================
predictor.py - Modulo per le Predizioni in Produzione
================================================================================

Questo modulo contiene la classe SmartphonePredictor che incapsula tutta
la logica di predizione, rendendo il codice RIUTILIZZABILE in diversi contesti:
- app.py (Streamlit)
- API REST (se sviluppata in futuro)
- CLI (Command Line Interface)
- Notebook di testing

CLASSE PRINCIPALE: SmartphonePredictor

La classe gestisce automaticamente:
1. Caricamento degli artefatti (modello, scaler, feature names)
2. Preparazione dell'input (encoding, scaling)
3. Predizione con e senza probabilità
4. Validazione delle specifiche inserite
5. Gestione del DATA DRIFT per dispositivi recenti

DATA DRIFT MANAGEMENT:
    Il modello è stato addestrato su dati fino al 2020.
    Se l'utente inserisce un telefono del 2024:
    - model_age sarebbe negativo (2020 - 2024 = -4)
    - Il modello non ha mai visto valori negativi
    
    SOLUZIONE: Dispositivi post-2020 vengono trattati come "nuovissimi"
    (model_age = 0), ovvero top di gamma dell'anno di riferimento.

USO:
    from predictor import SmartphonePredictor, quick_predict
    
    # Metodo 1: Con classe (raccomandato per più predizioni)
    predictor = SmartphonePredictor()
    result = predictor.predict({'brand': 'Samsung', 'ram': 8, ...})
    
    # Metodo 2: Con funzione helper (per predizioni singole)
    result = quick_predict({'brand': 'Samsung', 'ram': 8, ...})

================================================================================
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Optional, List, Tuple

from config import (
    MODEL_PATH,
    SCALER_PATH,
    FEATURES_PATH,
    FEATURES_TO_SCALE,
    REFERENCE_YEAR,
    PRICE_RANGES
)


# ==============================================================================
# CLASSE PRINCIPALE: SmartphonePredictor
# ==============================================================================

class SmartphonePredictor:
    """
    Classe per predire la fascia di prezzo di uno smartphone.
    
    Incapsula modello, scaler e feature names, gestendo automaticamente
    tutto il preprocessing necessario per una predizione.
    
    DESIGN PATTERN: Facade
        La classe nasconde la complessità del preprocessing
        dietro un'interfaccia semplice (predict, predict_proba).
    
    ESEMPIO DI UTILIZZO:
        predictor = SmartphonePredictor()
        
        specs = {
            'brand': 'Samsung',
            'ram': 4.0,
            'internal_memory': 64,
            'rear_camera_mp': 12.0,
            'front_camera_mp': 8.0,
            'battery': 3000,
            'screen_size': 15.0,
            'weight': 160.0,
            'release_year': 2020,
            '4g': True,
            '5g': False
        }
        
        result = predictor.predict(specs)  # 'Mid-Range'
        probas = predictor.predict_proba(specs)  # {'Budget': 0.1, ...}
    """
    
    def __init__(self, 
                 model_path: str = MODEL_PATH,
                 scaler_path: str = SCALER_PATH,
                 features_path: str = FEATURES_PATH):
        """
        Inizializza il predictor caricando gli artefatti salvati.
        
        Gli artefatti vengono caricati UNA SOLA VOLTA al momento
        dell'istanziazione, per efficienza.
        
        Args:
            model_path: Path del modello addestrato serializzato
            scaler_path: Path dello StandardScaler serializzato
            features_path: Path della lista dei nomi delle feature
        """
        # --- Caricamento artefatti da disco ---
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        self.reference_year = REFERENCE_YEAR
    
    # ==========================================================================
    # METODI DI PREDIZIONE
    # ==========================================================================
    
    def predict(self, specs: Dict) -> str:
        """
        Predice la fascia di prezzo dato un dizionario di specifiche.
        
        Questo è il metodo principale per ottenere una predizione semplice.
        
        Args:
            specs: Dizionario con le specifiche del telefono:
                - brand: str (es. 'Samsung', 'Apple')
                - ram: float (GB)
                - internal_memory: float (GB)
                - rear_camera_mp: float (MP)
                - front_camera_mp: float (MP)
                - battery: float (mAh)
                - screen_size: float (cm)
                - weight: float (grammi)
                - release_year: int (anno di uscita)
                - 4g: bool
                - 5g: bool
                
        Returns:
            Stringa con la fascia di prezzo predetta:
            'Budget', 'Mid-Range', 'High-End', o 'Premium'
        """
        # Prepara l'input applicando encoding e scaling
        input_df = self._prepare_input(specs)
        
        # Esegue la predizione
        return self.model.predict(input_df)[0]
    
    def predict_proba(self, specs: Dict) -> Dict[str, float]:
        """
        Restituisce le probabilità per ogni fascia di prezzo.
        
        Utile per capire quanto il modello è "sicuro" della predizione.
        Es: Budget=0.01, Mid-Range=0.85, High-End=0.10, Premium=0.04
        → Il modello è abbastanza sicuro che sia Mid-Range
        
        Args:
            specs: Dizionario con le specifiche del telefono
            
        Returns:
            Dizionario {fascia: probabilità}
            Esempio: {'Budget': 0.1, 'Mid-Range': 0.6, 'High-End': 0.2, 'Premium': 0.1}
        """
        input_df = self._prepare_input(specs)
        probabilities = self.model.predict_proba(input_df)[0]
        
        # Associa ogni probabilità alla sua classe
        return dict(zip(self.model.classes_, probabilities))
    
    def predict_with_confidence(self, specs: Dict) -> Tuple[str, float, Dict[str, float]]:
        """
        Restituisce predizione, confidenza e tutte le probabilità.
        
        Metodo più completo che fornisce:
        1. La classe predetta
        2. La confidenza (probabilità della classe predetta)
        3. Tutte le probabilità
        
        Args:
            specs: Dizionario con le specifiche del telefono
            
        Returns:
            Tuple (predizione, confidenza, dizionario_probabilità)
            Esempio: ('Mid-Range', 0.85, {'Budget': 0.1, 'Mid-Range': 0.85, ...})
        """
        probas = self.predict_proba(specs)
        prediction = max(probas, key=probas.get)  # Classe con probabilità massima
        confidence = probas[prediction]           # Probabilità della classe vincente
        
        return prediction, confidence, probas
    
    # ==========================================================================
    # HELPER PER VISUALIZZAZIONE
    # ==========================================================================
    
    def get_price_range(self, category: str) -> str:
        """
        Restituisce il range di prezzo in Euro per una categoria.
        
        Converte la categoria interna in un valore comprensibile
        per l'utente finale.
        
        Args:
            category: Nome della categoria ('Budget', 'Mid-Range', etc.)
            
        Returns:
            Stringa con il range (es. '< 150€', '150€ - 300€')
        """
        return PRICE_RANGES.get(category, 'Sconosciuto')
    
    # ==========================================================================
    # PREPARAZIONE INPUT (METODO PRIVATO)
    # ==========================================================================
    
    def _prepare_input(self, specs: Dict) -> pd.DataFrame:
        """
        Prepara i dati di input per il modello.
        
        Questo metodo esegue tutte le trasformazioni necessarie:
        1. Crea DataFrame con colonne corrette
        2. Riempie valori numerici
        3. Gestisce Data Drift per anno
        4. Applica One-Hot Encoding per brand e connettività
        5. Scala le feature numeriche
        
        NOTA: Questo metodo è PRIVATO (inizia con _) perché
        l'utente non dovrebbe chiamarlo direttamente.
        
        Args:
            specs: Dizionario con le specifiche grezze
            
        Returns:
            DataFrame pronto per la predizione
        """
        # --- STEP 1: Creiamo DataFrame vuoto con le colonne corrette ---
        # Inizializziamo tutto a 0 (importante per le colonne one-hot)
        input_df = pd.DataFrame(0, index=[0], columns=self.feature_names)
        
        # --- STEP 2: Riempiamo i valori numerici ---
        input_df['ram'] = specs.get('ram', 4.0)
        input_df['internal_memory'] = specs.get('internal_memory', 64)
        input_df['rear_camera_mp'] = specs.get('rear_camera_mp', 12.0)
        input_df['front_camera_mp'] = specs.get('front_camera_mp', 8.0)
        input_df['battery'] = specs.get('battery', 3000)
        input_df['screen_size'] = specs.get('screen_size', 15.0)
        input_df['weight'] = specs.get('weight', 160.0)
        input_df['days_used'] = specs.get('days_used', 100)  # Valore medio fittizio
        
        # --- STEP 3: Gestione Data Drift per anno ---
        # PROBLEMA: Il modello è addestrato su dati fino al 2020.
        # Se l'utente inserisce release_year=2024, model_age sarebbe negativo!
        # SOLUZIONE: Trattiamo dispositivi futuri come "nuovissimi" (model_age=0)
        release_year = specs.get('release_year', self.reference_year)
        if release_year > self.reference_year:
            model_age = 0  # Top di gamma dell'anno di riferimento
        else:
            model_age = self.reference_year - release_year
        input_df['model_age'] = model_age
        
        # --- STEP 4: One-Hot Encoding del Brand ---
        # Se l'utente inserisce "Samsung", attiviamo la colonna device_brand_Samsung
        brand = specs.get('brand', 'Others')
        brand_col = f"device_brand_{brand}"
        if brand_col in input_df.columns:
            input_df[brand_col] = 1
        # Se il brand non esiste (drop_first o sconosciuto), resta tutto a 0 → trattato come Others
        
        # --- STEP 5: Encoding 4G/5G ---
        if specs.get('4g', True) and '4g_yes' in input_df.columns:
            input_df['4g_yes'] = 1
        if specs.get('5g', False) and '5g_yes' in input_df.columns:
            input_df['5g_yes'] = 1
        
        # --- STEP 6: Scaling delle feature numeriche ---
        # FONDAMENTALE: Usiamo LO STESSO scaler del training!
        cols_to_scale = [c for c in FEATURES_TO_SCALE if c in input_df.columns]
        input_df[cols_to_scale] = self.scaler.transform(input_df[cols_to_scale])
        
        return input_df
    
    # ==========================================================================
    # VALIDAZIONE INPUT
    # ==========================================================================
    
    def validate_specs(self, specs: Dict) -> List[str]:
        """
        Valida le specifiche e restituisce eventuali warning.
        
        Controlla che i valori inseriti siano ragionevoli per uno smartphone.
        Non blocca la predizione, ma avvisa l'utente di possibili errori.
        
        Args:
            specs: Dizionario con le specifiche
            
        Returns:
            Lista di warning (vuota se tutto ok)
        """
        warnings = []
        
        # Controllo range ragionevoli basati su conoscenza del dominio
        if specs.get('ram', 0) > 16:
            warnings.append("RAM > 16GB è insolita per uno smartphone")
        
        if specs.get('weight', 0) > 300:
            warnings.append("Peso > 300g potrebbe indicare un tablet")
        
        if specs.get('screen_size', 0) > 18:
            warnings.append("Schermo > 18cm potrebbe indicare un tablet")
        
        if specs.get('battery', 0) > 6000:
            warnings.append("Batteria > 6000mAh è insolita per smartphone")
        
        if specs.get('release_year', 2020) > self.reference_year:
            warnings.append(
                f"Dispositivo più recente del {self.reference_year}: "
                "valutazione approssimativa (possibile data drift)"
            )
        
        return warnings


# ==============================================================================
# FUNZIONI HELPER (per retrocompatibilità e uso rapido)
# ==============================================================================

def quick_predict(specs: Dict) -> str:
    """
    Funzione helper per predizioni rapide senza istanziare la classe.
    
    NOTA: Per più predizioni consecutive, è più efficiente creare
    un'istanza di SmartphonePredictor e riutilizzarla.
    
    Args:
        specs: Dizionario con le specifiche del telefono
        
    Returns:
        Fascia di prezzo predetta
    """
    predictor = SmartphonePredictor()
    return predictor.predict(specs)


def quick_predict_proba(specs: Dict) -> Dict[str, float]:
    """
    Funzione helper per probabilità rapide senza istanziare la classe.
    
    Args:
        specs: Dizionario con le specifiche del telefono
        
    Returns:
        Dizionario con probabilità per ogni fascia
    """
    predictor = SmartphonePredictor()
    return predictor.predict_proba(specs)
