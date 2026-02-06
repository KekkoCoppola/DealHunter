# -*- coding: utf-8 -*-
"""
================================================================================
feature_engineering.py - Modulo per Trasformazione delle Feature
================================================================================

Questo modulo gestisce la seconda fase del processo di Machine Learning:
la creazione e trasformazione delle feature per renderle adatte al training.

FUNZIONALITÀ PRINCIPALI:

1. CREAZIONE TARGET (price_category):
   - Trasforma il prezzo numerico continuo in 4 CLASSI DISCRETE
   - Usa pd.qcut per garantire un dataset BILANCIATO
   - Classi: Budget, Mid-Range, High-End, Premium

2. CREAZIONE MODEL_AGE:
   - Trasforma 'release_year' in 'model_age' (età del dispositivo)
   - Più interpretabile: "dispositivo vecchio di 3 anni" vs "uscito nel 2017"

3. ONE-HOT ENCODING:
   - Trasforma colonne categoriche in binarie
   - Esempio: device_brand="Samsung" → device_brand_Samsung=1
   - drop_first=True per evitare multicollinearità

4. RIMOZIONE COLONNE RISCHIOSE:
   - normalized_new_price rimosso per evitare DATA LEAKAGE
   - release_year rimosso (sostituito da model_age)

PERCHÉ QCUT E NON CUT:
   - cut: divide per intervalli di prezzo uguali → classi sbilanciate
   - qcut: divide per quantili → classi con stesso numero di elementi
   
   ESEMPIO:
   cut([0.1, 0.2, 0.3, 0.4, 10], 4) → [0.1-0.2:3items], [0.2-0.3:1item], ...
   qcut([0.1, 0.2, 0.3, 0.4, 10], 4) → ogni classe ha ~1-2 items

USO:
    from feature_engineering import engineer_features, prepare_features
    
    # Applica tutte le trasformazioni
    df = engineer_features(df)
    
    # Separa features e target
    X, y = prepare_features(df)

================================================================================
"""

import pandas as pd
from typing import Tuple, List, Optional

from config import (
    REFERENCE_YEAR,
    CATEGORICAL_COLUMNS,
    COLUMNS_TO_DROP,
    PRICE_CATEGORIES
)


# ==============================================================================
# STEP 1: CREAZIONE CATEGORIE DI PREZZO (TARGET)
# ==============================================================================

def create_price_categories(df: pd.DataFrame, 
                            n_categories: int = 4,
                            labels: Optional[List[str]] = None,
                            verbose: bool = False) -> pd.DataFrame:
    """
    Trasforma il prezzo normalizzato in categorie discrete usando QUANTILI.
    
    PROBLEMA:
       Il prezzo è una variabile CONTINUA (es. 4.35, 7.82, 2.15).
       Per un problema di CLASSIFICAZIONE, dobbiamo convertirlo in classi.
    
    SOLUZIONE - QCUT (Quantile Cut):
       Dividiamo i dati in fasce con lo STESSO NUMERO di elementi.
       Questo garantisce un dataset BILANCIATO fin dall'inizio.
       
       Risultato:
       - Budget: ~800 dispositivi
       - Mid-Range: ~800 dispositivi  
       - High-End: ~800 dispositivi
       - Premium: ~800 dispositivi
    
    ALTERNATIVA NON USATA - CUT:
       cut() divide per intervalli di prezzo uguali, ma se i prezzi
       sono distribuiti in modo non uniforme, le classi saranno sbilanciate.
    
    Args:
        df: DataFrame con colonna 'normalized_used_price'
        n_categories: Numero di fasce di prezzo (default: 4)
        labels: Etichette per le categorie. Se None, usa PRICE_CATEGORIES
        verbose: Se True, stampa la distribuzione delle classi
        
    Returns:
        DataFrame con nuova colonna 'price_category'
    """
    df = df.copy()
    
    if labels is None:
        labels = PRICE_CATEGORIES
    
    # Applica qcut per creare classi bilanciate
    df['price_category'] = pd.qcut(
        df['normalized_used_price'], 
        q=n_categories, 
        labels=labels
    )
    
    if verbose:
        print("\n📊 Distribuzione classi create con qcut:")
        print(df['price_category'].value_counts())
    
    return df


# ==============================================================================
# STEP 2: CREAZIONE MODEL_AGE
# ==============================================================================

def create_model_age(df: pd.DataFrame, 
                     reference_year: int = REFERENCE_YEAR,
                     verbose: bool = False) -> pd.DataFrame:
    """
    Crea la feature 'model_age' che rappresenta l'età del dispositivo.
    
    PROBLEMA:
       La colonna 'release_year' contiene valori assoluti (2018, 2019, 2020).
       Questi valori non sono intuitivi per il modello: 2019 non è "migliore" di 2018.
    
    SOLUZIONE:
       Creiamo 'model_age' = anno_riferimento - anno_uscita
       
       ESEMPIO (con anno_riferimento=2020):
       - Dispositivo del 2020 → model_age = 0 (nuovo)
       - Dispositivo del 2018 → model_age = 2 (vecchio di 2 anni)
       
       ORA IL MODELLO CAPISCE: model_age basso = migliore (più nuovo)
    
    Args:
        df: DataFrame con colonna 'release_year'
        reference_year: Anno di riferimento per il calcolo (default: 2020)
        verbose: Se True, stampa statistiche
        
    Returns:
        DataFrame con nuova colonna 'model_age'
    """
    df = df.copy()
    
    df['model_age'] = reference_year - df['release_year']
    
    if verbose:
        print(f"\n📅 model_age creato (riferimento: {reference_year})")
        print(f"   Range: {df['model_age'].min()} - {df['model_age'].max()} anni")
    
    return df


# ==============================================================================
# STEP 3: ONE-HOT ENCODING
# ==============================================================================

def encode_categorical(df: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       drop_first: bool = True,
                       verbose: bool = False) -> pd.DataFrame:
    """
    Applica One-Hot Encoding alle colonne categoriche.
    
    PROBLEMA:
       Le colonne categoriche contengono TESTO (es. "Samsung", "Apple").
       I modelli di ML lavorano solo con NUMERI.
    
    SOLUZIONE - ONE-HOT ENCODING:
       Trasformiamo ogni categoria in una colonna binaria (0/1).
       
       ESEMPIO:
       device_brand: [Samsung, Apple, Xiaomi]
       
       Diventa:
       device_brand_Apple: [0, 1, 0]
       device_brand_Xiaomi: [0, 0, 1]
       
       NOTA: Samsung non ha colonna perché drop_first=True
       (evita multicollinearità: se Apple=0 e Xiaomi=0 → è Samsung)
    
    PERCHÉ DROP_FIRST=TRUE:
       Evita la "dummy variable trap". Se abbiamo N categorie,
       bastano N-1 colonne binarie per rappresentarle tutte.
    
    Args:
        df: DataFrame con colonne categoriche
        columns: Lista di colonne da codificare. Se None, usa CATEGORICAL_COLUMNS
        drop_first: Se True, rimuove la prima categoria di ogni colonna
        verbose: Se True, stampa quante colonne sono state create
        
    Returns:
        DataFrame con colonne categoriche trasformate in binarie
    """
    if columns is None:
        columns = CATEGORICAL_COLUMNS
    
    # Filtra solo le colonne effettivamente presenti nel DataFrame
    columns_to_encode = [col for col in columns if col in df.columns]
    
    original_cols = len(df.columns)
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=drop_first)
    new_cols = len(df_encoded.columns)
    
    if verbose:
        print(f"\n🔤 One-Hot Encoding applicato:")
        print(f"   Colonne originali: {original_cols}")
        print(f"   Colonne dopo encoding: {new_cols}")
        print(f"   Nuove colonne create: {new_cols - original_cols + len(columns_to_encode)}")
    
    return df_encoded


# ==============================================================================
# STEP 4: RIMOZIONE COLONNE NON NECESSARIE
# ==============================================================================

def drop_unused_columns(df: pd.DataFrame,
                        columns: Optional[List[str]] = None,
                        verbose: bool = False) -> pd.DataFrame:
    """
    Rimuove le colonne non necessarie per il training.
    
    COLONNE RIMOSSE:
    
    1. release_year: Sostituito da 'model_age' (più interpretabile)
    
    2. normalized_used_price: Sostituito da 'price_category' (il nostro target)
    
    3. normalized_new_price: ATTENZIONE - DATA LEAKAGE!
       Il prezzo nuovo è fortemente correlato al prezzo usato.
       Se lo lasciamo, il modello "imbroglia": predice il prezzo usato
       guardando il prezzo nuovo invece delle specifiche tecniche.
    
    Args:
        df: DataFrame originale
        columns: Colonne da rimuovere. Se None, usa COLUMNS_TO_DROP
        verbose: Se True, stampa quali colonne sono state rimosse
        
    Returns:
        DataFrame senza le colonne specificate
    """
    if columns is None:
        columns = COLUMNS_TO_DROP
    
    # Rimuovi solo le colonne che esistono
    columns_to_drop = [col for col in columns if col in df.columns]
    
    if verbose and columns_to_drop:
        print(f"\n🗑️  Colonne rimosse: {columns_to_drop}")
    
    return df.drop(columns=columns_to_drop)


# ==============================================================================
# STEP 5: SEPARAZIONE FEATURES E TARGET
# ==============================================================================

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa il DataFrame in features (X) e target (y).
    
    Questa è l'ultima operazione prima dello split train/test.
    
    Args:
        df: DataFrame già processato con 'price_category'
        
    Returns:
        Tuple (X, y) dove:
        - X: DataFrame con tutte le features (senza target)
        - y: Series con il target (price_category)
    """
    X = df.drop(columns=['price_category'])
    y = df['price_category']
    return X, y


# ==============================================================================
# PIPELINE COMPLETA DI FEATURE ENGINEERING
# ==============================================================================

def engineer_features(df: pd.DataFrame, 
                      reference_year: int = REFERENCE_YEAR,
                      verbose: bool = False) -> pd.DataFrame:
    """
    Pipeline completa di feature engineering.
    
    Esegue in sequenza tutti gli step di trasformazione:
    1. Creazione categorie di prezzo (target bilanciato)
    2. Creazione model_age (trasformazione anno → età)
    3. Rimozione colonne rischiose (data leakage prevention)
    4. One-Hot Encoding (categoriche → numeriche)
    
    Args:
        df: DataFrame pulito (output di clean_dataset)
        reference_year: Anno di riferimento per model_age
        verbose: Se True, stampa informazioni per ogni step
        
    Returns:
        DataFrame pronto per lo split e il training
    """
    if verbose:
        print("\n" + "-" * 40)
        print("   FEATURE ENGINEERING")
        print("-" * 40)
    
    # --- STEP 1: Creazione target (categorie di prezzo) ---
    df = create_price_categories(df, verbose=verbose)
    
    # --- STEP 2: Trasformazione anno → età ---
    df = create_model_age(df, reference_year, verbose=verbose)
    
    # --- STEP 3: Rimozione colonne inutili/rischiose ---
    df = drop_unused_columns(df, verbose=verbose)
    
    # --- STEP 4: One-Hot Encoding ---
    df = encode_categorical(df, verbose=verbose)
    
    if verbose:
        print(f"\n✅ Dataset finale: {df.shape[0]} righe × {df.shape[1]} colonne")
    
    return df
