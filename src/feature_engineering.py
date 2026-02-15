# -*- coding: utf-8 -*-
"""
================================================================================
feature_engineering.py - Modulo per Trasformazione delle Feature
================================================================================

Questo modulo gestisce la seconda fase del processo di Machine Learning:
la creazione e trasformazione delle feature per renderle adatte al training.

FUNZIONALITÀ PRINCIPALI:

1. DISCRETIZZAZIONE TARGET (price_category) - PATTERN FIT/TRANSFORM:
   - fit_price_categories(): calcola i bin con pd.qcut SOLO su y_train
   - transform_price_categories(): applica pd.cut con i bin del train su y_test
   - Questo PREVIENE il Data Leakage: i quantili del test non influenzano i bin

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

FLUSSO CORRETTO (SENZA DATA LEAKAGE):
    from feature_engineering import engineer_features, prepare_features
    from feature_engineering import fit_price_categories, transform_price_categories
    
    df = engineer_features(df)       # Feature engineering (SENZA target)
    X, y_price = prepare_features(df) # y_price è numerico
    X_train, X_test, y_train_price, y_test_price = split(...)
    y_train, bins = fit_price_categories(y_train_price)     # bin da SOLO train
    y_test = transform_price_categories(y_test_price, bins)  # applica su test

================================================================================
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

from config import (
    REFERENCE_YEAR,
    CATEGORICAL_COLUMNS,
    COLUMNS_TO_DROP,
    PRICE_CATEGORIES
)


# ==============================================================================
# STEP 1: DISCRETIZZAZIONE TARGET (PATTERN FIT/TRANSFORM)
# ==============================================================================

def fit_price_categories(y_train: pd.Series,
                         n_categories: int = 4,
                         labels: Optional[List[str]] = None,
                         verbose: bool = False) -> Tuple[pd.Series, np.ndarray]:
    """
    Calcola i bin dei quantili SOLO su y_train e discretizza il target.
    
    QUESTA È LA FUNZIONE "FIT":
       Usa pd.qcut per calcolare i bordi dei bin dalla distribuzione
       dei prezzi del SOLO training set. I bin vengono poi restituiti
       per essere riutilizzati su test set e dati futuri.
    
    PERCHÉ SOLO SU TRAIN:
       Calcolare i quantili sull'intero dataset (train+test) causerebbe
       DATA LEAKAGE: il modello "vedrebbe" la distribuzione dei prezzi
       del test set durante il training.
    
    Args:
        y_train: Series con i prezzi numerici del training set
        n_categories: Numero di fasce di prezzo (default: 4)
        labels: Etichette per le categorie. Se None, usa PRICE_CATEGORIES
        verbose: Se True, stampa distribuzione e bordi dei bin
        
    Returns:
        Tuple (y_train_cat, bins) dove:
        - y_train_cat: Series con le categorie (Budget, Mid-Range, ...)
        - bins: np.ndarray con i bordi dei bin (per riuso su test)
    """
    if labels is None:
        labels = PRICE_CATEGORIES
    
    # qcut calcola i quantili e restituisce anche i bordi (retbins=True)
    y_train_cat, bins = pd.qcut(
        y_train,
        q=n_categories,
        labels=labels,
        retbins=True
    )
    
    # Estendiamo i bordi estremi a -inf/+inf per gestire valori fuori range nel test
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    if verbose:
        print("\n📊 Bin calcolati su TRAINING SET (no data leakage):")
        print(f"   Bordi dei bin: {bins}")
        print(f"   Distribuzione classi train:")
        print(y_train_cat.value_counts().to_string())
    
    return y_train_cat, bins


def transform_price_categories(y: pd.Series,
                                bins: np.ndarray,
                                labels: Optional[List[str]] = None,
                                verbose: bool = False) -> pd.Series:
    """
    Applica i bin pre-calcolati dal train per discretizzare un qualsiasi set.
    
    QUESTA È LA FUNZIONE "TRANSFORM":
       Usa pd.cut (NON qcut!) con i bordi calcolati da fit_price_categories.
       In questo modo il test set viene trattato come dati futuri "invisibili".
    
    DIFFERENZA CUT vs QCUT:
       - qcut: calcola i quantili DAI DATI → data leakage se usato su test
       - cut: usa bordi FISSI pre-definiti → nessun leakage
    
    Args:
        y: Series con i prezzi numerici (test set o nuovi dati)
        bins: Bordi dei bin calcolati da fit_price_categories
        labels: Etichette per le categorie. Se None, usa PRICE_CATEGORIES
        verbose: Se True, stampa la distribuzione risultante
        
    Returns:
        Series con le categorie assegnate
    """
    if labels is None:
        labels = PRICE_CATEGORIES
    
    y_cat = pd.cut(y, bins=bins, labels=labels)
    
    if verbose:
        print("\n📊 Distribuzione classi (transform con bin del train):")
        print(y_cat.value_counts().to_string())
    
    return y_cat


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
    # NOTA: normalized_used_price NON viene rimossa qui perché serve
    # come target numerico per lo split. Sarà separata in prepare_features().
    columns_to_drop = [col for col in columns if col in df.columns]
    
    if verbose and columns_to_drop:
        print(f"\n🗑️  Colonne rimosse: {columns_to_drop}")
    
    return df.drop(columns=columns_to_drop)


# ==============================================================================
# STEP 5: SEPARAZIONE FEATURES E TARGET
# ==============================================================================

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa il DataFrame in features (X) e target numerico (y_price).
    
    IMPORTANTE: Restituisce y come prezzo NUMERICO (normalized_used_price),
    NON come categoria. La discretizzazione avviene DOPO lo split
    tramite fit_price_categories / transform_price_categories.
    
    Questa è l'ultima operazione prima dello split train/test.
    
    Args:
        df: DataFrame processato (con model_age, encoding, ecc.)
        
    Returns:
        Tuple (X, y_price) dove:
        - X: DataFrame con tutte le features (senza target)
        - y_price: Series con il prezzo numerico (normalized_used_price)
    """
    X = df.drop(columns=['normalized_used_price'])
    y_price = df['normalized_used_price']
    return X, y_price


# ==============================================================================
# PIPELINE COMPLETA DI FEATURE ENGINEERING
# ==============================================================================

def engineer_features(df: pd.DataFrame, 
                      reference_year: int = REFERENCE_YEAR,
                      verbose: bool = False) -> pd.DataFrame:
    """
    Pipeline completa di feature engineering.
    
    Esegue in sequenza tutti gli step di trasformazione:
    1. Creazione model_age (trasformazione anno → età)
    2. Rimozione colonne rischiose (data leakage prevention)
    3. One-Hot Encoding (categoriche → numeriche)
    
    NOTA: La discretizzazione del target (price_category) NON avviene qui.
    Viene eseguita DOPO lo split train/test tramite fit_price_categories
    e transform_price_categories, per evitare Data Leakage.
    
    Args:
        df: DataFrame pulito (output di clean_dataset)
        reference_year: Anno di riferimento per model_age
        verbose: Se True, stampa informazioni per ogni step
        
    Returns:
        DataFrame pronto per lo split (contiene ancora normalized_used_price)
    """
    if verbose:
        print("\n" + "-" * 40)
        print("   FEATURE ENGINEERING")
        print("-" * 40)
    
    # --- STEP 1: Trasformazione anno → età ---
    df = create_model_age(df, reference_year, verbose=verbose)
    
    # --- STEP 2: Rimozione colonne inutili/rischiose ---
    df = drop_unused_columns(df, verbose=verbose)
    
    # --- STEP 3: One-Hot Encoding ---
    df = encode_categorical(df, verbose=verbose)
    
    if verbose:
        print(f"\n✅ Dataset finale: {df.shape[0]} righe × {df.shape[1]} colonne")
    
    return df
