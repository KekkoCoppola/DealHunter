# -*- coding: utf-8 -*-
"""
================================================================================
preprocessing.py - Modulo per Pulizia e Preparazione dei Dati
================================================================================

Questo modulo gestisce la prima fase del processo di Machine Learning:
la pulizia e preparazione del dataset grezzo.

PROBLEMA AFFRONTATO:
    Il dataset contiene diversi problemi che devono essere risolti prima del training:
    1. ZERI FITTIZI: Valori 0 che rappresentano dati mancanti (es. peso=0 impossibile)
    2. MISSING VALUES: Valori NaN che devono essere imputati intelligentemente
    3. OUTLIER: Tablet inclusi nel dataset che devono essere rimossi

STRATEGIA DI IMPUTAZIONE:
    Utilizziamo una strategia a DUE LIVELLI:
    1. MEDIANA LOCALE: Raggruppiamo per (Brand, Anno) e usiamo la mediana del gruppo
       Es: Un Samsung del 2020 con RAM mancante → mediana RAM degli altri Samsung 2020
    2. FALLBACK GLOBALE: Se il gruppo è troppo piccolo, usiamo la mediana globale
    
    MOTIVAZIONE: Questa strategia è "domain-driven" e preserva la struttura dei dati
    meglio di una semplice mediana globale o KNN Imputer.

GESTIONE OUTLIER:
    Utilizziamo un'ANALISI BIVARIATA Peso vs Schermo per identificare i tablet.
    Soglie: Peso > 350g OR Schermo > 20cm → rimosso
    MOTIVAZIONE: Soglia basata su conoscenza del dominio (max smartphone ~250g)

USO:
    from preprocessing import clean_dataset, load_processed_dataset
    
    # Prima esecuzione: pulisce e salva
    df = clean_dataset('data/used_device_data.csv', save_path='data/processed.csv')
    
    # Esecuzioni successive: carica direttamente
    df = load_processed_dataset('data/processed.csv')

================================================================================
"""

import pandas as pd
import numpy as np
from typing import List, Optional

from config import (
    COLS_WITH_HIDDEN_MISSING,
    MAX_WEIGHT_THRESHOLD,
    MAX_SCREEN_THRESHOLD
)


# ==============================================================================
# STEP 1: CARICAMENTO DATI
# ==============================================================================

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Carica il dataset CSV in un DataFrame pandas.
    
    Args:
        filepath: Percorso del file CSV
        
    Returns:
        DataFrame con i dati grezzi, non processati
    """
    return pd.read_csv(filepath)


# ==============================================================================
# STEP 2: GESTIONE ZERI FITTIZI
# ==============================================================================

def replace_zeros_with_nan(df: pd.DataFrame, 
                           columns: Optional[List[str]] = None,
                           verbose: bool = False) -> pd.DataFrame:
    """
    Sostituisce i valori 0 con NaN nelle colonne specificate.
    
    PROBLEMA: Alcune colonne contengono zeri che non hanno senso fisico.
    Esempio: un telefono non può pesare 0 grammi o avere 0 MP di fotocamera.
    Questi zeri rappresentano DATI MANCANTI MASCHERATI.
    
    SOLUZIONE: Convertiamo 0 → NaN così pandas li tratta come missing values
    e possiamo imputarli correttamente nel prossimo step.
    
    Args:
        df: DataFrame originale
        columns: Lista colonne da processare. Se None, usa COLS_WITH_HIDDEN_MISSING
        verbose: Se True, stampa quanti zeri sono stati convertiti
        
    Returns:
        DataFrame con zeri convertiti in NaN
    """
    df = df.copy()
    
    if columns is None:
        columns = COLS_WITH_HIDDEN_MISSING
    
    for col in columns:
        if col in df.columns:
            # Conta quanti zeri sono presenti prima della conversione
            zeros_count = (df[col] == 0).sum()
            if zeros_count > 0:
                df[col] = df[col].replace(0, np.nan)
                if verbose:
                    print(f"   {col}: {zeros_count} zeri → NaN")
                
    return df


# ==============================================================================
# STEP 3: IMPUTAZIONE MISSING VALUES
# ==============================================================================

def impute_missing_values(df: pd.DataFrame, 
                          group_columns: List[str] = ['device_brand', 'release_year'],
                          columns_to_fix: Optional[List[str]] = None,
                          verbose: bool = False) -> pd.DataFrame:
    """
    Imputa i valori mancanti usando la MEDIANA LOCALE del gruppo.
    
    STRATEGIA A DUE LIVELLI:
    
    1° LIVELLO - MEDIANA LOCALE (Brand + Anno):
       Raggruppiamo i dispositivi per marca e anno di uscita.
       I valori mancanti vengono riempiti con la mediana del gruppo.
       
       ESEMPIO: Un Samsung 2020 con RAM mancante viene imputato con la 
       mediana delle RAM degli altri Samsung del 2020.
       
       VANTAGGIO: Preserva la struttura intrinseca dei dati. Un telefono
       economico del 2018 non viene imputato con valori di flagship del 2020.
    
    2° LIVELLO - FALLBACK GLOBALE:
       Se un gruppo è troppo piccolo (es. un solo dispositivo di quel brand/anno),
       la mediana locale non funziona. In questo caso, usiamo la mediana globale.
    
    PERCHÉ NON KNN IMPUTER:
       KNN considera la distanza nello spazio delle feature, ma non tiene conto
       della semantica. La nostra strategia è "domain-driven" e più interpretabile.
    
    Args:
        df: DataFrame con NaN da imputare
        group_columns: Colonne per il raggruppamento (default: brand + anno)
        columns_to_fix: Colonne da correggere. Se None, usa lista predefinita
        verbose: Se True, stampa informazioni sul processo
        
    Returns:
        DataFrame con valori mancanti imputati
    """
    df = df.copy()
    
    if columns_to_fix is None:
        columns_to_fix = [
            'rear_camera_mp', 'front_camera_mp', 'internal_memory', 
            'ram', 'battery', 'weight'
        ]
    
    # --- PRIMA PASSATA: Mediana Locale (Brand + Anno) ---
    for col in columns_to_fix:
        if col in df.columns:
            before_nans = df[col].isnull().sum()
            df[col] = df[col].fillna(
                df.groupby(group_columns)[col].transform('median')
            )
            after_nans = df[col].isnull().sum()
            if verbose and before_nans > 0:
                filled = before_nans - after_nans
                print(f"   {col}: {filled}/{before_nans} valori imputati con mediana locale")
    
    # --- SECONDA PASSATA: Fallback con Mediana Globale ---
    for col in columns_to_fix:
        if col in df.columns:
            remaining_nans = df[col].isnull().sum()
            if remaining_nans > 0:
                global_median = df[col].median()
                df[col] = df[col].fillna(global_median)
                if verbose:
                    print(f"   {col}: {remaining_nans} valori rimanenti imputati con mediana globale ({global_median:.2f})")
                
    return df


# ==============================================================================
# STEP 4: RIMOZIONE OUTLIER (Tablet)
# ==============================================================================

def remove_outliers(df: pd.DataFrame,
                    max_weight: float = MAX_WEIGHT_THRESHOLD,
                    max_screen: float = MAX_SCREEN_THRESHOLD,
                    verbose: bool = False) -> pd.DataFrame:
    """
    Rimuove i dispositivi outlier basandosi su analisi bivariata Peso/Schermo.
    
    PROBLEMA: Il dataset include anche TABLET, non solo smartphone.
    I tablet hanno specifiche molto diverse (schermo grande, peso elevato)
    e introdurrebbero bias nel modello.
    
    SOLUZIONE - ANALISI BIVARIATA:
       Plottiamo Peso vs Dimensione Schermo e identifichiamo un cluster
       di dispositivi in alto a destra (pesanti + schermo grande).
       
       SOGLIE EMPIRICHE:
       - Peso > 350g → probabile tablet
       - Schermo > 20cm → probabile tablet
       
       NOTA: Anche i foldable più pesanti (Samsung Fold) pesano ~270g,
       quindi 350g è una soglia sicura.
    
    PERCHÉ NON IQR O Z-SCORE:
       Le soglie statistiche (IQR, z-score) non considerano la conoscenza
       del dominio. Un approccio domain-driven è più appropriato qui.
    
    Args:
        df: DataFrame da filtrare
        max_weight: Peso massimo in grammi (default: 350g)
        max_screen: Dimensione schermo massima in cm (default: 20cm)
        verbose: Se True, stampa quanti record sono stati rimossi
        
    Returns:
        DataFrame filtrato senza outlier
    """
    original_count = len(df)
    
    # Applica filtro bivariato: mantieni solo smartphone
    df_clean = df[
        (df['weight'] < max_weight) & 
        (df['screen_size'] < max_screen)
    ].copy()
    
    removed_count = original_count - len(df_clean)
    
    if verbose:
        print(f"   Dispositivi rimossi: {removed_count}")
        print(f"   (Peso > {max_weight}g oppure Schermo > {max_screen}cm)")
    
    return df_clean


# ==============================================================================
# PIPELINE COMPLETA DI PULIZIA
# ==============================================================================

def clean_dataset(filepath: str, 
                  verbose: bool = False, 
                  save_path: str = None) -> pd.DataFrame:
    """
    Pipeline completa di pulizia dati.
    
    Esegue in sequenza tutti gli step di preprocessing:
    1. Caricamento dati
    2. Conversione zeri fittizi → NaN
    3. Imputazione missing values (mediana locale + fallback globale)
    4. Rimozione outlier (tablet)
    5. Salvataggio opzionale del dataset processato
    
    Args:
        filepath: Percorso del file CSV grezzo
        verbose: Se True, stampa informazioni di debug per ogni step
        save_path: Se specificato, salva il dataset processato in questo percorso
        
    Returns:
        DataFrame pulito e pronto per il feature engineering
    """
    # --- STEP 1: Caricamento ---
    df = load_dataset(filepath)
    if verbose:
        print(f"\n📥 Dataset caricato: {len(df)} righe")
    
    # --- STEP 2: Conversione zeri → NaN ---
    if verbose:
        print("\n🔄 Conversione zeri fittizi → NaN:")
    df = replace_zeros_with_nan(df, verbose=verbose)
    
    if verbose:
        total_missing = df.isnull().sum().sum()
        print(f"\n   Totale valori mancanti: {total_missing}")
    
    # --- STEP 3: Imputazione ---
    if verbose:
        print("\n🩹 Imputazione missing values:")
    df = impute_missing_values(df, verbose=verbose)
    
    if verbose:
        remaining = df.isnull().sum().sum()
        print(f"\n   Valori mancanti rimanenti: {remaining}")
    
    # --- STEP 4: Rimozione Outlier ---
    if verbose:
        print("\n🗑️  Rimozione outlier (tablet):")
    df = remove_outliers(df, verbose=verbose)
    
    if verbose:
        print(f"\n✅ Dataset finale: {len(df)} righe")
    
    # --- STEP 5: Salvataggio opzionale ---
    if save_path:
        df.to_csv(save_path, index=False)
        if verbose:
            print(f"\n💾 Dataset processato salvato in: {save_path}")
    
    return df


# ==============================================================================
# FUNZIONI DI UTILITÀ
# ==============================================================================

def save_processed_dataset(df: pd.DataFrame, filepath: str) -> None:
    """
    Salva il dataset processato su disco in formato CSV.
    
    Args:
        df: DataFrame processato
        filepath: Percorso dove salvare il file
    """
    df.to_csv(filepath, index=False)


def load_processed_dataset(filepath: str) -> pd.DataFrame:
    """
    Carica un dataset già processato da disco.
    
    Utile per saltare la fase di preprocessing nelle esecuzioni successive.
    
    Args:
        filepath: Percorso del file CSV processato
        
    Returns:
        DataFrame processato pronto per feature engineering
    """
    return pd.read_csv(filepath)
