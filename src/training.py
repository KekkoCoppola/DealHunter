# -*- coding: utf-8 -*-
"""
================================================================================
training.py - Modulo per Addestramento e Valutazione dei Modelli
================================================================================

Questo modulo gestisce la terza fase del processo di Machine Learning:
l'addestramento dei modelli, la loro ottimizzazione e valutazione.

MODELLI IMPLEMENTATI:

1. LOGISTIC REGRESSION (Baseline):
   - Modello LINEARE per classificazione
   - Veloce da addestrare, interpretabile
   - Funziona bene se le relazioni sono lineari
   - Usato come BASELINE per confronto

2. RANDOM FOREST (Ensemble - Bagging):
   - Costruisce N alberi INDIPENDENTI in parallelo
   - Ogni albero vota per una classe, vince la maggioranza
   - Robusto all'overfitting, gestisce bene feature correlate
   - Fornisce FEATURE IMPORTANCE

3. GRADIENT BOOSTING (Ensemble - Boosting):
   - Costruisce alberi in modo SEQUENZIALE
   - Ogni albero corregge gli errori del precedente
   - Spesso più accurato di Random Forest
   - Più lento da addestrare

HYPERPARAMETER TUNING:
   Utilizziamo GridSearchCV per ottimizzare Random Forest:
   - Testa tutte le combinazioni di iperparametri
   - Usa Cross-Validation a 3 fold per robustezza
   - Seleziona la combinazione con migliore accuracy media

SCALING:
   StandardScaler normalizza le feature numeriche:
   - Formula: z = (x - media) / deviazione_standard
   - IMPORTANTE: fit solo su training, transform su entrambi
   - Evita DATA LEAKAGE: il test set non influenza lo scaler

USO:
    from training import split_data, scale_features, train_random_forest
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    model = train_random_forest(X_train_scaled, y_train)

================================================================================
"""

import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict, Any, List, Optional

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import (
    FEATURES_TO_SCALE,
    RANDOM_STATE,
    RF_PARAM_GRID,
    MODEL_PATH,
    SCALER_PATH,
    FEATURES_PATH
)


# ==============================================================================
# STEP 1: SPLIT TRAIN/TEST
# ==============================================================================

def split_data(X: pd.DataFrame, 
               y: pd.Series, 
               test_size: float = 0.2,
               random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide i dati in training e test set con STRATIFICAZIONE.
    
    SPLIT 80/20:
       - Training (80%): usato per addestrare il modello
       - Test (20%): usato per valutare le performance finali
       
    STRATIFICAZIONE (stratify=y):
       Mantiene la stessa proporzione di classi in train e test.
       Se il dataset ha 25% Budget, 25% Mid-Range, ecc.,
       anche train e test avranno le stesse proporzioni.
    
    RANDOM STATE:
       Seed fisso per riproducibilità. Con random_state=42,
       lo stesso split viene generato ogni volta.
    
    Args:
        X: DataFrame delle features
        y: Series del target
        test_size: Proporzione del test set (default: 20%)
        random_state: Seed per riproducibilità
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y  # Mantiene la distribuzione delle classi
    )


# ==============================================================================
# STEP 2: SCALING DELLE FEATURE
# ==============================================================================

def scale_features(X_train: pd.DataFrame, 
                   X_test: pd.DataFrame,
                   columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scala le feature numeriche usando StandardScaler (z-score normalization).
    
    PROBLEMA:
       Le feature hanno scale diverse:
       - battery: 1000-5000 mAh
       - ram: 1-12 GB
       
       Senza scaling, le feature con valori grandi dominano quelle piccole.
    
    SOLUZIONE - STANDARD SCALER:
       Trasforma ogni feature per avere media=0 e deviazione_standard=1
       Formula: z = (x - media) / std
       
       ESEMPIO (battery):
       - Originale: 4000 mAh
       - Media batterie: 3500 mAh, Std: 800 mAh
       - Scalato: (4000 - 3500) / 800 = 0.625
    
    ATTENZIONE - PREVENZIONE DATA LEAKAGE:
       Lo scaler viene addestrato (fit) SOLO sul training set.
       Il test set viene solo trasformato (transform), non influenza i parametri.
       
       PERCHÉ: Se usiamo anche il test per calcolare media/std,
       stiamo "sbirciando" nel futuro e i risultati saranno ottimistici.
    
    Args:
        X_train: Training features
        X_test: Test features
        columns: Colonne da scalare. Se None, usa FEATURES_TO_SCALE
        
    Returns:
        Tuple (X_train_scaled, X_test_scaled, scaler)
    """
    if columns is None:
        columns = FEATURES_TO_SCALE
    
    # Filtra solo le colonne effettivamente presenti
    cols_to_scale = [c for c in columns if c in X_train.columns]
    
    scaler = StandardScaler()
    
    # Creiamo copie per non modificare i DataFrame originali
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # FIT solo su training, TRANSFORM su entrambi
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    return X_train_scaled, X_test_scaled, scaler


# ==============================================================================
# STEP 3A: TRAINING LOGISTIC REGRESSION (Baseline)
# ==============================================================================

def train_logistic_regression(X_train: pd.DataFrame, 
                               y_train: pd.Series,
                               max_iter: int = 1000,
                               random_state: int = RANDOM_STATE) -> LogisticRegression:
    """
    Addestra un modello Logistic Regression.
    
    RUOLO: BASELINE
       La Logistic Regression è un modello LINEARE.
       La usiamo come punto di riferimento per valutare se i modelli
       più complessi (RF, GB) offrono un reale miglioramento.
    
    COME FUNZIONA:
       Calcola una combinazione lineare delle feature:
       score = w1*feature1 + w2*feature2 + ... + bias
       
       Poi usa la funzione sigmoid per convertire in probabilità:
       P(classe) = 1 / (1 + e^(-score))
    
    INTERPRETAZIONE RISULTATI:
       Se LR performa simile a RF e GB, significa che le relazioni
       tra feature e target sono prevalentemente LINEARI.
       → Preferire il modello più semplice (Rasoio di Occam)
    
    Args:
        X_train: Training features (già scalate)
        y_train: Training target
        max_iter: Iterazioni massime per convergenza
        random_state: Seed per riproducibilità
        
    Returns:
        Modello Logistic Regression addestrato
    """
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model


# ==============================================================================
# STEP 3B: TRAINING RANDOM FOREST
# ==============================================================================

def train_random_forest(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        n_estimators: int = 100,
                        random_state: int = RANDOM_STATE,
                        **kwargs) -> RandomForestClassifier:
    """
    Addestra un modello Random Forest.
    
    COME FUNZIONA (BAGGING):
       1. Crea N alberi decisionali INDIPENDENTI
       2. Ogni albero è addestrato su un campione RANDOM del dataset
       3. Ogni albero vota per una classe
       4. La classe con più voti vince (majority voting)
    
    VANTAGGI:
       - Robusto all'OVERFITTING (media di molti alberi)
       - Gestisce bene feature CORRELATE (ogni albero vede feature diverse)
       - Fornisce FEATURE IMPORTANCE (quanto ogni feature contribuisce)
    
    PARAMETRI PRINCIPALI:
       - n_estimators: Numero di alberi (più = meglio ma più lento)
       - max_depth: Profondità massima (limita per evitare overfitting)
    
    Args:
        X_train: Training features (già scalate)
        y_train: Training target
        n_estimators: Numero di alberi
        random_state: Seed per riproducibilità
        **kwargs: Altri parametri per RandomForestClassifier
        
    Returns:
        Modello Random Forest addestrato
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


# ==============================================================================
# STEP 3C: TRAINING GRADIENT BOOSTING
# ==============================================================================

def train_gradient_boosting(X_train: pd.DataFrame,
                            y_train: pd.Series,
                            n_estimators: int = 100,
                            max_depth: int = 5,
                            learning_rate: float = 0.1,
                            random_state: int = RANDOM_STATE,
                            **kwargs) -> GradientBoostingClassifier:
    """
    Addestra un modello Gradient Boosting.
    
    COME FUNZIONA (BOOSTING):
       1. Crea un primo albero "debole"
       2. Calcola gli ERRORI di questo albero
       3. Il secondo albero cerca di correggere questi errori
       4. Ripete fino a N alberi
       
       Ogni albero migliora il precedente → apprendimento SEQUENZIALE
    
    DIFFERENZA DA RANDOM FOREST:
       - RF: alberi indipendenti, voting democratico
       - GB: alberi sequenziali, correzione errori
       
       GB spesso è PIÙ ACCURATO ma anche PIÙ LENTO.
    
    PARAMETRI PRINCIPALI:
       - learning_rate: Quanto ogni albero contribuisce (0.1 = 10%)
                       Basso = più robusto, ma serve più alberi
       - max_depth: Limitato (5) per evitare overfitting
    
    Args:
        X_train: Training features (già scalate)
        y_train: Training target
        n_estimators: Numero di alberi
        max_depth: Profondità massima per albero
        learning_rate: Velocità di apprendimento
        random_state: Seed per riproducibilità
        **kwargs: Altri parametri
        
    Returns:
        Modello Gradient Boosting addestrato
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


# ==============================================================================
# STEP 4: HYPERPARAMETER TUNING (GridSearchCV)
# ==============================================================================

def tune_random_forest(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       param_grid: Optional[Dict] = None,
                       cv: int = 3,
                       random_state: int = RANDOM_STATE,
                       verbose: int = 1) -> Tuple[RandomForestClassifier, Dict]:
    """
    Ottimizza gli iperparametri del Random Forest con GridSearchCV.
    
    COSA FA GRIDSEARCH:
       1. Definisce una GRIGLIA di combinazioni di parametri
       2. Addestra un modello per OGNI combinazione
       3. Valuta ogni modello con CROSS-VALIDATION
       4. Seleziona la combinazione con migliore accuracy media
    
    CROSS-VALIDATION (cv=3):
       Divide il training set in 3 parti (fold).
       Per ogni fold:
       - Usa 2 parti per training
       - Usa 1 parte per validazione
       Ripete 3 volte e fa la media → stima più robusta.
    
    GRIGLIA DI DEFAULT:
       n_estimators: [100, 200] → quanti alberi
       max_depth: [10, 20, None] → profondità massima
       min_samples_split: [2, 5, 10] → minimo campioni per split
       class_weight: ['balanced', None] → gestione classi sbilanciate
    
    Args:
        X_train: Training features (già scalate)
        y_train: Training target
        param_grid: Griglia di parametri. Se None, usa RF_PARAM_GRID
        cv: Numero di fold per cross-validation
        random_state: Seed per riproducibilità
        verbose: Livello di verbosità (0=silenzioso, 1=progresso)
        
    Returns:
        Tuple (miglior_modello, migliori_parametri)
    """
    if param_grid is None:
        param_grid = RF_PARAM_GRID
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=random_state),
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,  # Usa tutti i processori disponibili
        verbose=verbose
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_


# ==============================================================================
# STEP 5: VALUTAZIONE DEL MODELLO
# ==============================================================================

def evaluate_model(model, 
                   X_test: pd.DataFrame, 
                   y_test: pd.Series,
                   labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Valuta le performance di un modello sul test set.
    
    METRICHE CALCOLATE:
    
    1. ACCURACY: % di predizioni corrette
       accuracy = corrette / totali
       
    2. CONFUSION MATRIX: matrice NxN
       - Diagonale: predizioni corrette
       - Fuori diagonale: errori
       Mostra DOVE il modello sbaglia.
       
    3. CLASSIFICATION REPORT:
       Per ogni classe calcola:
       - Precision: tra quelli predetti come X, quanti sono veramente X?
       - Recall: tra tutti gli X veri, quanti sono stati trovati?
       - F1-score: media armonica di precision e recall
    
    Args:
        model: Modello addestrato
        X_test: Test features (già scalate)
        y_test: Test target
        labels: Ordine delle classi per confusion matrix
        
    Returns:
        Dizionario con accuracy, predictions, confusion_matrix, report
    """
    # Genera predizioni
    y_pred = model.predict(X_test)
    
    if labels is None:
        labels = ['Budget', 'Mid-Range', 'High-End', 'Premium']
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'predictions': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=labels),
        'classification_report': classification_report(y_test, y_pred)
    }


# ==============================================================================
# STEP 6: FEATURE IMPORTANCE
# ==============================================================================

def get_feature_importance(model: RandomForestClassifier,
                           feature_names: pd.Index) -> pd.DataFrame:
    """
    Estrae l'importanza delle feature dal Random Forest.
    
    COME FUNZIONA:
       Random Forest calcola quanto ogni feature contribuisce 
       alla riduzione dell'impurità (Gini) negli split degli alberi.
       
       Feature con alta importanza → usate spesso negli split → influenti
       Feature con bassa importanza → poco usate → meno rilevanti
    
    INTERPRETAZIONE RISULTATI:
       - front_camera_mp: 13% → molto influente
       - ram: 3% → poco influente (probabilmente correlata a internal_memory)
    
    NOTA SULLA CORRELAZIONE:
       Se due feature sono correlate (es. RAM e Memory), il modello
       ne sceglie una e "ignora" l'altra. Questo spiega perché alcune
       feature tecnicamente importanti hanno bassa feature importance.
    
    Args:
        model: Random Forest addestrato
        feature_names: Nomi delle colonne (X.columns)
        
    Returns:
        DataFrame ordinato per importanza decrescente
    """
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    
    return importance_df.sort_values(by='Importance', ascending=False)


# ==============================================================================
# STEP 7: SALVATAGGIO E CARICAMENTO ARTEFATTI
# ==============================================================================

def save_artifacts(model, 
                   scaler: StandardScaler, 
                   feature_names: pd.Index,
                   model_path: str = MODEL_PATH,
                   scaler_path: str = SCALER_PATH,
                   features_path: str = FEATURES_PATH) -> None:
    """
    Salva modello, scaler e feature names su disco per uso in produzione.
    
    ARTEFATTI SALVATI:
    
    1. MODELLO (.pkl):
       Il Random Forest addestrato, pronto per fare predizioni.
       
    2. SCALER (.pkl):
       Lo StandardScaler con i parametri (media, std) del training.
       FONDAMENTALE: per nuovi dati, usare lo STESSO scaler!
       
    3. FEATURE NAMES (.pkl):
       Lista ordinata delle colonne. Garantisce che i nuovi dati
       abbiano le stesse colonne nello stesso ordine.
    
    Args:
        model: Modello addestrato
        scaler: StandardScaler configurato
        feature_names: Nomi delle colonne
        model_path: Path per il modello
        scaler_path: Path per lo scaler
        features_path: Path per i nomi delle feature
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, features_path)


def load_artifacts(model_path: str = MODEL_PATH,
                   scaler_path: str = SCALER_PATH,
                   features_path: str = FEATURES_PATH) -> Tuple[Any, StandardScaler, pd.Index]:
    """
    Carica modello, scaler e feature names da disco.
    
    Usato dall'applicazione Streamlit per fare predizioni
    senza dover riaddestrare il modello ogni volta.
    
    Args:
        model_path: Path del modello
        scaler_path: Path dello scaler
        features_path: Path dei nomi delle feature
        
    Returns:
        Tuple (model, scaler, feature_names)
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    
    return model, scaler, feature_names
