# -*- coding: utf-8 -*-
"""
================================================================================
analytics.py - Modulo per Visualizzazione e Analisi Esplorativa
================================================================================

Questo modulo contiene tutte le funzioni per creare grafici e analisi visive.
È pensato per essere usato sia nella pipeline che nel notebook.

FUNZIONALITÀ PRINCIPALI:
    1. Analisi Outlier: grafico scatter peso vs schermo
    2. Distribuzione Classi: grafico a barre delle categorie target
    3. Correlazione Feature: heatmap per identificare ridondanze
    4. Confusion Matrix: valutazione errori del modello
    5. Feature Importance: grafico delle feature più influenti

USO:
    from analytics import plot_correlation_heatmap, plot_confusion_matrix
    
    # Per visualizzare la correlazione
    plot_correlation_heatmap(X, save_path='grafici/correlazione.png')
    
    # Per visualizzare la confusion matrix
    plot_confusion_matrix(y_test, y_pred, save_path='grafici/cm.png')

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple


# ==============================================================================
# 1. ANALISI OUTLIER (Grafico Scatter Peso vs Schermo)
# ==============================================================================

def plot_outlier_analysis(df: pd.DataFrame,
                          weight_threshold: float = 350,
                          screen_threshold: float = 20,
                          save_path: Optional[str] = None,
                          show: bool = True) -> None:
    """
    Crea un grafico scatter per identificare outlier (tablet) basato su peso e schermo.
    
    Questo grafico è fondamentale per giustificare la rimozione dei tablet dal dataset.
    I dispositivi oltre la soglia (linea rossa) vengono rimossi.
    
    Args:
        df: DataFrame con colonne 'weight' e 'screen_size'
        weight_threshold: Soglia peso in grammi (default: 350g)
        screen_threshold: Soglia schermo in cm (default: 20cm)
        save_path: Se specificato, salva il grafico in questo percorso
        show: Se True, mostra il grafico a schermo
    """
    # --- Creazione del grafico ---
    plt.figure(figsize=(10, 6))
    
    # Scatter plot colorato per brand
    sns.scatterplot(
        data=df, 
        x='weight', 
        y='screen_size', 
        hue='device_brand', 
        alpha=0.6, 
        legend=False
    )
    
    # Linee di soglia per identificare tablet
    plt.axvline(x=weight_threshold, color='red', linestyle='--', 
                label=f'Soglia Peso ({weight_threshold}g)')
    plt.axhline(y=screen_threshold, color='orange', linestyle='--', 
                label=f'Soglia Schermo ({screen_threshold}cm)')
    
    # Etichette e titolo
    plt.title('Analisi Bivariata: Identificazione Tablet (Outlier)', fontsize=14)
    plt.xlabel('Peso (grammi)', fontsize=12)
    plt.ylabel('Dimensione Schermo (cm)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    # Salvataggio opzionale
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Grafico outlier salvato in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# 2. DISTRIBUZIONE CLASSI TARGET
# ==============================================================================

def plot_class_distribution(df: pd.DataFrame,
                            target_column: str = 'price_category',
                            save_path: Optional[str] = None,
                            show: bool = True) -> None:
    """
    Visualizza la distribuzione delle classi target (fasce di prezzo).
    
    Questo grafico dimostra che il dataset è BILANCIATO grazie all'uso di qcut.
    Ogni classe dovrebbe avere circa lo stesso numero di elementi.
    
    Args:
        df: DataFrame con la colonna target
        target_column: Nome della colonna target (default: 'price_category')
        save_path: Se specificato, salva il grafico
        show: Se True, mostra il grafico
    """
    plt.figure(figsize=(8, 5))
    
    # Grafico a barre con palette viridis
    order = ['Budget', 'Mid-Range', 'High-End', 'Premium']
    sns.countplot(data=df, x=target_column, palette='viridis', order=order)
    
    # Etichette
    plt.title('Distribuzione delle Fasce di Prezzo (Target)', fontsize=14)
    plt.xlabel('Fascia di Prezzo', fontsize=12)
    plt.ylabel('Numero di Dispositivi', fontsize=12)
    
    # Aggiungi etichette sopra le barre
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Grafico distribuzione classi salvato in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# 3. HEATMAP CORRELAZIONE (Analisi Feature Ridondanti)
# ==============================================================================

def plot_correlation_heatmap(X: pd.DataFrame,
                              numeric_features: Optional[List[str]] = None,
                              save_path: Optional[str] = None,
                              show: bool = True) -> pd.DataFrame:
    """
    Crea una heatmap delle correlazioni tra feature numeriche.
    
    Questa analisi identifica feature RIDONDANTI (correlazione > 0.5).
    Ad esempio, RAM e Internal Memory sono correlate perché telefoni con
    più storage tendono ad avere più RAM.
    
    Args:
        X: DataFrame delle features
        numeric_features: Lista di feature numeriche da analizzare. 
                         Se None, usa le feature predefinite
        save_path: Se specificato, salva il grafico
        show: Se True, mostra il grafico
        
    Returns:
        Matrice di correlazione come DataFrame
    """
    # Feature numeriche di default
    if numeric_features is None:
        numeric_features = [
            'screen_size', 'rear_camera_mp', 'front_camera_mp',
            'internal_memory', 'ram', 'battery', 'weight', 'days_used', 'model_age'
        ]
    
    # Filtra solo le colonne presenti
    available_features = [f for f in numeric_features if f in X.columns]
    
    # Calcola la matrice di correlazione
    corr_matrix = X[available_features].corr()
    
    # --- Creazione Heatmap ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True,        # Mostra i valori nelle celle
        fmt='.2f',         # Due decimali
        cmap='coolwarm',   # Colori caldo-freddo
        center=0,          # Centra la scala sul valore 0
        square=True,       # Celle quadrate
        linewidths=0.5     # Bordi tra le celle
    )
    
    plt.title('Matrice di Correlazione - Feature Numeriche', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Heatmap correlazione salvata in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return corr_matrix


def get_high_correlations(corr_matrix: pd.DataFrame, 
                          threshold: float = 0.5) -> List[Tuple[str, str, float]]:
    """
    Estrae le coppie di feature con correlazione superiore alla soglia.
    
    Utile per identificare feature ridondanti che il modello potrebbe
    "ignorare" durante il training.
    
    Args:
        corr_matrix: Matrice di correlazione
        threshold: Soglia minima (default: 0.5)
        
    Returns:
        Lista di tuple (feature1, feature2, correlazione)
    """
    high_corr = []
    features = corr_matrix.columns.tolist()
    
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                high_corr.append((features[i], features[j], corr_val))
    
    return high_corr


# ==============================================================================
# 4. CONFUSION MATRIX (Analisi Errori del Modello)
# ==============================================================================

def plot_confusion_matrix(y_true: pd.Series,
                          y_pred: np.ndarray,
                          labels: Optional[List[str]] = None,
                          save_path: Optional[str] = None,
                          show: bool = True) -> np.ndarray:
    """
    Visualizza la Confusion Matrix del modello.
    
    La Confusion Matrix mostra DOVE il modello sbaglia:
    - Diagonale: predizioni corrette
    - Fuori diagonale: errori (es. Mid-Range predetto come High-End)
    
    Args:
        y_true: Etichette vere (ground truth)
        y_pred: Etichette predette dal modello
        labels: Ordine delle classi. Se None, usa ordine predefinito
        save_path: Se specificato, salva il grafico
        show: Se True, mostra il grafico
        
    Returns:
        Matrice di confusione come numpy array
    """
    from sklearn.metrics import confusion_matrix
    
    if labels is None:
        labels = ['Budget', 'Mid-Range', 'High-End', 'Premium']
    
    # Calcola la confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # --- Creazione Grafico ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True,           # Mostra i numeri
        fmt='d',              # Formato intero
        cmap='Blues',         # Palette blu
        xticklabels=labels,
        yticklabels=labels
    )
    
    plt.xlabel('Predizione del Modello', fontsize=12)
    plt.ylabel('Valore Reale', fontsize=12)
    plt.title('Confusion Matrix - Analisi degli Errori', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion Matrix salvata in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return cm


# ==============================================================================
# 5. FEATURE IMPORTANCE (Quali feature influenzano il prezzo?)
# ==============================================================================

def plot_feature_importance(importance_df: pd.DataFrame,
                             top_n: int = 10,
                             save_path: Optional[str] = None,
                             show: bool = True) -> None:
    """
    Visualizza le feature più importanti per il modello.
    
    Questo grafico spiega COSA guarda il modello per decidere il prezzo.
    Es: Fotocamera > RAM indica che i megapixel contano più della memoria RAM.
    
    Args:
        importance_df: DataFrame con colonne 'Feature' e 'Importance'
        top_n: Numero di feature da mostrare (default: 10)
        save_path: Se specificato, salva il grafico
        show: Se True, mostra il grafico
    """
    # Prendi le top N feature
    top_features = importance_df.head(top_n)
    
    # --- Creazione Grafico ---
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=top_features, 
        palette='magma'
    )
    
    plt.title(f'Top {top_n} Feature che Influenzano il Prezzo', fontsize=14)
    plt.xlabel('Importanza (0-1)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Grafico Feature Importance salvato in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# 6. CONFRONTO MODELLI (Grafico a Barre delle Accuracy)
# ==============================================================================

def plot_model_comparison(model_scores: dict,
                          save_path: Optional[str] = None,
                          show: bool = True) -> None:
    """
    Visualizza un confronto tra le accuracy di diversi modelli.
    
    Args:
        model_scores: Dizionario {nome_modello: accuracy}
                     Es: {'Logistic Regression': 0.65, 'Random Forest': 0.62}
        save_path: Se specificato, salva il grafico
        show: Se True, mostra il grafico
    """
    models = list(model_scores.keys())
    scores = list(model_scores.values())
    
    # Identifica il miglior modello
    best_idx = scores.index(max(scores))
    colors = ['#4CAF50' if i == best_idx else '#2196F3' for i in range(len(scores))]
    
    # --- Creazione Grafico ---
    plt.figure(figsize=(10, 5))
    bars = plt.bar(models, scores, color=colors, edgecolor='black')
    
    # Aggiungi etichette sopra le barre
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylim(0, 1)
    plt.title('Confronto Accuracy tra Modelli', fontsize=14)
    plt.xlabel('Modello', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.axhline(y=0.25, color='red', linestyle='--', label='Random Guess (25%)')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Grafico confronto modelli salvato in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# 7. REPORT TESTUALE DELLE ANALISI
# ==============================================================================

def print_correlation_analysis(corr_matrix: pd.DataFrame, threshold: float = 0.5) -> None:
    """
    Stampa un report testuale delle correlazioni significative.
    
    Utile per la documentazione e la presentazione al professore.
    
    Args:
        corr_matrix: Matrice di correlazione
        threshold: Soglia per considerare una correlazione "alta"
    """
    print("\n" + "=" * 60)
    print("      ANALISI DELLE CORRELAZIONI TRA FEATURE")
    print("=" * 60)
    
    high_corr = get_high_correlations(corr_matrix, threshold)
    
    if high_corr:
        print(f"\n📊 Correlazioni superiori a {threshold}:\n")
        for feat1, feat2, corr in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
            symbol = "🔴" if corr > 0.7 else "🟡"
            print(f"   {symbol} {feat1} ↔ {feat2}: {corr:.2f}")
        
        print("\n📝 Interpretazione:")
        print("   - Feature altamente correlate possono essere ridondanti")
        print("   - Il modello Random Forest gestisce automaticamente questa ridondanza")
        print("   - Spiega perché alcune feature hanno bassa Feature Importance")
    else:
        print(f"\n✅ Nessuna correlazione superiore a {threshold}")
    
    print("=" * 60)
