# -*- coding: utf-8 -*-
"""
================================================================================
pipeline.py - Pipeline Completa di Training per DealHunter
================================================================================

FLUSSO DELLA PIPELINE (8 step):
    1. PREPROCESSING: Carica dati, gestisce missing values, rimuove outlier
    2. FEATURE ENGINEERING: Crea categorie prezzo, model_age, one-hot encoding
    3. EDA: Analisi correlazione (identifica feature ridondanti)
    4. SPLIT & SCALING: Divide train/test (80/20), applica StandardScaler
    5. TRAINING: Confronta 3 modelli (LR, RF, GB)
    6. TUNING: Ottimizza RF per verificare se può battere LR
    7. VALUTAZIONE: Sceglie il modello migliore (tipicamente LR)
    8. SALVATAGGIO: Salva il modello vincitore su disco

NOTA IMPORTANTE:
    Logistic Regression tipicamente vince perché le relazioni nel dataset
    sono prevalentemente LINEARI. Manteniamo il tuning RF per dimostrare
    che il processo è stato rigoroso (Rasoio di Occam).

================================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASET_PATH, MODEL_PATH, SCALER_PATH, FEATURES_PATH, PROCESSED_DATASET_PATH
from preprocessing import clean_dataset
from feature_engineering import engineer_features, prepare_features
from training import (
    split_data,
    scale_features,
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting,
    tune_random_forest,
    evaluate_model,
    get_feature_importance,
    save_artifacts
)
from analytics import (
    plot_correlation_heatmap,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    print_correlation_analysis
)


def run_full_pipeline(data_path: str = DATASET_PATH,
                       save_model: bool = True,
                       save_processed: bool = True,
                       show_plots: bool = False,
                       verbose: bool = True) -> dict:
    """
    Esegue la pipeline completa di training.
    
    Il modello finale salvato sarà quello con accuracy più alta.
    Tipicamente Logistic Regression vince perché le relazioni sono lineari.
    """
    results = {}
    
    # ==========================================================================
    # STEP 1: PREPROCESSING
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 1: PREPROCESSING")
        print("=" * 60)
    
    processed_path = PROCESSED_DATASET_PATH if save_processed else None
    df = clean_dataset(data_path, verbose=verbose, save_path=processed_path)
    results['rows_after_cleaning'] = len(df)
    
    # ==========================================================================
    # STEP 2: FEATURE ENGINEERING
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 2: FEATURE ENGINEERING")
        print("=" * 60)
    
    df_engineered = engineer_features(df, verbose=verbose)
    X, y = prepare_features(df_engineered)
    results['n_features'] = X.shape[1]
    
    # ==========================================================================
    # STEP 3: EDA (Analisi Esplorativa)
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 3: EDA (Analisi Correlazione)")
        print("=" * 60)
    
    corr_matrix = plot_correlation_heatmap(X, save_path=None, show=show_plots)
    if verbose:
        print_correlation_analysis(corr_matrix, threshold=0.5)
    results['correlation_matrix'] = corr_matrix
    
    # ==========================================================================
    # STEP 4: SPLIT & SCALING
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 4: SPLIT & SCALING")
        print("=" * 60)
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    if verbose:
        print(f"\n📚 Training: {len(X_train)} | Test: {len(X_test)}")
    
    # ==========================================================================
    # STEP 5: TRAINING - CONFRONTO 3 MODELLI
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 5: TRAINING - CONFRONTO MODELLI")
        print("=" * 60)
    
    # Logistic Regression
    if verbose:
        print("\n🔹 Training Logistic Regression...")
    log_model = train_logistic_regression(X_train_scaled, y_train)
    log_metrics = evaluate_model(log_model, X_test_scaled, y_test)
    
    # Random Forest
    if verbose:
        print("🔹 Training Random Forest...")
    rf_model = train_random_forest(X_train_scaled, y_train)
    rf_metrics = evaluate_model(rf_model, X_test_scaled, y_test)
    
    # Gradient Boosting
    if verbose:
        print("🔹 Training Gradient Boosting...")
    gb_model = train_gradient_boosting(X_train_scaled, y_train)
    gb_metrics = evaluate_model(gb_model, X_test_scaled, y_test)
    
    # Salva risultati
    results['logistic_regression_accuracy'] = log_metrics['accuracy']
    results['random_forest_accuracy'] = rf_metrics['accuracy']
    results['gradient_boosting_accuracy'] = gb_metrics['accuracy']
    
    if verbose:
        print("\n" + "-" * 60)
        print("        RISULTATI CONFRONTO")
        print("-" * 60)
        print(f"1. Logistic Regression:    {log_metrics['accuracy']:.2%}")
        print(f"2. Random Forest:          {rf_metrics['accuracy']:.2%}")
        print(f"3. Gradient Boosting:      {gb_metrics['accuracy']:.2%}")
        print("-" * 60)
    
    # ==========================================================================
    # STEP 6: TUNING RF (per dimostrare metodologia)
    # Anche se LR vince, proviamo a ottimizzare RF per vedere se può migliorare
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 6: TUNING RF (verifica metodologica)")
        print("=" * 60)
        print("⏳ Ottimizzazione RF con GridSearchCV...\n")
    
    best_rf, best_params = tune_random_forest(
        X_train_scaled, y_train, 
        verbose=1 if verbose else 0
    )
    tuned_rf_metrics = evaluate_model(best_rf, X_test_scaled, y_test)
    results['tuned_rf_accuracy'] = tuned_rf_metrics['accuracy']
    results['best_rf_params'] = best_params
    
    if verbose:
        print(f"\n📊 RF Ottimizzato: {tuned_rf_metrics['accuracy']:.2%}")
        print(f"   Miglioramento: {(tuned_rf_metrics['accuracy'] - rf_metrics['accuracy'])*100:+.2f}%")
    
    # ==========================================================================
    # STEP 7: SELEZIONE MODELLO FINALE
    # Scegliamo il modello con accuracy più alta
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 7: SELEZIONE MODELLO FINALE")
        print("=" * 60)
    
    # Confronto finale (include RF ottimizzato)
    all_models = {
        'Logistic Regression': (log_model, log_metrics['accuracy']),
        'Random Forest Base': (rf_model, rf_metrics['accuracy']),
        'Random Forest Tuned': (best_rf, tuned_rf_metrics['accuracy']),
        'Gradient Boosting': (gb_model, gb_metrics['accuracy'])
    }
    
    # Trova il migliore
    best_name = max(all_models, key=lambda k: all_models[k][1])
    best_model, best_accuracy = all_models[best_name]
    
    results['best_model_name'] = best_name
    results['best_accuracy'] = best_accuracy
    
    if verbose:
        print(f"\n🏆 MODELLO VINCITORE: {best_name}")
        print(f"   Accuracy: {best_accuracy:.2%}")
        
        if 'Logistic' in best_name:
            print("\n📝 INTERPRETAZIONE:")
            print("   Le relazioni tra feature e prezzo sono LINEARI.")
            print("   Un modello semplice è preferibile (Rasoio di Occam).")
    
    # Valutazione finale del modello vincitore
    final_metrics = evaluate_model(best_model, X_test_scaled, y_test)
    results['confusion_matrix'] = final_metrics['confusion_matrix']
    results['classification_report'] = final_metrics['classification_report']
    
    if verbose:
        print(f"\n{final_metrics['classification_report']}")
    
    if show_plots:
        plot_confusion_matrix(y_test, final_metrics['predictions'])
    
    # Feature Importance (solo per modelli tree-based)
    if hasattr(best_model, 'feature_importances_'):
        importance = get_feature_importance(best_model, X.columns)
        results['top_features'] = importance.head(10).to_dict('records')
        if verbose:
            print("\n🔝 TOP 10 FEATURE:")
            print(importance.head(10).to_string(index=False))
        if show_plots:
            plot_feature_importance(importance)
    else:
        # Per Logistic Regression usiamo i coefficienti
        if verbose:
            print("\n🔝 TOP 10 COEFFICIENTI (valore assoluto):")
        coef_importance = get_logistic_coefficients(best_model, X.columns)
        results['top_features'] = coef_importance.head(10).to_dict('records')
        if verbose:
            print(coef_importance.head(10).to_string(index=False))
    
    # ==========================================================================
    # STEP 8: SALVATAGGIO
    # ==========================================================================
    if save_model:
        if verbose:
            print("\n" + "=" * 60)
            print("  STEP 8: SALVATAGGIO")
            print("=" * 60)
        
        save_artifacts(best_model, scaler, X.columns)
        
        if verbose:
            print(f"\n✅ Modello ({best_name}): {MODEL_PATH}")
            print(f"✅ Scaler: {SCALER_PATH}")
            print(f"✅ Features: {FEATURES_PATH}")
    
    return results


def get_logistic_coefficients(model, feature_names):
    """
    Estrae i coefficienti della Logistic Regression come importanza.
    
    Per LR multiclasse, prendiamo il valore assoluto medio dei coefficienti
    su tutte le classi.
    """
    import pandas as pd
    import numpy as np
    
    # model.coef_ ha shape (n_classes, n_features)
    # Prendiamo la media dei valori assoluti su tutte le classi
    mean_abs_coef = np.mean(np.abs(model.coef_), axis=0)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_coef
    })
    
    return importance_df.sort_values(by='Importance', ascending=False)


# ==============================================================================
# ESECUZIONE STANDALONE
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("      🚀 DealHunter - Training Pipeline")
    print("=" * 60)
    
    results = run_full_pipeline(
        save_model=True,
        save_processed=True,
        show_plots=False,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("      📊 RIEPILOGO FINALE")
    print("=" * 60)
    print(f"  Righe processate:           {results['rows_after_cleaning']}")
    print(f"  Numero features:            {results['n_features']}")
    print(f"  Logistic Regression:        {results['logistic_regression_accuracy']:.2%}")
    print(f"  Random Forest Base:         {results['random_forest_accuracy']:.2%}")
    print(f"  Random Forest Tuned:        {results['tuned_rf_accuracy']:.2%}")
    print(f"  Gradient Boosting:          {results['gradient_boosting_accuracy']:.2%}")
    print("-" * 60)
    print(f"  🏆 VINCITORE: {results['best_model_name']} ({results['best_accuracy']:.2%})")
    print("=" * 60)
    print("\n✅ Pipeline completata!\n")
