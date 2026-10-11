"""
Módulo de Modelagem Preditiva e Avaliação
Projeto: O Desafio do Oráculo Preditivo | Health Score
FATEC Indaiatuba - Prof. Medina
Autores: Gustavo Bravo, Gustavo Martins, Lucas, Thiago Bastos
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


@st.cache_resource
def train_and_evaluate_models(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    X_test_scaled: np.ndarray,
    y_test: pd.Series,
    feature_names: list[str],
) -> dict:
    """
    Treina os 4 modelos requisitados:
    1. Dummy Regressor (strategy='mean') [Baseline]
    2. Regressão Linear Múltipla
    3. Random Forest Regressor (random_state=42)
    4. Support Vector Regressor (SVR - kernel='rbf', C=100, epsilon=0.1)

    Calcula métricas no conjunto de Teste: R², Cross-Validation 5-Fold (CV R²),
    MAE, MAPE (%), RMSE e Tempo de Treinamento (s).
    Extrai a Importância das Features e Coeficientes Beta Padronizados.

    Retorna um dicionário contendo:
    - 'models': instâncias treinadas
    - 'metrics_df': DataFrame com comparativo ordenado por R² Teste
    - 'predictions': dict de predições no teste
    - 'predictions_train': dict de predições no treino
    - 'overfitting_df': diagnóstico R² Treino vs R² Teste
    - 'feature_importance_df': importâncias RF e betas da Regressão Linear
    """
    models_to_train = {
        "Dummy Regressor": DummyRegressor(strategy="mean"),
        "Regressão Linear Múltipla": LinearRegression(),
        "Support Vector Regressor (SVR)": SVR(kernel="rbf", C=100, epsilon=0.1),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    trained_models = {}
    test_predictions = {}
    train_predictions = {}
    metrics_list = []
    overfitting_list = []

    for name, model in models_to_train.items():
        # Medição de tempo de treinamento
        start_time = time.perf_counter()
        model.fit(X_train_scaled, y_train)
        end_time = time.perf_counter()
        tempo_treino = end_time - start_time

        # Predições
        y_pred_test = model.predict(X_test_scaled)
        y_pred_train = model.predict(X_train_scaled)

        trained_models[name] = model
        test_predictions[name] = y_pred_test
        train_predictions[name] = y_pred_train

        # Validação cruzada (5-Fold) no treino
        if name == "Dummy Regressor":
            cv_r2 = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2").mean()
        else:
            cv_r2 = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2").mean()

        # Métricas no teste
        r2_test = r2_score(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        mae = mean_absolute_error(y_test, y_pred_test)
        mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        metrics_list.append({
            "Modelo": name,
            "R² (Teste)": round(r2_test, 4),
            "R² (CV 5-Fold)": round(cv_r2, 4),
            "MAE": round(mae, 2),
            "MAPE (%)": round(mape, 2),
            "RMSE": round(rmse, 2),
            "Tempo de Treino (s)": round(tempo_treino, 4),
        })

        gap_overfitting = r2_train - r2_test
        overfitting_list.append({
            "Modelo": name,
            "R² Treino": round(r2_train, 4),
            "R² Teste": round(r2_test, 4),
            "Gap (Treino - Teste)": round(gap_overfitting, 4),
            "Status": (
                "Excelente Generalização"
                if gap_overfitting < 0.05 and r2_test > 0.5
                else "Sobreajuste / Overfitting Parcial"
                if gap_overfitting >= 0.10
                else "Baseline Sem Ajuste"
                if r2_test < 0.05
                else "Ajuste Adequado"
            ),
        })

    # DataFrames estruturados
    metrics_df = pd.DataFrame(metrics_list).sort_values(by="R² (Teste)", ascending=False).reset_index(drop=True)
    overfitting_df = pd.DataFrame(overfitting_list).sort_values(by="R² Teste", ascending=False).reset_index(drop=True)

    # Extração de Importância de Features (RF) e Betas Padronizados (RLM)
    rf_model = trained_models["Random Forest Regressor"]
    lr_model = trained_models["Regressão Linear Múltipla"]

    rf_importances = rf_model.feature_importances_ * 100
    lr_betas = lr_model.coef_

    feature_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importância RF (%)": np.round(rf_importances, 2),
        "Coeficiente Beta (RLM)": np.round(lr_betas, 4),
        "Impacto Clínico": [
            "Positivo (+ Saúde)" if b > 0 else "Negativo (- Saúde)" for b in lr_betas
        ],
    }).sort_values(by="Importância RF (%)", ascending=False).reset_index(drop=True)

    return {
        "models": trained_models,
        "metrics_df": metrics_df,
        "overfitting_df": overfitting_df,
        "predictions": test_predictions,
        "predictions_train": train_predictions,
        "feature_importance_df": feature_imp_df,
    }


def predict_single_patient(
    model,
    scaler,
    patient_data: dict,
    feature_names: list[str]
) -> float:
    """
    Executa a predição para um novo paciente baseado nos hábitos fornecidos.
    Aplica o escalonamento StandardScaler prévio.
    """
    input_df = pd.DataFrame([patient_data])[feature_names]
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    return float(np.clip(pred, 0.0, 100.0))
