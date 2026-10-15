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
from sklearn.metrics.pairwise import pairwise_kernels


class RelevanceVectorRegressor:
    """
    Implementação de Relevance Vector Regression (RVR - Tipping, 2001)
    baseada em Sparse Bayesian Learning (Automatic Relevance Determination - ARD)
    sobre matriz de Kernel ('rbf', 'linear', 'poly').
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: float = 0.15,
        degree: int = 3,
        coef0: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 300,
        alpha_max: float = 1e5,
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.alpha_max = alpha_max

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.kernel == "rbf":
            return pairwise_kernels(X1, X2, metric="rbf", gamma=self.gamma)
        elif self.kernel == "poly":
            return pairwise_kernels(
                X1, X2, metric="poly", gamma=self.gamma, degree=self.degree, coef0=self.coef0
            )
        else:
            return pairwise_kernels(X1, X2, metric="linear")

    def fit(self, X: np.ndarray, y: pd.Series | np.ndarray):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n_samples = X_arr.shape[0]
        self.n_train_samples_ = n_samples

        # Matriz de projeto Phi = [1, K(X, X)] de dimensão (N, N + 1)
        K = self._compute_kernel(X_arr, X_arr)
        Phi = np.hstack([np.ones((n_samples, 1)), K])
        n_basis = Phi.shape[1]

        # Inicialização das precisões a priori (alpha) e ruído (beta)
        alpha = np.ones(n_basis, dtype=np.float64)
        var_y = max(float(np.var(y_arr)), 1e-4)
        beta = 1.0 / (0.1 * var_y)

        active_mask = np.ones(n_basis, dtype=bool)
        mu_active = np.zeros(n_basis, dtype=np.float64)
        self.converged_ = False
        self.n_iter_ = 0

        for it in range(1, self.max_iter + 1):
            self.n_iter_ = it
            Phi_a = Phi[:, active_mask]
            alpha_a = alpha[active_mask]
            m_active = Phi_a.shape[1]

            # Matriz de precisão posterior: A = diag(alpha_a) + beta * Phi_a^T * Phi_a
            A = np.diag(alpha_a) + beta * (Phi_a.T @ Phi_a)
            A += np.eye(m_active) * 1e-8  # Estabilidade numérica

            try:
                Sigma = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                Sigma = np.linalg.pinv(A)

            mu_a = beta * (Sigma @ (Phi_a.T @ y_arr))
            diag_Sigma = np.clip(np.diag(Sigma), 0.0, None)

            # Graus de determinação efetivos (gamma_i = 1 - alpha_i * Sigma_ii)
            gamma_eff = np.clip(1.0 - alpha_a * diag_Sigma, 1e-9, 1.0)

            # Atualização de MacKay para alpha e beta
            alpha_new = gamma_eff / np.maximum(mu_a ** 2, 1e-12)
            residuals = y_arr - (Phi_a @ mu_a)
            denom = max(n_samples - float(np.sum(gamma_eff)), 1.0)
            sigma2 = max(float(np.sum(residuals ** 2)) / denom, 1e-6)
            beta = 1.0 / sigma2

            # Critério de convergência na variação relativa de log(alpha)
            valid_idx = alpha_new < self.alpha_max
            if np.any(valid_idx):
                delta = float(
                    np.max(np.abs(np.log(alpha_new[valid_idx] + 1e-12) - np.log(alpha_a[valid_idx] + 1e-12)))
                )
            else:
                delta = 0.0

            alpha[active_mask] = alpha_new

            # Poda (pruning) de funções de base irrelevantes (alpha > alpha_max)
            keep_local = alpha_new < self.alpha_max
            # Garantir que ao menos o intercepto e 1 vetor permaneçam ativos
            if np.sum(keep_local[1:]) == 0:
                best_rv_local = 1 + int(np.argmin(alpha_new[1:])) if len(alpha_new) > 1 else 0
                keep_local[best_rv_local] = True
            keep_local[0] = True

            active_indices = np.where(active_mask)[0]
            pruned_indices = active_indices[~keep_local]
            active_mask[pruned_indices] = False

            if delta < self.tol and len(pruned_indices) == 0:
                self.converged_ = True
                break

        # Ajuste final dos pesos posteriores no conjunto ativo podado
        Phi_a = Phi[:, active_mask]
        alpha_a = alpha[active_mask]
        A = np.diag(alpha_a) + beta * (Phi_a.T @ Phi_a) + np.eye(Phi_a.shape[1]) * 1e-8
        Sigma = np.linalg.pinv(A)
        mu_active = beta * (Sigma @ (Phi_a.T @ y_arr))

        self.intercept_ = float(mu_active[0]) if active_mask[0] else 0.0
        self.beta_ = float(beta)
        active_sample_mask = active_mask[1:]
        self.relevance_indices_ = np.where(active_sample_mask)[0]
        self.relevance_vectors_ = X_arr[self.relevance_indices_]
        self.n_relevance_vectors_ = int(len(self.relevance_indices_))
        self.sparsity_ratio_ = float(1.0 - (self.n_relevance_vectors_ / max(n_samples, 1)))

        if active_mask[0]:
            self.weights_rv_ = mu_active[1:]
        else:
            self.weights_rv_ = mu_active

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if self.n_relevance_vectors_ == 0:
            return np.full(X_arr.shape[0], self.intercept_, dtype=np.float64)
        K_test = self._compute_kernel(X_arr, self.relevance_vectors_)
        return self.intercept_ + (K_test @ self.weights_rv_)


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


@st.cache_resource
def train_diagnostic_model(
    X_train_s: np.ndarray,
    y_train: pd.Series,
    X_test_s: np.ndarray,
    y_test: pd.Series,
    model_type: str = "Random Forest",
    # Hiperparâmetros Random Forest
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    # Hiperparâmetros RVR
    kernel: str = "rbf",
    gamma: float = 0.1,
    degree: int = 3,
    tol: float = 1e-3,
    max_iter: int = 300,
    threshold_alpha: float = 1e5,
) -> dict:
    """
    Treina e avalia o modelo selecionado (Random Forest ou RVR) separadamente nas bases
    de TREINO e TESTE para diagnóstico de overfitting e sintonia de hiperparâmetros.
    """
    start_time = time.perf_counter()

    if model_type == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
    else:
        model = RelevanceVectorRegressor(
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            tol=tol,
            max_iter=max_iter,
            alpha_max=threshold_alpha,
        )

    model.fit(X_train_s, y_train)
    train_time = time.perf_counter() - start_time

    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    # Cálculo das métricas em Treino e Teste
    r2_train = float(r2_score(y_train, y_pred_train))
    r2_test = float(r2_score(y_test, y_pred_test))
    r2_diff = r2_train - r2_test

    rmse_train = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    rmse_diff = rmse_train - rmse_test

    mae_train = float(mean_absolute_error(y_train, y_pred_train))
    mae_test = float(mean_absolute_error(y_test, y_pred_test))
    mae_diff = mae_train - mae_test

    # Alertas por métrica
    if r2_diff > 0.15:
        alerta_r2 = "⚠️ ALERTA: Overfitting Detectado (Gap > 0.15)"
    elif r2_diff > 0.10:
        alerta_r2 = "🟡 Atenção: Sobreajuste Moderado (0.10 < Gap ≤ 0.15)"
    elif r2_test < 0.50:
        alerta_r2 = "🔴 ALERTA: Possível Underfitting (Baixo R²)"
    else:
        alerta_r2 = "✅ OK: Generalização Saudável (Gap ≤ 0.15)"

    alerta_rmse = (
        "⚠️ Divergência Elevada no RMSE"
        if abs(rmse_diff) > 2.2
        else "✅ OK: Desvio de Erro Controlado"
    )
    alerta_mae = (
        "⚠️ Divergência Elevada no MAE"
        if abs(mae_diff) > 1.8
        else "✅ OK: Consistência entre Treino e Teste"
    )

    comparison_df = pd.DataFrame(
        [
            {
                "Métrica": "R² (Coeficiente de Determinação)",
                "Treino": round(r2_train, 4),
                "Teste": round(r2_test, 4),
                "Diferença (Treino - Teste)": round(r2_diff, 4),
                "Alerta": alerta_r2,
            },
            {
                "Métrica": "RMSE (Raiz do Erro Quadrático Médio)",
                "Treino": round(rmse_train, 4),
                "Teste": round(rmse_test, 4),
                "Diferença (Treino - Teste)": round(rmse_diff, 4),
                "Alerta": alerta_rmse,
            },
            {
                "Métrica": "MAE (Erro Absoluto Médio)",
                "Treino": round(mae_train, 4),
                "Teste": round(mae_test, 4),
                "Diferença (Treino - Teste)": round(mae_diff, 4),
                "Alerta": alerta_mae,
            },
        ]
    )

    rv_info = None
    if model_type != "Random Forest":
        n_rv = model.n_relevance_vectors_
        n_train = model.n_train_samples_
        rv_pct = (n_rv / max(n_train, 1)) * 100.0
        rv_info = {
            "n_relevance_vectors": n_rv,
            "n_train_samples": n_train,
            "rv_percentage": rv_pct,
            "sparsity_percentage": 100.0 - rv_pct,
            "relevance_indices": model.relevance_indices_,
            "n_iter": model.n_iter_,
            "converged": model.converged_,
            "beta": model.beta_,
        }

    return {
        "model": model,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "train_metrics": {"R²": r2_train, "RMSE": rmse_train, "MAE": mae_train},
        "test_metrics": {"R²": r2_test, "RMSE": rmse_test, "MAE": mae_test},
        "r2_diff": r2_diff,
        "rmse_diff": rmse_diff,
        "mae_diff": mae_diff,
        "train_time": train_time,
        "is_overfitting": r2_diff > 0.15,
        "comparison_df": comparison_df,
        "rv_info": rv_info,
    }


@st.cache_data
def compute_tuning_curve(
    X_train_s: np.ndarray,
    y_train: pd.Series,
    X_test_s: np.ndarray,
    y_test: pd.Series,
    model_type: str = "Random Forest",
    tuning_param: str = "max_depth",
    n_estimators: int = 100,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    kernel: str = "rbf",
    gamma: float = 0.1,
    degree: int = 3,
    tol: float = 1e-3,
    max_iter: int = 300,
    threshold_alpha: float = 1e5,
) -> pd.DataFrame:
    """
    Calcula a Curva de Sintonia (Validation Curve) avaliando o comportamento
    do R² no Treino e no Teste ao variar a profundidade máxima (max_depth no Random Forest)
    ou o hiperparâmetro gamma/tolerância no RVR.
    """
    rows = []

    if model_type == "Random Forest":
        depths = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, None]
        for d in depths:
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=d,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            )
            rf.fit(X_train_s, y_train)
            r2_tr = float(r2_score(y_train, rf.predict(X_train_s)))
            r2_te = float(r2_score(y_test, rf.predict(X_test_s)))
            rows.append(
                {
                    "param_label": "None (Ilimitado)" if d is None else str(d),
                    "R2_Treino": round(r2_tr, 4),
                    "R2_Teste": round(r2_te, 4),
                    "Gap_R2": round(r2_tr - r2_te, 4),
                }
            )
    else:
        if tuning_param == "gamma" and kernel in ("rbf", "poly"):
            gammas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0]
            for g in gammas:
                rvr = RelevanceVectorRegressor(
                    kernel=kernel,
                    gamma=g,
                    degree=degree,
                    tol=tol,
                    max_iter=max_iter,
                    alpha_max=threshold_alpha,
                )
                rvr.fit(X_train_s, y_train)
                r2_tr = float(r2_score(y_train, rvr.predict(X_train_s)))
                r2_te = float(r2_score(y_test, rvr.predict(X_test_s)))
                rows.append(
                    {
                        "param_label": str(g),
                        "R2_Treino": round(r2_tr, 4),
                        "R2_Teste": round(r2_te, 4),
                        "Gap_R2": round(r2_tr - r2_te, 4),
                    }
                )
        else:
            tols = [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
            for t_val in tols:
                rvr = RelevanceVectorRegressor(
                    kernel=kernel,
                    gamma=gamma,
                    degree=degree,
                    tol=t_val,
                    max_iter=max_iter,
                    alpha_max=threshold_alpha,
                )
                rvr.fit(X_train_s, y_train)
                r2_tr = float(r2_score(y_train, rvr.predict(X_train_s)))
                r2_te = float(r2_score(y_test, rvr.predict(X_test_s)))
                lbl = f"{t_val:.0e}" if t_val < 1e-3 else str(t_val)
                rows.append(
                    {
                        "param_label": lbl,
                        "R2_Treino": round(r2_tr, 4),
                        "R2_Teste": round(r2_te, 4),
                        "Gap_R2": round(r2_tr - r2_te, 4),
                    }
                )

    return pd.DataFrame(rows)

