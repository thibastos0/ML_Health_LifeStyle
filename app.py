"""
Aplicação Streamlit: O Desafio do Oráculo Preditivo | Health Score
FATEC Indaiatuba - Prof. Medina
Autores: Gustavo Bravo, Gustavo Martins, Lucas, Thiago Bastos
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from src.data_processing import (
    load_raw_data,
    clean_data,
    get_modified_rows_comparison,
    prepare_train_test_data,
)
from src.models import (
    train_and_evaluate_models,
    predict_single_patient,
    train_diagnostic_model,
    compute_tuning_curve,
)
from src.utils import (
    plot_eda_distributions,
    plot_correlation_matrix,
    plot_real_vs_predicted,
    plot_residuals_diagnostics,
    plot_feature_importance,
    plot_train_test_scatter,
    plot_train_test_residuals_kde,
    plot_hyperparameter_tuning_curve,
    FEATURE_TRANSLATION,
)

# -----------------------------------------------------------------------------
# 1. Configuração Global da Página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Oráculo Preditivo | Health Score",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilização CSS customizada com compatibilidade total para Dark Mode e Light Mode
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2563eb;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: var(--text-color);
        opacity: 0.85;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        color: var(--text-color);
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .banca-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-left: 5px solid #22c55e;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 15px;
        color: var(--text-color);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    .banca-title {
        color: #22c55e;
        font-size: 1.12rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .banca-label {
        color: #38bdf8;
        font-weight: 600;
    }
    .banca-highlight {
        color: #eab308;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Carregamento dos Dados e Treinamento em Cache
# -----------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Health_LifeStyle-DS.csv")
if not os.path.exists(DATA_PATH):
    # Fallback para caso esteja na raiz
    DATA_PATH = "Health_LifeStyle-DS.csv"

raw_df = load_raw_data(DATA_PATH)
clean_df, num_modified = clean_data(raw_df)
X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = prepare_train_test_data(clean_df)
feature_names = list(X_train.columns)

models_results = train_and_evaluate_models(
    X_train_s, y_train, X_test_s, y_test, feature_names
)
metrics_df = models_results["metrics_df"]
overfitting_df = models_results["overfitting_df"]
predictions = models_results["predictions"]
feature_importance_df = models_results["feature_importance_df"]
trained_models = models_results["models"]


# -----------------------------------------------------------------------------
# 3. Barra Lateral (Sidebar)
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🩺 Oráculo Preditivo")
st.sidebar.caption("**FATEC Indaiatuba | Lab. Machine Learning**")
st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **👨‍🏫 Orientador:** Prof. Medina  
    **👥 Integrantes do Grupo:**
    - Gustavo Bravo
    - Gustavo Martins
    - Lucas
    - Thiago Bastos
    """
)
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navegação pelos Painéis:",
    [
        "📊 1. EDA & Higienização dos Dados",
        "📈 2. Comparativo de Performance dos Modelos",
        "🔍 3. Diagnóstico de Resíduos & Explicabilidade (XAI)",
        "🔮 4. Simulador Clínico Preditivo 🚀",
        "🧪 Diagnóstico & Overfitting",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    📌 **Repositório do Projeto:**  
    [GitHub: ML_Health_LifeStyle](https://github.com/thibastos0/ML_Health_LifeStyle)  
    *Previsão de Score de Saúde a partir de Hábitos de Vida.*
    """
)


# -----------------------------------------------------------------------------
# 4. Painel 1: EDA & Higienização dos Dados
# -----------------------------------------------------------------------------
if menu == "📊 1. EDA & Higienização dos Dados":
    st.markdown('<div class="main-header">📊 1. Análise Exploratória & Higienização dos Dados</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Exploração estatística e aplicação da técnica de Winsorization (.clip) para correção de incongruências físicas e fisiológicas.</div>',
        unsafe_allow_html=True,
    )

    # Cards de KPI
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="👥 Total de Pacientes", value=f"{len(raw_df):,}".replace(",", "."))
    with col2:
        st.metric(
            label="🎯 Health Score Médio",
            value=f"{clean_df['Health_Score'].mean():.1f} pts",
            delta=f"Desvio: ±{clean_df['Health_Score'].std():.1f}",
        )
    with col3:
        pct_mod = (num_modified / len(raw_df)) * 100
        st.metric(
            label="🧹 Registros Higienizados (.clip)",
            value=f"{num_modified} ({pct_mod:.1f}%)",
            delta="Ruídos Corrigidos",
        )
    with col4:
        st.metric(label="🔬 Features Explicativas", value=f"{len(feature_names)} preditores")

    st.markdown("---")

    # Seção: Higienização dos Dados
    st.subheader("🧹 Higienização de Dados: O Método `.clip()` na Prática Clínica")
    st.markdown(
        """
        Durante a análise exploratória, foram identificadas inconsistências geradas no processo de amostragem:
        - **Consumo de Álcool:** Existência de valores negativos (ex: -3.59 doses/semana), fisicamente impossíveis. Corrigidos com `clip(lower=0.0)`.
        - **Qualidade da Dieta:** Registros extrapolando a escala metodológica [0 - 100] (ex: 110.2). Corrigidos com `clip(lower=0.0, upper=100.0)`.
        - **IMC e Fisiologia:** Valores de IMC inferiores a 15 e sono fora do espectro viável. Corrigidos para limites fisiológicos plausíveis.
        """
    )

    with st.expander("🔍 Ver Amostra da Tabela Comparativa: Antes vs. Depois da Higienização", expanded=True):
        df_comparison = get_modified_rows_comparison(raw_df, clean_df)
        st.dataframe(
            df_comparison.head(15),
            width='stretch',
            hide_index=True,
        )
        st.caption(f"Mostrando as primeiras 15 correções de um total de {len(df_comparison)} campos ajustados.")

    st.markdown("---")

    # Seção: Matriz de Correlação
    st.subheader("📐 Matriz de Correlação Linear de Pearson & Multicolinearidade")
    st.markdown(
        """
        O Heatmap abaixo ilustra o coeficiente de correlação linear ($r$) entre todos os pares de variáveis.
        **Conclusão Crítica:** Todos os preditores de hábitos apresentam correlações extremamente fracas entre si ($|r| < 0.15$),
        o que **comprova empiricamente a ausência de multicolinearidade** e valida o uso da Regressão Linear Múltipla sem necessidade de eliminação prévia de variáveis.
        """
    )
    fig_corr = plot_correlation_matrix(clean_df)
    st.plotly_chart(fig_corr, width='stretch')

    st.markdown("---")

    # Seção: Distribuições Univariadas
    st.subheader("📊 Distribuição e Boxplot por Variável")
    var_options = ["Health_Score"] + feature_names
    selected_var = st.selectbox(
        "Selecione uma variável para inspecionar:",
        var_options,
        format_func=lambda x: FEATURE_TRANSLATION.get(x, x),
    )

    fig_dist = plot_eda_distributions(clean_df, selected_var)
    st.plotly_chart(fig_dist, width='stretch')


# -----------------------------------------------------------------------------
# 5. Painel 2: Comparativo de Performance dos Modelos
# -----------------------------------------------------------------------------
elif menu == "📈 2. Comparativo de Performance dos Modelos":
    st.markdown('<div class="main-header">📈 2. Comparativo de Performance dos Modelos</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Avaliação rigorosa dos 4 algoritmos preditivos através de Hold-out (80/20) e Validação Cruzada 5-Fold.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("🏆 Tabela Consolidada de Métricas de Avaliação")
    st.dataframe(
        metrics_df.style.highlight_max(subset=["R² (Teste)", "R² (CV 5-Fold)"], color="#C8E6C9")
        .highlight_min(subset=["MAE", "RMSE", "Tempo de Treino (s)"], color="#DCEDC8"),
        width='stretch',
        hide_index=True,
    )

    col_inf1, col_inf2, col_inf3 = st.columns(3)
    with col_inf1:
        st.info("⚡ **Menor Latência:** Regressão Linear Múltipla (~0.002s) — ideal para APIs de alto throughput.")
    with col_inf2:
        st.success("🎯 **Maior Acurácia:** Random Forest Regressor ($R^2 = 83.1\\%$, $MAE = 4.15$).")
    with col_inf3:
        st.warning("⚖️ **Equilíbrio Não Linear:** SVR ($R^2 = 82.1\\%$, $MAE = 4.22$).")

    st.markdown("---")

    # Gráfico Reais vs Preditos
    st.subheader("🎯 Dispersão: Valores Reais vs. Valores Preditos")
    st.markdown(
        "Clique nos nomes dos modelos na legenda para ativar ou ocultar cada modelo individualmente. "
        "A linha tracejada vermelha representa a precisão perfeita (45°)."
    )
    fig_pred = plot_real_vs_predicted(y_test, predictions)
    st.plotly_chart(fig_pred, width='stretch')

    st.markdown("---")

    # Expander de Overfitting
    with st.expander("⚠️ Diagnóstico de Overfitting: R² Treino vs. R² Teste & Bias-Variance", expanded=True):
        st.markdown(
            """
            A tabela abaixo compara o desempenho no conjunto de treino contra o conjunto de teste para identificar sobreajuste:
            """
        )
        st.dataframe(overfitting_df, width='stretch', hide_index=True)

        st.markdown(
            """
            **Análise Técnica para a Banca:**
            - **Random Forest Regressor:** Atingiu $R^2 = 97.10\\%$ no treino e $83.05\\%$ no teste ($Gap \\approx 14\\%$).
              Essa disparidade ocorre porque o modelo foi instanciado com `max_depth=None`, permitindo que as 100 árvores
              crescessem até a pureza total das folhas, memorizando nuances do conjunto de treino.
            - **Regressão Linear Múltipla:** Apresenta $R^2 = 83.36\\%$ no treino e $81.62\\%$ no teste ($Gap < 2\\%$),
              demonstrando **excelente generalização**, estabilidade e baixo risco de overfitting.
            """
        )


# -----------------------------------------------------------------------------
# 6. Painel 3: Diagnóstico de Resíduos & Explicabilidade (XAI)
# -----------------------------------------------------------------------------
elif menu == "🔍 3. Diagnóstico de Resíduos & Explicabilidade (XAI)":
    st.markdown('<div class="main-header">🔍 3. Diagnóstico de Resíduos & Explicabilidade (XAI)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Auditoria estatística dos resíduos (Homocedasticidade) e Interpretação dos fatores determinantes da saúde humana.</div>',
        unsafe_allow_html=True,
    )

    # Diagnóstico de Resíduos
    st.subheader("🔬 Validação de Homocedasticidade e Aleatoriedade dos Erros")
    modelo_residuo = st.selectbox(
        "Selecione o Modelo para Análise de Resíduos:",
        ["Regressão Linear Múltipla", "Random Forest Regressor", "Support Vector Regressor (SVR)"],
    )

    y_pred_sel = predictions[modelo_residuo]
    fig_res = plot_residuals_diagnostics(y_test, y_pred_sel, model_name=modelo_residuo)
    st.plotly_chart(fig_res, width='stretch')

    st.markdown(
        """
        **Critérios de Homocedasticidade Atendidos:**
        1. **Média Zero:** Os resíduos flutuam uniformemente ao redor da linha tracejada vermelha ($y=0$).
        2. **Variância Constante:** Não há formato de funil (heterocedasticidade), comprovando que o erro de previsão não cresce conforme o escore do paciente aumenta.
        3. **Normalidade:** O gráfico de violino à direita comprova distribuição unimodal e simétrica dos erros.
        """
    )

    st.markdown("---")

    # Explicabilidade: Feature Importance e Betas
    st.subheader("🧠 Explicabilidade de Modelos: O que Realmente Move o Health Score?")
    st.markdown(
        """
        Comparativo entre a **Importância Relativa de Features** (Gini Importance no Random Forest) e os
        **Coeficientes Angulares Padronizados $\\beta$** (Regressão Linear Múltipla):
        """
    )

    fig_imp = plot_feature_importance(feature_importance_df)
    st.plotly_chart(fig_imp, width='stretch')

    col_xai1, col_xai2 = st.columns(2)
    with col_xai1:
        st.markdown(
            """
            #### 🥗 1. Qualidade da Dieta (~50,4% de Importância)
            - É o preditor isolado mais influente sobre a saúde.
            - Na Regressão Linear, seu $\\beta$ é de **+8.95**, o maior impacto positivo unitário.
            - Cada salto qualitativo na alimentação alavanca diretamente a expectativa de vida.
            """
        )
    with col_xai2:
        st.markdown(
            """
            #### ⚖️ 2. Índice de Massa Corporal - IMC (~21,2% de Importância)
            - É o segundo preditor mais crítico e o maior fator de risco biológico.
            - Na Regressão Linear, seu $\\beta$ é de **-5.83**, o maior coeficiente penalizador.
            - Elevações no IMC reduzem drasticamente o score de saúde do indivíduo.
            """
        )


# -----------------------------------------------------------------------------
# 7. Painel 4: Simulador Clínico Preditivo 🚀 (Gran Finale)
# -----------------------------------------------------------------------------
elif menu == "🔮 4. Simulador Clínico Preditivo 🚀":
    st.markdown('<div class="main-header">🔮 4. Simulador Clínico Preditivo & Recomendador</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Insira os parâmetros clínicos e hábitos de um paciente para obter a predição imediata do Health Score e recomendações preventivas personalizadas.</div>',
        unsafe_allow_html=True,
    )

    with st.form("form_simulador"):
        col_form1, col_form2 = st.columns(2)

        with col_form1:
            st.markdown("##### 👤 Perfil Biológico e Hábitos Nutricionais")
            idade = st.slider("Idade (anos):", min_value=18, max_value=90, value=42, step=1)
            imc = st.slider("Índice de Massa Corporal (IMC em kg/m²):", min_value=15.0, max_value=45.0, value=25.5, step=0.5)
            dieta = st.slider("Qualidade da Dieta (Escala de 0 a 100):", min_value=0, max_value=100, value=65, step=1)
            sono = st.slider("Média de Horas de Sono (h/noite):", min_value=3.0, max_value=11.0, value=7.0, step=0.5)

        with col_form2:
            st.markdown("##### 🏃 Atividades e Exposição a Fatores de Risco")
            exercicio = st.slider("Frequência de Exercício Físico (dias por semana):", min_value=0, max_value=7, value=3, step=1)
            alcool = st.slider("Consumo de Álcool (doses padrão por semana):", min_value=0.0, max_value=25.0, value=2.0, step=0.5)
            tabagismo_label = st.radio("Status de Tabagismo:", ["Não Tabagista (0)", "Tabagista Ativo (1)"], index=0)
            tabagismo = 1 if "1" in tabagismo_label else 0

            modelo_escolhido = st.selectbox(
                "Algoritmo para Inferência:",
                [
                    "Regressão Linear Múltipla (Recomendado pela Navalha de Ockham)",
                    "Random Forest Regressor (Máxima Acurácia Empírica)",
                ],
            )

        submitted = st.form_submit_button("🔮 Calcular Health Score Preditivo", type="primary", width='stretch')

    if submitted:
        patient_dict = {
            "Age": float(idade),
            "BMI": float(imc),
            "Exercise_Frequency": float(exercicio),
            "Diet_Quality": float(dieta),
            "Sleep_Hours": float(sono),
            "Smoking_Status": float(tabagismo),
            "Alcohol_Consumption": float(alcool),
        }

        # Selecionar o estimador
        model_name = "Regressão Linear Múltipla" if "Linear" in modelo_escolhido else "Random Forest Regressor"
        selected_estimator = trained_models[model_name]

        predicted_score = predict_single_patient(selected_estimator, scaler, patient_dict, feature_names)

        st.markdown("---")
        st.subheader("🎯 Resultado da Avaliação Clínica")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            # Classificação por faixa de saúde
            if predicted_score >= 85.0:
                categoria = "🟢 Saúde Excelente / Baixo Risco"
                badge_color = "#2E7D32"
            elif predicted_score >= 70.0:
                categoria = "🟡 Saúde Boa / Estável"
                badge_color = "#F57F17"
            elif predicted_score >= 50.0:
                categoria = "🟠 Atenção Médica Recomendada"
                badge_color = "#E65100"
            else:
                categoria = "🔴 Alerta Crítico / Risco Elevado"
                badge_color = "#C62828"

            st.metric(
                label=f"Health Score Predito ({model_name})",
                value=f"{predicted_score:.1f} pts",
                delta=f"{predicted_score - clean_df['Health_Score'].mean():+.1f} vs. Média Populacional",
            )
            st.markdown(
                f"<div style='background-color: {badge_color}; color: white; padding: 10px; border-radius: 6px; font-weight: bold; text-align: center;'>{categoria}</div>",
                unsafe_allow_html=True,
            )

        with res_col2:
            st.markdown("#### 💡 Prescrições e Recomendações Dinâmicas para Aumento de Score:")

            # Simulações de melhorias clínicas
            recomendacoes = []

            # 1. Simulação de Dieta (+15 pontos)
            if dieta < 85:
                dict_dieta = patient_dict.copy()
                dict_dieta["Diet_Quality"] = min(100.0, dieta + 15)
                score_dieta = predict_single_patient(selected_estimator, scaler, dict_dieta, feature_names)
                ganho_dieta = score_dieta - predicted_score
                recomendacoes.append(
                    f"🥗 **Melhorar a Dieta em +15 pontos (de {dieta} para {dict_dieta['Diet_Quality']:.0f}):** "
                    f"Eleva o Health Score para **{score_dieta:.1f} pts** (+{ganho_dieta:.1f} pontos)."
                )

            # 2. Simulação de IMC (reduzir para faixa ideal 22-23)
            if imc > 25.0:
                dict_imc = patient_dict.copy()
                dict_imc["BMI"] = 23.0
                score_imc = predict_single_patient(selected_estimator, scaler, dict_imc, feature_names)
                ganho_imc = score_imc - predicted_score
                recomendacoes.append(
                    f"⚖️ **Reduzir o IMC para a faixa saudável (23.0 kg/m²):** "
                    f"Eleva o Health Score para **{score_imc:.1f} pts** (+{ganho_imc:.1f} pontos)."
                )

            # 3. Cessação do Tabagismo
            if tabagismo == 1:
                dict_smoke = patient_dict.copy()
                dict_smoke["Smoking_Status"] = 0.0
                score_smoke = predict_single_patient(selected_estimator, scaler, dict_smoke, feature_names)
                ganho_smoke = score_smoke - predicted_score
                recomendacoes.append(
                    f"🚭 **Cessação Imediata do Tabagismo:** "
                    f"Eleva o Health Score para **{score_smoke:.1f} pts** (+{ganho_smoke:.1f} pontos instantâneos)."
                )

            # 4. Exercício Físico
            if exercicio < 4:
                dict_exe = patient_dict.copy()
                dict_exe["Exercise_Frequency"] = 5.0
                score_exe = predict_single_patient(selected_estimator, scaler, dict_exe, feature_names)
                ganho_exe = score_exe - predicted_score
                recomendacoes.append(
                    f"🏃 **Aumentar Exercícios para 5 dias/semana:** "
                    f"Eleva o Health Score para **{score_exe:.1f} pts** (+{ganho_exe:.1f} pontos)."
                )

            # 5. Moderação do Álcool
            if alcool > 3.0:
                dict_alc = patient_dict.copy()
                dict_alc["Alcohol_Consumption"] = 0.0
                score_alc = predict_single_patient(selected_estimator, scaler, dict_alc, feature_names)
                ganho_alc = score_alc - predicted_score
                recomendacoes.append(
                    f"🍷 **Reduzir Álcool para 0 doses/semana:** "
                    f"Eleva o Health Score para **{score_alc:.1f} pts** (+{ganho_alc:.1f} pontos)."
                )

            if recomendacoes:
                for rec in recomendacoes:
                    st.markdown(f"- {rec}")
            else:
                st.balloons()
                st.success("🎉 Parabéns! Seus hábitos de vida já são exemplares e situam-se nos percentis superiores de saúde!")


# -----------------------------------------------------------------------------
# 8. Painel 5: Diagnóstico de Overfitting e Sintonia de Hiperparâmetros
# -----------------------------------------------------------------------------
elif menu == "🧪 Diagnóstico & Overfitting":
    st.markdown(
        '<div class="main-header">🧪 Diagnóstico de Overfitting & Sintonia de Hiperparâmetros</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Laboratório interativo para análise do Trade-off Viés-Variância, '
        "poda estrutural de árvores (Random Forest) e esparsidade Bayesiana (Relevance Vector Regression - RVR).</div>",
        unsafe_allow_html=True,
    )

    # 1. Painel de Controles Interativos (Seleção do Modelo, Split e Hiperparâmetros)
    st.subheader("🎛️ 1. Configuração Experimental & Hiperparâmetros")

    with st.container():
        col_cfg1, col_cfg2 = st.columns([1.3, 1.0])
        with col_cfg1:
            diag_model_choice = st.radio(
                "Selecione a Arquitetura do Modelo:",
                options=["Random Forest", "RVR (Relevance Vector Regression)"],
                horizontal=True,
                help="alterne entre o ensemble de árvores (Random Forest) e o modelo esparso probabilístico (RVR).",
            )
        with col_cfg2:
            test_size_pct = st.slider(
                "Proporção do Conjunto de Teste (test_size):",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                format="%d%%",
                help="Define a fração de dados retida exclusivamente para validação fora da amostra (entre 10% e 40%).",
            )
            test_size_ratio = test_size_pct / 100.0

    st.markdown("#### ⚙️ Sintonia Fina de Hiperparâmetros")

    # Inicializar variáveis de hiperparâmetros com defaults seguros
    rf_n_estimators = 100
    rf_max_depth = None
    rf_min_samples_split = 2
    rf_min_samples_leaf = 1

    rvr_kernel = "rbf"
    rvr_gamma = 0.1
    rvr_degree = 3
    rvr_tol = 1e-3
    rvr_max_iter = 300
    rvr_threshold_alpha = 1e5

    if diag_model_choice == "Random Forest":
        c_rf1, c_rf2, c_rf3, c_rf4 = st.columns(4)
        with c_rf1:
            rf_n_estimators = st.slider(
                "Nº de Árvores (n_estimators):",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Quantidade de árvores de decisão no comitê (ensemble).",
            )
        with c_rf2:
            depth_options = ["None (Ilimitado)"] + [str(d) for d in range(2, 31)]
            selected_depth_str = st.select_slider(
                "Profundidade Máx. (max_depth):",
                options=depth_options,
                value="None (Ilimitado)",
                help="Controla o crescimento vertical das árvores. 'None' permite ramificação até pureza total (maior risco de overfitting).",
            )
            rf_max_depth = None if selected_depth_str == "None (Ilimitado)" else int(selected_depth_str)
        with c_rf3:
            rf_min_samples_split = st.slider(
                "Mín. Amostras p/ Divisão (min_samples_split):",
                min_value=2,
                max_value=30,
                value=2,
                step=1,
                help="Número mínimo de amostras exigido para dividir um nó interno.",
            )
        with c_rf4:
            rf_min_samples_leaf = st.slider(
                "Mín. Amostras na Folha (min_samples_leaf):",
                min_value=1,
                max_value=25,
                value=1,
                step=1,
                help="Número mínimo de amostras que todo nó folha terminal deve conter (regularização estrutural).",
            )
        model_type_key = "Random Forest"
    else:
        c_rvr1, c_rvr2, c_rvr3, c_rvr4 = st.columns(4)
        with c_rvr1:
            rvr_kernel = st.selectbox(
                "Função de Kernel (kernel):",
                options=["rbf", "linear", "poly"],
                index=0,
                help="Define a base de projeção dos Vetores de Relevância ('rbf' Gaussiano, 'linear' ou 'poly' Polinomial).",
            )
            if rvr_kernel == "poly":
                rvr_degree = st.slider(
                    "Grau do Polinômio (degree):",
                    min_value=2,
                    max_value=5,
                    value=3,
                    step=1,
                )
        with c_rvr2:
            rvr_gamma = st.select_slider(
                "Coeficiente de Kernel (gamma):",
                options=[0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0],
                value=0.1,
                disabled=(rvr_kernel == "linear"),
                help="Controla o raio de influência de cada Vetor de Relevância. Valores altos criam funções muito sinuosas (risco de overfitting).",
            )
        with c_rvr3:
            rvr_tol = st.select_slider(
                "Tolerância de Convergência (tol):",
                options=[1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
                value=1e-3,
                format_func=lambda v: f"{v:.0e}" if v < 1e-3 else f"{v}",
                help="Critério de parada para a variação máxima do logaritmo das precisões a priori (α).",
            )
        with c_rvr4:
            rvr_max_iter = st.slider(
                "Máx. Iterações EM (max_iter):",
                min_value=50,
                max_value=800,
                value=300,
                step=50,
                help="Limite máximo de ciclos de atualização Bayesiana (MacKay / ARD).",
            )
        model_type_key = "RVR"

    st.markdown("---")

    # 2. Particionamento Dinâmico (Cacheado) e Avaliação Treino vs. Teste
    (
        X_tr_diag,
        X_te_diag,
        y_tr_diag,
        y_te_diag,
        X_tr_s_diag,
        X_te_s_diag,
        _,
    ) = prepare_train_test_data(clean_df, test_size=test_size_ratio, random_state=42)

    diag_res = train_diagnostic_model(
        X_train_s=X_tr_s_diag,
        y_train=y_tr_diag,
        X_test_s=X_te_s_diag,
        y_test=y_te_diag,
        model_type=model_type_key,
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_split=rf_min_samples_split,
        min_samples_leaf=rf_min_samples_leaf,
        kernel=rvr_kernel,
        gamma=rvr_gamma,
        degree=rvr_degree,
        tol=rvr_tol,
        max_iter=rvr_max_iter,
        threshold_alpha=rvr_threshold_alpha,
    )

    tr_metrics = diag_res["train_metrics"]
    te_metrics = diag_res["test_metrics"]
    r2_diff = diag_res["r2_diff"]
    is_overfitting = diag_res["is_overfitting"]
    comp_df = diag_res["comparison_df"]
    rv_info = diag_res["rv_info"]

    # 3. Alerta Visual de Overfitting (Limiar: R² Treino - R² Teste > 0.15)
    st.subheader("📊 2. Avaliação Comparativa: Treino vs. Teste")

    if is_overfitting:
        st.warning(
            f"⚠️ **ALERTA DE OVERFITTING DETECTADO!** A diferença de **R² entre Treino ({tr_metrics['R²']:.4f}) e Teste ({te_metrics['R²']:.4f})** "
            f"é de **{r2_diff:+.4f}** (superior ao limite crítico de **0.15**). "
            "O modelo está memorizando ruídos da partição de treinamento e perdendo capacidade de generalização. "
            "Sugestão: reduza `max_depth` / aumente `min_samples_leaf` (no Random Forest) ou reduza `gamma` (no RVR)."
        )
    else:
        st.success(
            f"✅ **GENERALIZAÇÃO SAUDÁVEL!** A diferença de **R² entre Treino ({tr_metrics['R²']:.4f}) e Teste ({te_metrics['R²']:.4f})** "
            f"é de **{r2_diff:+.4f}** (dentro do limite seguro ≤ **0.15**). "
            "O modelo apresenta equilíbrio adequado entre viés e variância."
        )

    # KPIs principais de Desempenho e Particionamento
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric(
        "Amostras (Treino / Teste)",
        f"{len(y_tr_diag)} / {len(y_te_diag)}",
        f"Split: {100 - test_size_pct}% / {test_size_pct}%",
        delta_color="off",
    )
    kpi2.metric(
        "R² (Treino vs. Teste)",
        f"{te_metrics['R²']:.4f}",
        f"Treino: {tr_metrics['R²']:.4f} (Gap: {r2_diff:+.4f})",
        delta_color="inverse" if is_overfitting else "normal",
    )
    kpi3.metric(
        "RMSE (Teste)",
        f"{te_metrics['RMSE']:.3f} pts",
        f"Treino: {tr_metrics['RMSE']:.3f} pts",
        delta_color="off",
    )
    kpi4.metric(
        "MAE (Teste)",
        f"{te_metrics['MAE']:.3f} pts",
        f"Treino: {tr_metrics['MAE']:.3f} pts",
        delta_color="off",
    )
    kpi5.metric(
        "Tempo de Treino",
        f"{diag_res['train_time']:.4f} s",
        diag_model_choice.split(" ")[0],
        delta_color="off",
    )

    # Destaque Exclusivo para RVR: Vetores de Relevância (Relevance Vectors)
    if model_type_key == "RVR" and rv_info is not None:
        st.markdown("#### 🎯 Esparsidade Bayesiana: Vetores de Relevância (Relevance Vectors)")
        rv_col1, rv_col2, rv_col3, rv_col4 = st.columns(4)
        n_rv = rv_info["n_relevance_vectors"]
        n_tr_tot = rv_info["n_train_samples"]
        rv_pct = rv_info["rv_percentage"]
        sparsity_pct = rv_info["sparsity_percentage"]
        conv_status = "Convergiu ✅" if rv_info["converged"] else "Limite Atingido ⚠️"

        rv_col1.metric(
            "Vetores de Relevância Retidos",
            f"{n_rv} / {n_tr_tot}",
            f"{rv_pct:.1f}% do conjunto de treino",
            delta_color="off",
        )
        rv_col2.metric(
            "Taxa de Compressão (Esparsidade)",
            f"{sparsity_pct:.1f}%",
            f"{n_tr_tot - n_rv} amostras podadas (α → ∞)",
            delta_color="normal",
        )
        rv_col3.metric(
            "Iterações EM / ARD",
            f"{rv_info['n_iter']} ciclos",
            conv_status,
            delta_color="off",
        )
        rv_col4.metric(
            "Precisão do Ruído Estimada (β)",
            f"{rv_info['beta']:.4f}",
            f"σ² ≈ {1.0 / max(rv_info['beta'], 1e-8):.2f}",
            delta_color="off",
        )
        st.progress(
            min(max(rv_pct / 100.0, 0.0), 1.0),
            text=(
                f"Proporção de Vetores de Relevância ativos: {n_rv} de {n_tr_tot} amostras de treino "
                f"({rv_pct:.2f}% utilizados | {sparsity_pct:.2f}% descartados via Automatic Relevance Determination)."
            ),
        )

    # Tabela Comparativa Obrigatória: Métrica | Treino | Teste | Diferença (Treino - Teste) | Alerta
    st.markdown("#### 📋 Tabela Comparativa de Métricas (Treino vs. Teste)")

    def _style_alert_col(val: str) -> str:
        if "ALERTA" in str(val) or "Elevada" in str(val):
            return "background-color: rgba(239, 68, 68, 0.18); color: #b91c1c; font-weight: 700;"
        return "background-color: rgba(34, 197, 94, 0.15); color: #15803d; font-weight: 600;"

    styled_comp = comp_df.style.format(
        {
            "Treino": "{:.4f}",
            "Teste": "{:.4f}",
            "Diferença (Treino - Teste)": "{:+.4f}",
        }
    ).map(_style_alert_col, subset=["Alerta"])

    st.dataframe(styled_comp, width='stretch', hide_index=True)

    st.markdown("---")

    # 4. Visualizações Gráficas Interativas
    st.subheader("📈 3. Diagnósticos Visuais de Generalização e Sintonia")

    viz_tab1, viz_tab2, viz_tab3 = st.tabs(
        [
            "🎯 Dispersão Real vs. Predito (Treino vs. Teste)",
            "📊 Distribuição de Resíduos (Histograma & KDE)",
            "📈 Curva de Sintonia de Hiperparâmetros (R²)",
        ]
    )

    with viz_tab1:
        st.markdown(
            "Comparação lado a lado da aderência das predições à **linha ideal ($y = x$)** nos conjuntos de "
            "**Treino** e **Teste**. Quando há *overfitting*, os pontos de Treino colam perfeitamente na reta diagonal, "
            "enquanto no Teste observa-se forte dispersão."
        )
        rv_indices = rv_info["relevance_indices"] if (model_type_key == "RVR" and rv_info is not None) else None
        fig_scatter_tt = plot_train_test_scatter(
            y_train=y_tr_diag,
            y_pred_train=diag_res["y_pred_train"],
            y_test=y_te_diag,
            y_pred_test=diag_res["y_pred_test"],
            model_name=diag_model_choice,
            relevance_indices=rv_indices,
        )
        st.plotly_chart(fig_scatter_tt, width='stretch')

    with viz_tab2:
        st.markdown(
            "Análise comparativa da **distribuição dos erros ($e = y - \\hat{y}$)** via Histograma de Densidade e "
            "**Curva KDE (Kernel Density Estimation)**. Um pico excessivamente estreito e alto no Treino acompanhado "
            "de uma cauda larga no Teste evidencia memorização."
        )
        fig_res_kde = plot_train_test_residuals_kde(
            y_train=y_tr_diag,
            y_pred_train=diag_res["y_pred_train"],
            y_test=y_te_diag,
            y_pred_test=diag_res["y_pred_test"],
            model_name=diag_model_choice,
        )
        st.plotly_chart(fig_res_kde, width='stretch')

    with viz_tab3:
        st.markdown(
            "A **Curva de Sintonia (Validation Curve)** ilustra como a complexidade do modelo afeta simultaneamente "
            "o $R^2$ de Treino e o $R^2$ de Teste, permitindo identificar o ponto ótimo antes da abertura do gap de overfitting."
        )
        if model_type_key == "Random Forest":
            tuning_param_name = "max_depth"
            tuning_param_label = "Profundidade Máxima das Árvores (max_depth)"
            current_val_lbl = "None (Ilimitado)" if rf_max_depth is None else str(rf_max_depth)
        else:
            rvr_curve_choice = st.radio(
                "Parâmetro do RVR para Curva de Sintonia:",
                options=["gamma (Largura do Kernel)", "tol (Tolerância de Convergência)"],
                horizontal=True,
            )
            if rvr_curve_choice.startswith("gamma"):
                tuning_param_name = "gamma"
                tuning_param_label = "Parâmetro Gamma do Kernel (gamma)"
                current_val_lbl = str(rvr_gamma)
            else:
                tuning_param_name = "tol"
                tuning_param_label = "Tolerância de Convergência (tol)"
                current_val_lbl = f"{rvr_tol:.0e}" if rvr_tol < 1e-3 else str(rvr_tol)

        tuning_df = compute_tuning_curve(
            X_train_s=X_tr_s_diag,
            y_train=y_tr_diag,
            X_test_s=X_te_s_diag,
            y_test=y_te_diag,
            model_type=model_type_key,
            tuning_param=tuning_param_name,
            n_estimators=rf_n_estimators,
            min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf,
            kernel=rvr_kernel,
            gamma=rvr_gamma,
            degree=rvr_degree,
            tol=rvr_tol,
            max_iter=rvr_max_iter,
        )

        fig_tuning = plot_hyperparameter_tuning_curve(
            tuning_df=tuning_df,
            param_label=tuning_param_label,
            current_val_label=current_val_lbl,
            model_name=diag_model_choice,
        )
        st.plotly_chart(fig_tuning, width='stretch')

        with st.expander("🔎 Visualizar Tabela Numérica da Curva de Sintonia"):
            st.dataframe(
                tuning_df.rename(
                    columns={
                        "param_label": tuning_param_label,
                        "R2_Treino": "R² Treino",
                        "R2_Teste": "R² Teste",
                        "Gap_R2": "Gap (Treino - Teste)",
                    }
                ).style.format(
                    {
                        "R² Treino": "{:.4f}",
                        "R² Teste": "{:.4f}",
                        "Gap (Treino - Teste)": "{:+.4f}",
                    }
                ),
                width='stretch',
                hide_index=True,
            )


# -----------------------------------------------------------------------------
# 9. Seção Expander ao Fim de Todas as Páginas (Suporte para Arguição da Banca)
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("🎓 Suporte Teórico & Respostas para Arguição da Banca Avaliadora (Prof. Medina)", expanded=False):
    st.markdown("### 📚 Fundamentação Técnica e Evidências Experimentais")

    st.markdown(
        """
        <div class="banca-card">
            <div class="banca-title">❓ Pergunta 1: Como o grupo identificou e tratou a Multicolinearidade?</div>
            <p><span class="banca-label">🔬 Fundamentação Teórica:</span> A multicolinearidade ocorre quando duas ou mais variáveis preditoras possuem forte correlação linear mútua, inflando a variância dos coeficientes da Regressão Linear e gerando instabilidade nas estimativas.</p>
            <p><span class="banca-label">📊 Evidência Prática no Projeto:</span> A Matriz de Correlação de Pearson comprovou que todos os coeficientes cruzados entre variáveis explicativas estão contidos na faixa de <span class="banca-highlight">|r| &lt; 0.15</span>. Não há colinearidade prejudicial entre os preditores; portanto, todas as 7 features foram retidas sem risco de distorção ou inflação de variância (baixo VIF).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="banca-card">
            <div class="banca-title">❓ Pergunta 2: Como explicar o Trade-off Viés-Variância (Bias-Variance) e o Overfitting no Random Forest?</div>
            <p><span class="banca-label">🔬 Fundamentação Teórica:</span> Modelos de alta complexidade tendem a exibir baixo viés e alta variância. No Random Forest, quando as árvores são treinadas sem restrição de profundidade (<code>max_depth=None</code>), os nós se ramificam até a pureza total das folhas, decorando os ruídos amostrais do conjunto de treino.</p>
            <p><span class="banca-label">📊 Evidência Prática no Projeto:</span> O Random Forest atingiu <span class="banca-highlight">R² = 97.10%</span> no treino frente a <span class="banca-highlight">83.05%</span> no teste (Gap de ~14%), caracterizando memorização parcial. Em contraste, a Regressão Linear Múltipla obteve <span class="banca-highlight">R² = 83.36%</span> no treino e <span class="banca-highlight">81.62%</span> no teste (Gap &lt; 2%), comprovando alta capacidade de generalização e estabilidade.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="banca-card">
            <div class="banca-title">❓ Pergunta 3: Se vocês tivessem que apresentar os resultados para a diretoria médica de um hospital, como explicariam o MAPE (%) e por que ele é mais intuitivo que o R²?</div>
            <p><span class="banca-label">🔬 Fundamentação Teórica:</span> O R² (coeficiente de determinação) é uma métrica abstrata que expressa proporção de variância explicada, algo de difícil digestão para a diretoria executiva ou médica. O <b>MAPE (Mean Absolute Percentage Error)</b> expressa o desvio relativo médio em porcentagem: (|y - ŷ| / y) * 100%.</p>
            <p><span class="banca-label">📊 Evidência Prática no Projeto:</span> Nosso modelo atingiu <span class="banca-highlight">MAPE = 5.52%</span>. Na linguagem médica: para um paciente cujo Health Score real é de 80 pontos, o modelo comete um erro médio de apenas <span class="banca-highlight">~4.4 pontos</span> (uma precisão clínica de <span class="banca-highlight">94.5%</span>), ideal para um sistema seguro de triagem e direcionamento de cuidados preventivos.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="banca-card">
            <div class="banca-title">❓ Pergunta 4: Para colocar em produção com milhares de requisições por segundo (baixa latência), qual modelo escolher e por quê? (Navalha de Ockham)</div>
            <p><span class="banca-label">🔬 Fundamentação Teórica:</span> O Princípio da Parsimônia estabelece que, diante de performances similares, o modelo mais simples deve ser adotado. Em produção de alta volumetria, a Regressão Linear Múltipla calcula a inferência via produto escalar <code>O(p)</code> em microssegundos com consumo ínfimo de CPU/RAM, enquanto o Random Forest precisa percorrer 100 árvores de decisão individuais <code>O(M * profundidade)</code>.</p>
            <p><span class="banca-label">📊 Evidência Prática no Projeto:</span> A Regressão Linear Múltipla treinou e inferiu em apenas <span class="banca-highlight">0.0021s</span> (<span class="banca-highlight">~200 vezes mais rápida</span> que o Random Forest a 0.42s), com diferença de apenas 1.4% no R² (81.62% vs 83.05%). Além do ganho de latência, a Regressão Linear oferece explicabilidade médica total através dos seus coeficientes β.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
