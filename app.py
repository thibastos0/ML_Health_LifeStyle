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
)
from src.utils import (
    plot_eda_distributions,
    plot_correlation_matrix,
    plot_real_vs_predicted,
    plot_residuals_diagnostics,
    plot_feature_importance,
    FEATURE_TRANSLATION,
)

# -----------------------------------------------------------------------------
# 1. Configuração Global da Página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🩺 Oráculo Preditivo | Health Score",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilização CSS customizada para visual limpo e profissional
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1565C0;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #546E7A;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .banca-card {
        background-color: #F1F8E9;
        border-radius: 8px;
        padding: 14px;
        border-left: 5px solid #43A047;
        margin-bottom: 10px;
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
            use_container_width=True,
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
    st.plotly_chart(fig_corr, use_container_width=True)

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
    st.plotly_chart(fig_dist, use_container_width=True)


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
        use_container_width=True,
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
    st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown("---")

    # Expander de Overfitting
    with st.expander("⚠️ Diagnóstico de Overfitting: R² Treino vs. R² Teste & Bias-Variance", expanded=True):
        st.markdown(
            """
            A tabela abaixo compara o desempenho no conjunto de treino contra o conjunto de teste para identificar sobreajuste:
            """
        )
        st.dataframe(overfitting_df, use_container_width=True, hide_index=True)

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
    st.plotly_chart(fig_res, use_container_width=True)

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
    st.plotly_chart(fig_imp, use_container_width=True)

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

        submitted = st.form_submit_button("🔮 Calcular Health Score Preditivo", type="primary", use_container_width=True)

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
# 8. Seção Expander ao Fim de Todas as Páginas (Suporte para Arguição da Banca)
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("🎓 Suporte Teórico & Respostas para Arguição da Banca Avaliadora (Prof. Medina)", expanded=False):
    st.markdown("### 📚 Fundamentação Técnica e Evidências Experimentais")

    st.markdown(
        """
        <div class="banca-card">
        <b>Pergunta 1: Como o grupo identificou e tratou a Multicolinearidade?</b><br>
        <i>Fundamentação:</i> A multicolinearidade ocorre quando duas ou mais variáveis preditoras possuem alta correlação linear entre si, inflando a variância dos coeficientes da Regressão Linear e tornando o modelo instável.<br>
        <i>Evidência Prática no Projeto:</i> O cálculo da Matriz de Correlação de Pearson comprovou que todos os coeficientes cruzados entre variáveis explicativas estão abaixo de <b>|r| < 0.15</b>. Não há dependência linear mútua; logo, todas as 7 features foram retidas sem risco de distorção ou inflação de variância (VIF baixo).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="banca-card">
        <b>Pergunta 2: Como explicar o Trade-off Viés-Variância (Bias-Variance) e o Overfitting no Random Forest?</b><br>
        <i>Fundamentação:</i> Modelos complexos tendem a ter baixo viés mas alta variância. No Random Forest, quando as árvores são construídas sem limite de profundidade (<code>max_depth=None</code>), elas dividem os nós até a pureza extrema das folhas, decorando particularidades do conjunto de treino.<br>
        <i>Evidência Prática no Projeto:</i> O Random Forest obteve <b>R² = 97.10%</b> no treino frente a <b>83.05%</b> no teste (Gap de ~14%), caracterizando memorização parcial. Em contrapartida, a Regressão Linear Múltipla obteve <b>R² = 83.36%</b> no treino e <b>81.62%</b> no teste (Gap < 2%), apresentando excelente estabilidade e capacidade de generalização.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="banca-card">
        <b>Pergunta 3: Se vocês tivessem que apresentar os resultados para a diretoria médica de um hospital, como explicariam o MAPE (%) e por que ele é mais intuitivo que o R²?</b><br>
        <i>Fundamentação:</i> O R² (coeficiente de determinação) é uma medida estatística abstrata que representa a proporção da variância explicada. Stakeholders e gestores de saúde não raciocinam em 'variância explicada'. O <b>MAPE (Mean Absolute Percentage Error)</b> expressa o desvio médio relativo em porcentagem: (|y - ŷ| / y) * 100%.<br>
        <i>Evidência Prática no Projeto:</i> Nosso melhor modelo alcançou <b>MAPE = 5.52%</b>. Tradução clínica direta: para um paciente com Health Score real de 80 pontos, o modelo erra em média apenas <b>4.4 pontos</b> (precisão prática de <b>94.5%</b>), permitindo triagem segura e confiável.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="banca-card">
        <b>Pergunta 4: Para colocar em produção com milhares de requisições por segundo (baixa latência), qual modelo escolher e por quê? (Navalha de Ockham)</b><br>
        <i>Fundamentação:</i> O Princípio da Parsimônia estabelece que, perante desempenhos preditivos semelhantes, a solução de menor complexidade deve ser adotada. Na inferência, a Regressão Linear Múltipla executa um produto escalar vetorial simples <code>O(p)</code> em microssegundos, enquanto o Random Forest precisa percorrer 100 árvores de decisão individuais <code>O(M * profundidade)</code>.<br>
        <i>Evidência Prática no Projeto:</i> A Regressão Linear Múltipla treinou e inferiu em apenas <b>0.0021s</b> (cerca de <b>200 vezes mais rápida</b> que o Random Forest, que levou 0.42s), com uma perda insignificante de R² (81.62% vs 83.05%). Portanto, a <b>Regressão Linear Múltipla é a escolha ideal para produção de alta escala</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
