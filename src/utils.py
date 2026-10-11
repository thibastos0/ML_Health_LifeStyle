"""
Módulo de Visualizações Gráficas Interativas com Plotly
Projeto: O Desafio do Oráculo Preditivo | Health Score
FATEC Indaiatuba - Prof. Medina
Autores: Gustavo Bravo, Gustavo Martins, Lucas, Thiago Bastos
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Dicionário de tradução amigável das variáveis
FEATURE_TRANSLATION = {
    "Age": "Idade (anos)",
    "BMI": "Índice de Massa Corporal (IMC)",
    "Exercise_Frequency": "Exercício Físico (dias/sem)",
    "Diet_Quality": "Qualidade da Dieta (0-100)",
    "Sleep_Hours": "Horas de Sono (h/dia)",
    "Smoking_Status": "Status Tabagista (0=Não, 1=Sim)",
    "Alcohol_Consumption": "Consumo de Álcool (doses/sem)",
    "Health_Score": "Escore de Saúde (0-100 pts)",
}


def plot_eda_distributions(df: pd.DataFrame, col: str = "Health_Score") -> go.Figure:
    """
    Gera um painel com Histograma (com curva de densidade) e Boxplot
    para análise da distribuição da variável selecionada.
    """
    col_name = FEATURE_TRANSLATION.get(col, col)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Histograma de Distribuição: {col_name}",
            f"Boxplot & Dispersão de Outliers: {col_name}",
        ],
        horizontal_spacing=0.12,
    )

    # Histograma
    fig.add_trace(
        go.Histogram(
            x=df[col],
            nbinsx=30,
            name="Frequência",
            marker_color="#1E88E5",
            opacity=0.8,
            hovertemplate="Faixa: %{x}<br>Contagem: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Média e Mediana como linhas verticais
    media = df[col].mean()
    mediana = df[col].median()

    fig.add_vline(
        x=media,
        line_dash="dash",
        line_color="#E53935",
        annotation_text=f"Média: {media:.1f}",
        annotation_position="top right",
        row=1,
        col=1,
    )
    fig.add_vline(
        x=mediana,
        line_dash="dot",
        line_color="#43A047",
        annotation_text=f"Mediana: {mediana:.1f}",
        annotation_position="bottom right",
        row=1,
        col=1,
    )

    # Boxplot
    fig.add_trace(
        go.Box(
            y=df[col],
            name=col_name,
            marker_color="#3949AB",
            boxpoints="outliers",
            jitter=0.3,
            pointpos=-1.8,
            hovertemplate="Valor: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=420,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(title_text=col_name, row=1, col=1)
    fig.update_yaxes(title_text="Frequência", row=1, col=1)
    fig.update_yaxes(title_text=col_name, row=1, col=2)

    return fig


def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap de correlação de Pearson interativo demonstrando
    as relações lineares e a ausência de multicolinearidade entre preditores.
    """
    corr = df.corr(method="pearson")

    # Mapear nomes para rótulos legíveis
    labels = [FEATURE_TRANSLATION.get(c, c) for c in corr.columns]

    fig = px.imshow(
        corr.values,
        x=labels,
        y=labels,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title="Matriz de Correlação Linear de Pearson (Ausência de Multicolinearidade)",
        aspect="auto",
    )

    fig.update_layout(
        template="plotly_white",
        height=540,
        margin=dict(l=60, r=60, t=60, b=60),
        xaxis_tickangle=-30,
        coloraxis_colorbar=dict(title="Correlação (r)"),
    )

    return fig


def plot_real_vs_predicted(y_test: pd.Series, predictions_dict: dict) -> go.Figure:
    """
    Scatter plot interativo comparando 'Valores Reais vs. Preditos' com linha ideal de 45°
    e seletores interativos na legenda para ligar/desligar cada modelo.
    """
    fig = go.Figure()

    # Linha ideal de 45 graus (y = x)
    min_val = min(y_test.min(), min(np.min(p) for p in predictions_dict.values())) - 2
    max_val = max(y_test.max(), max(np.max(p) for p in predictions_dict.values())) + 2

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ajuste Ideal (45°: Real = Previsto)",
            line=dict(color="#D32F2F", width=2.5, dash="dash"),
            hoverinfo="none",
        )
    )

    cores = {
        "Dummy Regressor": "#9E9E9E",
        "Regressão Linear Múltipla": "#1976D2",
        "Support Vector Regressor (SVR)": "#7B1FA2",
        "Random Forest Regressor": "#388E3C",
    }

    simbolos = {
        "Dummy Regressor": "x",
        "Regressão Linear Múltipla": "circle",
        "Support Vector Regressor (SVR)": "diamond",
        "Random Forest Regressor": "square",
    }

    for nome, y_pred in predictions_dict.items():
        cor = cores.get(nome, "#555555")
        simb = simbolos.get(nome, "circle")

        # Dummy regressor visível como 'legendonly' para não poluir
        visible = "legendonly" if nome == "Dummy Regressor" else True

        fig.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred,
                mode="markers",
                name=nome,
                visible=visible,
                marker=dict(color=cor, size=7, opacity=0.75, symbol=simb),
                hovertemplate=(
                    f"<b>{nome}</b><br>"
                    + "Real: %{x:.2f} pts<br>"
                    + "Previsto: %{y:.2f} pts<br>"
                    + "Erro: %{customdata:.2f} pts<extra></extra>"
                ),
                customdata=np.abs(y_test - y_pred),
            )
        )

    fig.update_layout(
        title="Valores Reais vs. Valores Preditos (Clique na legenda para isolar modelos)",
        xaxis_title="Health Score Real (Teste)",
        yaxis_title="Health Score Predito",
        template="plotly_white",
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig


def plot_residuals_diagnostics(
    y_test: pd.Series, y_pred: np.ndarray, model_name: str = "Regressão Linear Múltipla"
) -> go.Figure:
    """
    Scatter plot dos Resíduos vs. Preditos acompanhado do gráfico de Violino
    para validação da homocedasticidade e aleatoriedade/normalidade dos resíduos.
    """
    residuos = y_test - y_pred

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=[
            f"Resíduos vs. Valores Preditos ({model_name})",
            "Distribuição dos Erros (Violin & Box)",
        ],
        horizontal_spacing=0.1,
    )

    # 1. Resíduos vs. Preditos
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuos,
            mode="markers",
            marker=dict(
                color=residuos,
                colorscale="Tealrose",
                showscale=False,
                size=8,
                opacity=0.8,
            ),
            hovertemplate="Predito: %{x:.2f}<br>Resíduo: %{y:.2f}<extra></extra>",
            name="Resíduos",
        ),
        row=1,
        col=1,
    )

    # Linha zero de referência (erro zero)
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#E53935",
        line_width=2,
        row=1,
        col=1,
        annotation_text="Erro Zero (Resíduo = 0)",
        annotation_position="top right",
    )

    # 2. Gráfico de Violino da Distribuição dos Resíduos
    fig.add_trace(
        go.Violin(
            y=residuos,
            box_visible=True,
            line_color="#00897B",
            meanline_visible=True,
            fillcolor="#80CBC4",
            opacity=0.7,
            name="Distribuição",
            hovertemplate="Resíduo: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=450,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(title_text="Valores Preditos (Health Score)", row=1, col=1)
    fig.update_yaxes(title_text="Resíduo (Real - Predito)", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude do Resíduo", row=1, col=2)

    return fig


def plot_feature_importance(df_importance: pd.DataFrame) -> go.Figure:
    """
    Gráfico de barras horizontais destacando a Importância das Features (Random Forest)
    e os Coeficientes Beta Padronizados (Regressão Linear Múltipla).
    Destaca Qualidade da Dieta (~50.4%) e IMC (~21.2%).
    """
    # Ordenar por importância do RF
    df_sorted = df_importance.sort_values(by="Importância RF (%)", ascending=True).copy()
    
    # Nomes traduzidos
    df_sorted["Feature_Label"] = df_sorted["Feature"].apply(
        lambda x: FEATURE_TRANSLATION.get(x, x)
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Importância das Variáveis (Random Forest)",
            "Coeficientes Angulares Padronizados Beta (Regressão Linear)",
        ],
        horizontal_spacing=0.15,
    )

    # Cores de destaque para Diet Quality e BMI
    rf_colors = [
        "#00C853" if "Dieta" in lbl else "#FF6D00" if "IMC" in lbl else "#0288D1"
        for lbl in df_sorted["Feature_Label"]
    ]

    fig.add_trace(
        go.Bar(
            y=df_sorted["Feature_Label"],
            x=df_sorted["Importância RF (%)"],
            orientation="h",
            marker=dict(color=rf_colors),
            text=[f"{v:.1f}%" for v in df_sorted["Importância RF (%)"]],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Importância: %{x:.2f}%<extra></extra>",
            name="Random Forest (%)",
        ),
        row=1,
        col=1,
    )

    # Cores para Coeficientes Beta (Verde = Positivo, Vermelho = Negativo)
    lr_colors = [
        "#2E7D32" if b > 0 else "#C62828" for b in df_sorted["Coeficiente Beta (RLM)"]
    ]

    fig.add_trace(
        go.Bar(
            y=df_sorted["Feature_Label"],
            x=df_sorted["Coeficiente Beta (RLM)"],
            orientation="h",
            marker=dict(color=lr_colors),
            text=[f"{b:+.2f}" for b in df_sorted["Coeficiente Beta (RLM)"]],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Coeficiente Beta: %{x:+.4f}<extra></extra>",
            name="Beta Padronizado",
        ),
        row=1,
        col=2,
    )

    # Linha zero no gráfico de betas
    fig.add_vline(x=0, line_dash="solid", line_color="#333333", line_width=1, row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=450,
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=40),
    )
    fig.update_xaxes(title_text="Importância Relativa (%)", row=1, col=1)
    fig.update_xaxes(title_text="Impacto no Health Score (Desvios-Padrão)", row=1, col=2)

    return fig
