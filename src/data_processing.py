"""
Módulo de Processamento e Higienização de Dados
Projeto: O Desafio do Oráculo Preditivo | Health Score
FATEC Indaiatuba - Prof. Medina
Autores: Gustavo Bravo, Gustavo Martins, Lucas, Thiago Bastos
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@st.cache_data
def load_raw_data(filepath: str = "data/Health_LifeStyle-DS.csv") -> pd.DataFrame:
    """
    Carrega o dataset bruto de estilo de vida e saúde.
    Utiliza cache do Streamlit para otimizar tempo de carregamento.
    Busca preferencialmente no caminho indicado com fallback para caminhos alternativos.
    """
    if not os.path.exists(filepath):
        # Fallback para diretório base do projeto
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt_path = os.path.join(base_dir, "data", "Health_LifeStyle-DS.csv")
        if os.path.exists(alt_path):
            filepath = alt_path
        elif os.path.exists("data/Health_LifeStyle-DS.csv"):
            filepath = "data/Health_LifeStyle-DS.csv"
        elif os.path.exists("Health_LifeStyle-DS.csv"):
            filepath = "Health_LifeStyle-DS.csv"
    return pd.read_csv(filepath)


@st.cache_data
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Realiza a higienização dos dados reais aplicando o método .clip() com base
    em critérios de domínio fisiológico, físico e metodológico:
    - Alcohol_Consumption: lower=0.0 (corrige os 70 valores negativos para 0.0)
    - Diet_Quality: lower=0.0, upper=100.0 (corrige os 18 valores > 100 para o teto da escala)
    - Age: lower=18.0 (corrige idades incongruentes com hábitos de adultos para 18 anos)
    - BMI: lower=15.0, upper=40.0 (corrige valores incongruentes com a vida biológica < 15)
    - Sleep_Hours: lower=3.0, upper=11.0 (preserva a variação fisiológica plausível)

    Retorna:
        tuple[pd.DataFrame, int]: (DataFrame higienizado, total de registros modificados)
    """
    df_clean = df.copy()

    # 1. Consumo de Álcool (Piso físico em 0.0 - não existe consumo negativo)
    df_clean["Alcohol_Consumption"] = df_clean["Alcohol_Consumption"].clip(lower=0.0)

    # 2. Qualidade da Dieta (Teto de escala em 100.0 - índice delimitado de 0 a 100)
    df_clean["Diet_Quality"] = df_clean["Diet_Quality"].clip(lower=0.0, upper=100.0)

    # 3. Idade (Piso do público-alvo em 18 anos - estudo voltado para adultos)
    df_clean["Age"] = df_clean["Age"].clip(lower=18.0)

    # 4. IMC / BMI (Piso fisiológico em 15.0 e teto em 40.0)
    df_clean["BMI"] = df_clean["BMI"].clip(lower=15.0, upper=40.0)

    # 5. Horas de Sono (Intervalo de viabilidade biológica entre 3.0h e 11.0h)
    df_clean["Sleep_Hours"] = df_clean["Sleep_Hours"].clip(lower=3.0, upper=11.0)

    # Identificação dos registros que sofreram qualquer alteração
    linhas_alteradas = df.ne(df_clean).any(axis=1)
    num_modified = int(linhas_alteradas.sum())

    return df_clean, num_modified


def get_modified_rows_comparison(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Gera uma tabela comparativa com amostras dos dados antes e depois da higienização,
    destacando as correções efetuadas.
    """
    mask = df_raw.ne(df_clean).any(axis=1)
    indices_modificados = df_raw[mask].index

    cols_verificadas = ["Alcohol_Consumption", "Diet_Quality", "BMI", "Age", "Sleep_Hours"]
    
    comparativo = []
    for idx in indices_modificados:
        for col in cols_verificadas:
            val_antes = df_raw.loc[idx, col]
            val_depois = df_clean.loc[idx, col]
            if not np.isclose(val_antes, val_depois, atol=1e-5):
                motivo = ""
                if col == "Alcohol_Consumption" and val_antes < 0:
                    motivo = "Consumo negativo corrigido para 0.0"
                elif col == "Diet_Quality" and val_antes > 100:
                    motivo = "Score > 100 ajustado para teto 100.0"
                elif col == "BMI" and val_antes < 15:
                    motivo = "IMC < 15 corrigido para piso fisiológico 15.0"
                elif col == "BMI" and val_antes > 40:
                    motivo = "IMC > 40 ajustado para teto 40.0"
                elif col == "Age" and val_antes < 18:
                    motivo = "Idade < 18 corrigida para piso adulto 18 anos"
                elif col == "Sleep_Hours" and val_antes < 3:
                    motivo = "Sono < 3h ajustado para 3.0h"
                elif col == "Sleep_Hours" and val_antes > 11:
                    motivo = "Sono > 11h ajustado para 11.0h"

                comparativo.append({
                    "ID Paciente": int(idx),
                    "Variável": col,
                    "Valor Original": round(val_antes, 2),
                    "Valor Higienizado": round(val_depois, 2),
                    "Regra Aplicada (.clip)": motivo
                })

    return pd.DataFrame(comparativo)


@st.cache_data
def prepare_train_test_data(
    df_clean: pd.DataFrame,
    target_col: str = "Health_Score",
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray, StandardScaler]:
    """
    Realiza a separação entre variáveis explicativas (X) e alvo (y), divisão
    treino/teste estratificada ou aleatória e aplica o StandardScaler nas variáveis.

    Retorna:
        tuple: (X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler)
    """
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
