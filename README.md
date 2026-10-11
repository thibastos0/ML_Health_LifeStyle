# 🩺 O Desafio do Oráculo Preditivo | Health Score

Aplicação interativa e didática em **Streamlit** desenvolvida para o repositório [`ML_Health_LifeStyle`](https://github.com/thibastos0/ML_Health_LifeStyle), com foco na modelagem preditiva de escores de saúde baseados em biomarcadores e hábitos de vida, voltada para a defesa do projeto acadêmico na **FATEC Indaiatuba**.

---

## 🌐 Aplicação Publicada

O projeto está disponível para acesso online em: [healthlifestyle.streamlit.app](https://healthlifestyle.streamlit.app/).

---

## 👥 Informações Acadêmicas
- **Instituição:** Faculdade de Tecnologia de Indaiatuba (FATEC Indaiatuba)
- **Disciplina:** Laboratório de Machine Learning
- **Docente Orientador:** Prof. Medina
- **Integrantes da Equipe:**
  - Gustavo Bravo
  - Gustavo Martins
  - Lucas
  - Thiago Bastos

---

## 📁 Estrutura do Repositório

```text
ML_Health_LifeStyle/
├── data/
│   └── Health_LifeStyle-DS.csv            # Dataset bruto de hábitos e saúde
├── notebooks/
│   └── P1_Projeto_ML_Health_Lifestyle.ipynb  # Notebook experimental completo (Google Colab / Jupyter)
├── src/
│   ├── __init__.py                         # Pacote Python
│   ├── data_processing.py                 # Ingestão, limpeza com .clip() e split treino/teste
│   ├── models.py                          # Treinamento dos 4 modelos, métricas e predição
│   └── utils.py                           # Visualizações interativas com Plotly
├── app.py                                 # Aplicação interativa em Streamlit
├── README.md                              # Documentação completa do projeto
└── requirements.txt                       # Dependências do projeto
```

---

## 🚀 Como Executar o Projeto Localmente

### 1. Clonar o Repositório e Navegar até a Pasta
```bash
git clone https://github.com/thibastos0/ML_Health_LifeStyle.git
cd ML_Health_LifeStyle
```

### 2. Criar e Ativar o Ambiente Virtual
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux / macOS
# No Windows: .venv\Scripts\activate
```

### 3. Instalar as Dependências
```bash
pip install -r requirements.txt
```

### 4. Executar a Aplicação Streamlit
```bash
streamlit run app.py
```
Acesse a aplicação no navegador em `http://localhost:8501`.

---

## 🖥️ Arquitetura dos Painéis da Aplicação

A aplicação possui layout "wide" e navegação lateral estruturada em 4 módulos principais:

### 📊 1. EDA & Higienização dos Dados
- **KPIs em Tempo Real:** 1.000 pacientes, Health Score médio de 85.5 pontos, 143 registros corrigidos (~14.3%).
- **Tabela Antes vs. Depois:** Demonstração do impacto do método `.clip()` (Winsorization) para eliminação de incongruências físicas (álcool < 0), de escala (dieta > 100) e fisiológicas (IMC < 15).
- **Matriz de Correlação de Pearson:** Comprovação gráfica e analítica da ausência de multicolinearidade entre os preditores ($|r| < 0.15$).
- **Exploração Univariada:** Histograma com curva de densidade e Boxplot interativo para todas as variáveis.

### 📈 2. Comparativo de Performance dos Modelos
- **Métricas Consolidadas:** Comparativo entre Dummy Regressor, Regressão Linear Múltipla, Random Forest Regressor e SVR com $R^2$ de Teste, $R^2$ de Validação Cruzada (5-Fold CV), MAE, MAPE (%), RMSE e Tempo de Treino (s).
- **Dispersão Reais vs. Preditos:** Gráfico com linha de 45° permitindo isolar modelos clicando na legenda.
- **Diagnóstico de Overfitting:** Avaliação do gap de generalização ($R^2_{\text{treino}} - R^2_{\text{teste}}$), detalhando a memorização do Random Forest sem limite de profundidade (`max_depth=None`).

### 🔍 3. Diagnóstico de Resíduos & Explicabilidade (XAI)
- **Homocedasticidade dos Resíduos:** Scatter plot dos resíduos vs. valores preditos com linha de erro zero ($y=0$) e gráfico de violino com boxplot acoplado.
- **Importância das Variáveis (Random Forest):** Destaque para Qualidade da Dieta (~50,4%) e IMC (~21,2%).
- **Coeficientes Padronizados Beta (Regressão Linear):** Identificação dos fatores protetivos ($\beta > 0$) e de risco ($\beta < 0$).

### 🔮 4. Simulador Clínico Preditivo 🚀 (Gran Finale)
- **Formulário de Hábitos Clínicos:** Sliders interativos para Idade, IMC, Dieta, Sono, Exercício, Álcool e Tabagismo.
- **Seletor de Algoritmo:** Alternância entre Regressão Linear Múltipla e Random Forest Regressor.
- **Predição Imediata com Classificação:** Categorização em Excelente (verde), Boa (amarelo), Atenção (laranja) e Crítica (vermelho).
- **Prescrições e Feedback Dinâmico:** Recomendações personalizadas calculando o impacto exato no Health Score caso o paciente melhore a dieta (+15 pts), normalize o IMC (23 kg/m²), cesse o tabagismo ou aumente a prática de exercícios.

---

## 📊 Síntese dos Resultados dos Modelos

| Modelo | $R^2$ (Teste) | $R^2$ (CV 5-Fold) | MAE | MAPE (%) | RMSE | Tempo de Treino (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **0.8305** | **0.8251** | **4.15** | **5.52%** | **5.75** | ~0.4206s |
| **Support Vector Regressor (SVR)** | 0.8215 | 0.8310 | 4.22 | 5.54% | 5.90 | ~0.2023s |
| **Regressão Linear Múltipla** | 0.8162 | 0.8336 | 4.62 | 5.74% | 5.99 | **~0.0021s** |
| **Dummy Regressor (Baseline)** | -0.0000 | -0.0163 | 11.23 | 13.91% | 13.96 | ~0.0037s |

---

## 🎓 Tópicos para Arguição da Banca (Prof. Medina)

1. **Multicolinearidade:**
   - O heatmap de Pearson comprovou ausência de colinearidade entre as 7 variáveis independentes ($|r| < 0.15$), assegurando estabilidade analítica dos coeficientes $\beta$.
2. **Trade-off Viés-Variância (Bias-Variance):**
   - O Random Forest sem restrição de profundidade atingiu $R^2 = 97.10\%$ no treino e $83.05\%$ no teste (Gap de ~14%), evidenciando sobreajuste parcial. Em contraste, a Regressão Linear obteve $R^2$ de $83.36\%$ (treino) e $81.62\%$ (teste), provando alta capacidade de generalização.
3. **Interpretação do MAPE (%) na Área da Saúde:**
   - O MAPE expressa o erro percentual médio relativo. Um MAPE de ~5.5% significa que o erro médio em uma nota de 80 pontos é de apenas 4.4 pontos (94.5% de precisão clínica).
4. **Modelo Recomendado para Produção (Navalha de Ockham):**
   - Para ambientes em tempo real com alta volumetria (milhares de requisições por segundo), a **Regressão Linear Múltipla** é a escolha ideal: oferece inferência por produto escalar vetorial $O(p)$ em microssegundos (200x mais veloz que o Random Forest) com perda mínima de acurácia (81.6% vs 83.1%) e total transparência e explicabilidade clínica.
