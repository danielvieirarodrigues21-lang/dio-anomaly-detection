# 🔍 Detecção de Anomalias em Transações Financeiras

> **Projeto DIO — Bootcamp Afya / Accenture Dados**  
> Detector de fraudes bancárias com Python e Machine Learning, inspirado nos sistemas antifraude do Nubank.

---

## 🎯 Objetivo

Construir um pipeline de **detecção de anomalias em transações financeiras** utilizando algoritmos de Machine Learning não-supervisionado, simulando o funcionamento de sistemas antifraude reais.

---

## 🧠 Algoritmos Implementados

| Algoritmo | Abordagem |
|---|---|
| **Isolation Forest** | Isola anomalias com árvores de decisão aleatórias — pontos isolados mais rápido são anomalias |
| **Local Outlier Factor (LOF)** | Compara a densidade local de cada ponto com seus vizinhos (n_neighbors=50) |
| **Z-Score** | Detecta outliers univariados com base no desvio padrão de cada feature |
| **Ensemble (Votação)** | Marca como fraude se ≥ 2 dos 3 modelos concordam |

---

## 📁 Estrutura do Projeto

```
dio-anomaly-detection/
│
├── main.py                          ← ▶ Execute este para rodar tudo
│
├── src/
│   ├── anomaly_detector.py          ← Pipeline principal de ML
│   └── visualizacoes.py             ← Geração de gráficos
│
├── notebooks/
│   └── deteccao_anomalias.ipynb     ← Notebook explicativo completo
│
├── data/                            ← Criado automaticamente
│   └── transacoes_com_previsoes.csv
│
├── reports/
│   └── figures/                     ← Criado automaticamente
│       ├── 01_distribuicoes.png
│       ├── 02_scatter_modelos.png
│       ├── 03_matrizes_confusao.png
│       ├── 04_comparacao_modelos.png
│       └── 05_analise_temporal.png
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalação e Execução

### 1. Clone ou baixe o projeto

```bash
git clone https://github.com/SEU-USUARIO/dio-anomaly-detection.git
cd dio-anomaly-detection
```

> **Dica:** se baixou o `.zip` pela DIO, extraia e entre na pasta extraída.

### 2. (Opcional) Crie um ambiente virtual

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute o projeto completo

```bash
python main.py
```

> Os diretórios `data/` e `reports/figures/` são criados automaticamente na primeira execução.

### 5. (Opcional) Explore o Notebook

```bash
jupyter notebook notebooks/deteccao_anomalias.ipynb
```

---

## 📊 Features do Dataset Sintético

| Feature | Transação Normal | Fraude |
|---|---|---|
| **valor** | R$ 50 – R$ 400 (média R$ 150) | < R$ 1 ou > R$ 2.500 |
| **hora** | 6h – 22h (horário comercial) | 0h – 5h (madrugada) |
| **frequencia_dia** | 1 – 5 transações/dia | 15 – 40 transações/dia |
| **distancia_km** | 0 – 25 km (média 10 km) | 500+ km (média 800 km) |

---

## 📈 Exemplo de Saída

```
============================================================
  DIO — Detecção de Anomalias em Transações Financeiras
============================================================

[1/4] Gerando dataset de transações...
      Total: 1000 transações | 50 fraudes reais (5.0%)
[2/4] Pré-processando features...
[3/4] Aplicando modelos de detecção...
[4/4] Calculando métricas...

────────────────────────────────────────────────────────────
Modelo                    Precisão   Recall       F1
────────────────────────────────────────────────────────────
  Isolation Forest         100.00%  100.00%  100.00%
  Local Outlier Factor     100.00%  100.00%  100.00%
  Z-Score                  100.00%  100.00%  100.00%
  Ensemble (Votação)       100.00%  100.00%  100.00%
────────────────────────────────────────────────────────────
```

---

## 🛠️ Tecnologias

- **Python 3.10+**
- `scikit-learn` — Isolation Forest, LOF, StandardScaler
- `scipy` — Z-Score
- `pandas` / `numpy` — Manipulação de dados
- `matplotlib` / `seaborn` — Visualizações
- `jupyter` — Notebook interativo

---

## 📚 Conceitos Abordados

- Detecção de anomalias **não-supervisionada**
- **Isolation Forest** — isolamento via árvores aleatórias
- **LOF** — densidade de vizinhança local
- **Z-Score** — desvio padrão univariado
- **Ensemble Learning** — votação majoritária
- **Métricas de avaliação**: Precisão, Recall, F1-Score
- Análise exploratória e visualização de dados

---

## 🤝 Créditos

Projeto desenvolvido como parte do **Bootcamp DIO** (Afya / Accenture Dados).  
Inspirado em sistemas reais de detecção de fraudes financeiras.
