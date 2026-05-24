"""
=============================================================
  Visualizações — Detecção de Anomalias em Transações
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# Importa o detector
import sys
sys.path.insert(0, str(Path(__file__).parent))
from anomaly_detector import (
    gerar_transacoes,
    preprocessar,
    detectar_isolation_forest,
    detectar_lof,
    detectar_zscore,
)

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PALETA = {
    "normal":  "#2196F3",   # azul
    "fraude":  "#F44336",   # vermelho
    "detectado": "#FF9800", # laranja
}

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi":      120,
})


# ─────────────────────────────────────────────
#  1. DISTRIBUIÇÃO DAS FEATURES
# ─────────────────────────────────────────────

def plot_distribuicoes(df: pd.DataFrame) -> None:
    features = ["valor", "hora", "frequencia_dia", "distancia_km"]
    labels   = ["Valor (R$)", "Hora do dia", "Frequência/dia", "Distância (km)"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Distribuição das Features por Classe", fontsize=14, weight="bold", y=1.01)

    for ax, feat, lbl in zip(axes.flat, features, labels):
        for classe, cor, rotulo in [
            (0, PALETA["normal"], "Normal"),
            (1, PALETA["fraude"], "Fraude"),
        ]:
            dados = df.loc[df["is_fraude_real"] == classe, feat]
            ax.hist(dados, bins=30, alpha=0.6, color=cor, label=rotulo, edgecolor="white")
        ax.set_title(lbl, weight="bold")
        ax.set_ylabel("Contagem")
        ax.legend(fontsize=8)

    plt.tight_layout()
    caminho = OUTPUT_DIR / "01_distribuicoes.png"
    plt.savefig(caminho, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {caminho}")


# ─────────────────────────────────────────────
#  2. SCATTER: VALOR × DISTÂNCIA
# ─────────────────────────────────────────────

def plot_scatter_principal(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    modelos = ["pred_isolation_forest", "pred_lof", "pred_zscore"]
    nomes   = ["Isolation Forest", "Local Outlier Factor", "Z-Score"]

    for ax, col, nome in zip(axes, modelos, nomes):
        normais = df[df[col] == 1]
        fraudes = df[df[col] == -1]

        ax.scatter(normais["valor"], normais["distancia_km"],
                   c=PALETA["normal"], alpha=0.4, s=20, label="Normal")
        ax.scatter(fraudes["valor"], fraudes["distancia_km"],
                   c=PALETA["detectado"], alpha=0.8, s=50,
                   edgecolors="black", linewidths=0.4, label="Anomalia detectada")

        # marca as fraudes reais que foram perdidas
        perdidas = df[(df[col] == 1) & (df["is_fraude_real"] == 1)]
        ax.scatter(perdidas["valor"], perdidas["distancia_km"],
                   c=PALETA["fraude"], marker="x", s=80,
                   linewidths=1.5, label="Fraude não detectada")

        ax.set_title(nome, weight="bold")
        ax.set_xlabel("Valor (R$)")

    axes[0].set_ylabel("Distância (km)")
    axes[0].legend(fontsize=7, loc="upper right")

    fig.suptitle("Detecção de Anomalias: Valor × Distância", fontsize=13, weight="bold")
    plt.tight_layout()
    caminho = OUTPUT_DIR / "02_scatter_modelos.png"
    plt.savefig(caminho, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {caminho}")


# ─────────────────────────────────────────────
#  3. MATRIZ DE CONFUSÃO (SIMPLIFICADA)
# ─────────────────────────────────────────────

def _matriz_confusao(y_real, y_pred):
    pred_bin = (y_pred == -1).astype(int)
    real_bin  = y_real.astype(int)
    tp = ((pred_bin == 1) & (real_bin == 1)).sum()
    fp = ((pred_bin == 1) & (real_bin == 0)).sum()
    fn = ((pred_bin == 0) & (real_bin == 1)).sum()
    tn = ((pred_bin == 0) & (real_bin == 0)).sum()
    return np.array([[tn, fp], [fn, tp]])


def plot_matrizes_confusao(df: pd.DataFrame) -> None:
    modelos = {
        "Isolation Forest":    df["pred_isolation_forest"].values,
        "Local Outlier Factor": df["pred_lof"].values,
        "Z-Score":             df["pred_zscore"].values,
        "Ensemble":            df["pred_ensemble"].values,
    }
    y_real = df["is_fraude_real"].values

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Matrizes de Confusão por Modelo", fontsize=13, weight="bold")

    for ax, (nome, preds) in zip(axes, modelos.items()):
        mat = _matriz_confusao(y_real, preds)
        sns.heatmap(mat, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Fraude"],
                    yticklabels=["Normal", "Fraude"],
                    ax=ax, cbar=False, linewidths=0.5)
        ax.set_title(nome, weight="bold", fontsize=10)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")

    plt.tight_layout()
    caminho = OUTPUT_DIR / "03_matrizes_confusao.png"
    plt.savefig(caminho, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {caminho}")


# ─────────────────────────────────────────────
#  4. COMPARAÇÃO F1-SCORE
# ─────────────────────────────────────────────

def plot_comparacao_f1(df: pd.DataFrame) -> None:
    from anomaly_detector import calcular_metricas

    modelos = {
        "Isolation\nForest":  df["pred_isolation_forest"].values,
        "Local Outlier\nFactor": df["pred_lof"].values,
        "Z-Score":            df["pred_zscore"].values,
        "Ensemble\n(Votação)": df["pred_ensemble"].values,
    }
    y_real = df["is_fraude_real"].values

    nomes, precisoes, recalls, f1s = [], [], [], []
    for nome, preds in modelos.items():
        m = calcular_metricas(y_real, preds)
        nomes.append(nome)
        precisoes.append(m["Precisão"])
        recalls.append(m["Recall"])
        f1s.append(m["F1-Score"])

    x = np.arange(len(nomes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width, precisoes, width, label="Precisão",
                   color="#42A5F5", edgecolor="white")
    bars2 = ax.bar(x,          recalls,   width, label="Recall",
                   color="#EF5350", edgecolor="white")
    bars3 = ax.bar(x + width,  f1s,       width, label="F1-Score",
                   color="#66BB6A", edgecolor="white")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(nomes, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Comparação de Desempenho dos Modelos", weight="bold", fontsize=13)
    ax.legend(loc="upper right")

    plt.tight_layout()
    caminho = OUTPUT_DIR / "04_comparacao_modelos.png"
    plt.savefig(caminho, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {caminho}")


# ─────────────────────────────────────────────
#  5. ANÁLISE TEMPORAL (hora × valor)
# ─────────────────────────────────────────────

def plot_analise_temporal(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    normais = df[df["pred_ensemble"] == 1]
    anomalias = df[df["pred_ensemble"] == -1]

    ax.scatter(normais["hora"], normais["valor"], c=PALETA["normal"],
               alpha=0.3, s=15, label="Normal")
    ax.scatter(anomalias["hora"], anomalias["valor"], c=PALETA["detectado"],
               alpha=0.9, s=60, edgecolors="black", linewidths=0.5,
               label="Anomalia (Ensemble)", zorder=5)

    # destaque faixas de risco
    ax.axvspan(0, 5.5, alpha=0.05, color="red", label="Horário de risco (0–5h)")

    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Valor da transação (R$)")
    ax.set_title("Padrão Temporal das Anomalias", weight="bold", fontsize=13)
    ax.set_xticks(range(0, 24))
    ax.legend()

    plt.tight_layout()
    caminho = OUTPUT_DIR / "05_analise_temporal.png"
    plt.savefig(caminho, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {caminho}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def gerar_todos_os_graficos(df: pd.DataFrame = None) -> None:
    if df is None:
        from anomaly_detector import executar_pipeline
        df = executar_pipeline(salvar_csv=False)

    print("\n[📊] Gerando visualizações...")
    plot_distribuicoes(df)
    plot_scatter_principal(df)
    plot_matrizes_confusao(df)
    plot_comparacao_f1(df)
    plot_analise_temporal(df)
    print(f"\n✅ Gráficos salvos em '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    gerar_todos_os_graficos()
