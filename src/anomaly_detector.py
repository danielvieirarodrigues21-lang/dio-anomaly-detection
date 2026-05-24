"""
=============================================================
  Detecção de Anomalias em Transações Financeiras
  Projeto DIO - Bootcamp Afya / Accenture Dados
=============================================================
  Algoritmos: Isolation Forest | Local Outlier Factor | Z-Score
  Autores: DIO + Estudante
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  GERAÇÃO DE DADOS SINTÉTICOS
# ─────────────────────────────────────────────

def gerar_transacoes(n_normais: int = 950, n_fraudes: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Gera um dataset sintético de transações bancárias com fraudes.

    Parâmetros
    ----------
    n_normais : int
        Número de transações normais (default 950).
    n_fraudes : int
        Número de transações fraudulentas (default 50).
    seed : int
        Semente aleatória para reprodutibilidade.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas: valor, hora, frequencia_dia,
        distancia_km, is_fraude_real.
    """
    rng = np.random.default_rng(seed)

    # ── Transações NORMAIS ──────────────────────────────────
    normais = pd.DataFrame({
        "valor":           rng.normal(loc=150, scale=80,  size=n_normais).clip(1),
        "hora":            rng.integers(6, 23,             size=n_normais),
        "frequencia_dia":  rng.integers(1, 5,              size=n_normais),
        "distancia_km":    rng.normal(loc=10, scale=5,    size=n_normais).clip(0),
        "is_fraude_real":  np.zeros(n_normais, dtype=int),
    })

    # ── Transações FRAUDULENTAS ─────────────────────────────
    fraudes = pd.DataFrame({
        "valor":           rng.choice(
                               np.concatenate([
                                   rng.normal(3000, 500, n_fraudes // 2),   # valor alto
                                   rng.uniform(0.01, 1.0, n_fraudes // 2),  # valor suspeito baixo
                               ])
                           ),
        "hora":            rng.choice([0, 1, 2, 3, 4, 5], size=n_fraudes),  # madrugada
        "frequencia_dia":  rng.integers(15, 40, size=n_fraudes),              # muitas transações
        "distancia_km":    rng.normal(loc=800, scale=200, size=n_fraudes).clip(0),  # longe
        "is_fraude_real":  np.ones(n_fraudes, dtype=int),
    })

    df = pd.concat([normais, fraudes], ignore_index=True).sample(frac=1, random_state=seed)
    df.reset_index(drop=True, inplace=True)
    df.index.name = "id_transacao"
    return df


# ─────────────────────────────────────────────
#  PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────

FEATURES = ["valor", "hora", "frequencia_dia", "distancia_km"]


def preprocessar(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Normaliza as features e retorna o scaler aplicado."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    return X, scaler


# ─────────────────────────────────────────────
#  MODELOS DE DETECÇÃO
# ─────────────────────────────────────────────

def detectar_isolation_forest(X: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    """
    Isolation Forest: isola anomalias construindo árvores aleatórias.
    Pontos isolados mais rápido → maior score de anomalia.

    Retorna array com 1 (normal) ou -1 (anomalia).
    """
    modelo = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    return modelo.fit_predict(X)


def detectar_lof(X: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    """
    Local Outlier Factor: compara a densidade local de cada ponto
    com seus vizinhos. Score alto → anomalia.

    n_neighbors=50 garante que cada ponto compare com vizinhança
    ampla o suficiente para capturar os clusters de fraude.

    Retorna array com 1 (normal) ou -1 (anomalia).
    """
    modelo = LocalOutlierFactor(
        n_neighbors=50,
        contamination=contamination,
    )
    return modelo.fit_predict(X)


def detectar_zscore(df: pd.DataFrame, threshold: float = 3.0) -> np.ndarray:
    """
    Z-Score: detecta outliers univariados em cada feature.
    Marcado como anomalia se qualquer feature > threshold desvios padrão.

    Retorna array com 1 (normal) ou -1 (anomalia).
    """
    z_scores = np.abs(stats.zscore(df[FEATURES]))
    anomalia = (z_scores > threshold).any(axis=1)
    return np.where(anomalia, -1, 1)


# ─────────────────────────────────────────────
#  MÉTRICAS DE AVALIAÇÃO
# ─────────────────────────────────────────────

def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula Precision, Recall e F1-Score para detecção de fraudes.

    y_real: 1 = fraude, 0 = normal
    y_pred: -1 = anomalia detectada, 1 = normal detectado
    """
    # converte previsões para 0/1 (fraude = 1)
    pred_bin = (y_pred == -1).astype(int)
    real_bin = y_real.astype(int)

    vp = ((pred_bin == 1) & (real_bin == 1)).sum()
    fp = ((pred_bin == 1) & (real_bin == 0)).sum()
    fn = ((pred_bin == 0) & (real_bin == 1)).sum()

    precisao = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    recall   = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    f1       = (2 * precisao * recall / (precisao + recall)
                if (precisao + recall) > 0 else 0.0)

    return {
        "VP (Fraudes detectadas)":  int(vp),
        "FP (Falsos alarmes)":      int(fp),
        "FN (Fraudes perdidas)":    int(fn),
        "Precisão":                 round(precisao, 4),
        "Recall":                   round(recall,   4),
        "F1-Score":                 round(f1,       4),
    }


# ─────────────────────────────────────────────
#  PIPELINE COMPLETO
# ─────────────────────────────────────────────

def executar_pipeline(n_normais=950, n_fraudes=50, salvar_csv=True) -> pd.DataFrame:
    """
    Executa o pipeline completo de detecção de anomalias.

    1. Gera dados sintéticos
    2. Pré-processa (normalização)
    3. Aplica os três modelos
    4. Calcula métricas
    5. Exibe relatório no console
    6. Retorna DataFrame com resultados

    Retorna
    -------
    pd.DataFrame com as previsões de cada modelo.
    """
    print("=" * 60)
    print("  DIO — Detecção de Anomalias em Transações Financeiras")
    print("=" * 60)

    # 1. Dados
    print("\n[1/4] Gerando dataset de transações...")
    df = gerar_transacoes(n_normais, n_fraudes)
    print(f"      Total: {len(df)} transações | {n_fraudes} fraudes reais ({n_fraudes/len(df)*100:.1f}%)")

    # 2. Pré-processamento
    print("[2/4] Pré-processando features...")
    X, _ = preprocessar(df)

    # 3. Modelos
    print("[3/4] Aplicando modelos de detecção...")
    df["pred_isolation_forest"] = detectar_isolation_forest(X)
    df["pred_lof"]              = detectar_lof(X)
    df["pred_zscore"]           = detectar_zscore(df)

    # Votação majoritária (ensemble): anomalia se pelo menos 2 dos 3 concordam
    votos = (
        (df["pred_isolation_forest"] == -1).astype(int)
        + (df["pred_lof"]            == -1).astype(int)
        + (df["pred_zscore"]         == -1).astype(int)
    )
    df["pred_ensemble"] = np.where(votos >= 2, -1, 1)

    # 4. Métricas
    print("[4/4] Calculando métricas...\n")
    y_real = df["is_fraude_real"].values

    modelos = {
        "Isolation Forest": df["pred_isolation_forest"].values,
        "Local Outlier Factor": df["pred_lof"].values,
        "Z-Score": df["pred_zscore"].values,
        "Ensemble (Votação)": df["pred_ensemble"].values,
    }

    resultados = {}
    for nome, preds in modelos.items():
        resultados[nome] = calcular_metricas(y_real, preds)

    # ── Relatório ────────────────────────────────────────────
    print("─" * 60)
    print(f"{'Modelo':<25} {'Precisão':>9} {'Recall':>8} {'F1':>8}")
    print("─" * 60)
    for nome, m in resultados.items():
        print(f"  {nome:<23} {m['Precisão']:>9.2%} {m['Recall']:>8.2%} {m['F1-Score']:>8.2%}")
    print("─" * 60)

    print("\nDetalhes — Ensemble (Votação Majoritária):")
    for k, v in resultados["Ensemble (Votação)"].items():
        print(f"  {k}: {v}")

    if salvar_csv:
        caminho = "data/transacoes_com_previsoes.csv"
        df.to_csv(caminho, index=True)
        print(f"\n✅ Resultados salvos em '{caminho}'")

    print("\n" + "=" * 60)
    return df


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df_resultado = executar_pipeline()
