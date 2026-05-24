"""
=============================================================
  MAIN — Detecção de Anomalias em Transações Financeiras
  Execute este arquivo para rodar o projeto completo.
=============================================================

  Uso:
      python main.py

  Saídas:
      data/transacoes_com_previsoes.csv
      reports/figures/*.png
=============================================================
"""

import sys
from pathlib import Path

# Garante imports mesmo quando executado da raiz do projeto
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Cria diretórios necessários automaticamente
for pasta in ["data", "reports/figures"]:
    Path(pasta).mkdir(parents=True, exist_ok=True)

from anomaly_detector import executar_pipeline
from visualizacoes    import gerar_todos_os_graficos


def main():
    # 1. Pipeline de ML
    df = executar_pipeline(n_normais=950, n_fraudes=50, salvar_csv=True)

    # 2. Visualizações
    gerar_todos_os_graficos(df)

    print("\n🚀 Projeto concluído! Consulte:")
    print("   • data/transacoes_com_previsoes.csv")
    print("   • reports/figures/")


if __name__ == "__main__":
    main()
