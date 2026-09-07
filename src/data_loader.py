import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NOME_ARQUIVO_SERIE = "ec2_cpu_utilization_24ae8d.csv"
CHAVE_LABELS = f"realAWSCloudwatch/{NOME_ARQUIVO_SERIE}"


def carregar_serie() -> pd.DataFrame:
    """Carrega a série de CPU utilization, indexada por timestamp, frequência de 5min."""
    caminho = DATA_DIR / "raw" / NOME_ARQUIVO_SERIE
    df = pd.read_csv(caminho, parse_dates=["timestamp"])
    df = df.set_index("timestamp").asfreq("5min")
    return df


def carregar_janelas_anomalia() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Carrega as janelas de anomalia real (rótulos do NAB) para a série escolhida."""
    caminho = DATA_DIR / "labels" / "combined_windows.json"
    with open(caminho) as f:
        labels = json.load(f)

    windows = labels[CHAVE_LABELS]
    return [(pd.Timestamp(inicio), pd.Timestamp(fim)) for inicio, fim in windows]


def split_treino_teste(df: pd.DataFrame, proporcao_treino: float = 0.8):
    """Divide a série em treino/teste respeitando a ordem cronológica."""
    tamanho_treino = int(len(df) * proporcao_treino)
    treino = df.iloc[:tamanho_treino].copy()
    teste = df.iloc[tamanho_treino:].copy()
    return treino, teste