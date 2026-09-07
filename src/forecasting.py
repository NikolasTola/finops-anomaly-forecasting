import numpy as np
import pandas as pd

PERIODO_SAZONAL = 288  # 1 dia = 288 pontos de 5 minutos


def calcular_perfil_sazonal(treino: pd.DataFrame) -> pd.Series:
    """Calcula o valor médio esperado para cada posição do ciclo diário, a partir do treino."""
    posicao = np.arange(len(treino)) % PERIODO_SAZONAL
    valores = pd.Series(treino["value"].values, index=posicao)
    return valores.groupby(level=0).mean()


def prever(perfil_sazonal: pd.Series, teste: pd.DataFrame, offset: int) -> pd.Series:
    """Aplica o perfil sazonal ao período de teste, mantendo a fase correta."""
    posicao_global = np.arange(offset, offset + len(teste)) % PERIODO_SAZONAL
    esperado = perfil_sazonal.reindex(posicao_global).values
    return pd.Series(esperado, index=teste.index)


def calcular_residuo(teste: pd.DataFrame, previsto: pd.Series) -> pd.Series:
    return pd.Series(teste["value"].values - previsto.values, index=teste.index)
