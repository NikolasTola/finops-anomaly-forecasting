import pandas as pd

JANELA_PADRAO = 24  # 2 horas (24 pontos de 5 minutos)
PERCENTIL_LIMIAR = 0.99


def calcular_sinal(residuo: pd.Series, janela: int = JANELA_PADRAO) -> pd.Series:
    return residuo.abs().rolling(window=janela, min_periods=janela).std()


def calcular_limiar(sinal: pd.Series, mascara_normal: pd.Series, percentil: float = PERCENTIL_LIMIAR) -> float:
    sinal_normal = sinal[mascara_normal].dropna()
    return sinal_normal.quantile(percentil)


def detectar(sinal: pd.Series, limiar: float) -> pd.Series:
    return (sinal > limiar).fillna(False)


def agrupar_alarmes(deteccoes: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Agrupa pontos de detecção consecutivos em eventos (alarmes)."""
    mudanca = deteccoes.astype(int).diff().fillna(0)
    inicios = deteccoes.index[mudanca == 1].tolist()
    fins = deteccoes.index[mudanca == -1].tolist()

    if deteccoes.iloc[0]:
        inicios = [deteccoes.index[0]] + inicios
    if len(inicios) > len(fins):
        fins.append(deteccoes.index[-1])

    return list(zip(inicios, fins))


def _sobrepoe(a_inicio, a_fim, b_inicio, b_fim) -> bool:
    return a_inicio <= b_fim and b_inicio <= a_fim


def avaliar_por_janela(alarmes: list, janelas_reais: list) -> dict:
    """Avaliação estilo NAB: um alarme conta como acerto se tocar qualquer parte de uma janela real."""
    alarmes_verdadeiros = [
        a for a in alarmes
        if any(_sobrepoe(a[0], a[1], w[0], w[1]) for w in janelas_reais)
    ]
    alarmes_falsos = [a for a in alarmes if a not in alarmes_verdadeiros]
    janelas_detectadas = [
        w for w in janelas_reais
        if any(_sobrepoe(a[0], a[1], w[0], w[1]) for a in alarmes)
    ]

    precisao = len(alarmes_verdadeiros) / len(alarmes) if alarmes else 0.0
    recall = len(janelas_detectadas) / len(janelas_reais) if janelas_reais else 0.0
    f1 = (2 * precisao * recall / (precisao + recall)) if (precisao + recall) > 0 else 0.0

    return {
        "alarmes_verdadeiros": alarmes_verdadeiros,
        "alarmes_falsos": alarmes_falsos,
        "janelas_detectadas": janelas_detectadas,
        "precisao": precisao,
        "recall": recall,
        "f1": f1,
    }
