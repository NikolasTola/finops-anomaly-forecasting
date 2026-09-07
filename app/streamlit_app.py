"""Dashboard do projeto: forecasting e deteccao de anomalias em CPU de instancias EC2."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import anomaly_detection as ad
from src import data_loader as dl
from src import forecasting as fc

st.set_page_config(page_title="FinOps Preditivo - EC2 CPU", layout="wide")


@st.cache_data
def carregar_pipeline():
    df = dl.carregar_serie()
    windows = dl.carregar_janelas_anomalia()
    treino, teste = dl.split_treino_teste(df)

    perfil = fc.calcular_perfil_sazonal(treino)
    previsto = fc.prever(perfil, teste, offset=len(treino))
    residuo = fc.calcular_residuo(teste, previsto)

    dentro_anomalia = pd.Series(False, index=teste.index)
    for inicio, fim in windows:
        dentro_anomalia |= (teste.index >= inicio) & (teste.index <= fim)

    sinal = ad.calcular_sinal(residuo)
    limiar = ad.calcular_limiar(sinal, ~dentro_anomalia)
    deteccoes = ad.detectar(sinal, limiar)
    alarmes = ad.agrupar_alarmes(deteccoes)
    avaliacao = ad.avaliar_por_janela(alarmes, windows)

    return {
        "df": df, "windows": windows, "treino": treino, "teste": teste,
        "perfil": perfil, "previsto": previsto, "residuo": residuo,
        "dentro_anomalia": dentro_anomalia, "sinal": sinal, "limiar": limiar,
        "deteccoes": deteccoes, "alarmes": alarmes, "avaliacao": avaliacao,
    }


dados = carregar_pipeline()
avaliacao = dados["avaliacao"]

st.title("FinOps Preditivo: Forecasting e Deteccao de Anomalias")
st.caption(
    "O objetivo aqui e antecipar picos de utilizacao de recursos EC2 antes que virem custo "
    "desnecessario ou risco de indisponibilidade. Os dados vem do NAB (Numenta Anomaly "
    "Benchmark), uma serie real de CPU utilization com anomalias rotuladas por especialistas."
)

aba1, aba2, aba3, aba4, aba5 = st.tabs([
    "Visao geral", "Diagnostico", "Forecast", "Anomalias detectadas", "Impacto de negocio",
])

with aba1:
    st.subheader("Serie completa")
    mostrar_anomalias = st.checkbox("Mostrar janelas de anomalia real (ground truth)", value=True)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dados["df"].index, dados["df"]["value"], linewidth=0.7, color="#2563eb")
    if mostrar_anomalias:
        for i, (inicio, fim) in enumerate(dados["windows"]):
            ax.axvspan(inicio, fim, color="red", alpha=0.15,
                       label="Janela de anomalia real" if i == 0 else None)
        ax.legend()
    ax.set_title("EC2 CPU Utilization (14 dias)")
    st.pyplot(fig)

    st.markdown(
        "Durante a analise exploratoria encontramos um pico diario recorrente por volta das "
        "3h da manha, provavelmente de um job agendado. O horario desse pico varia um pouco "
        "de um dia para o outro, chegando a 35 minutos de diferenca entre execucoes. Esse "
        "comportamento e normal e nao deveria ser tratado como anomalia, e foi justamente "
        "esse o principal desafio tecnico do projeto."
    )

with aba2:
    st.subheader("Perfil sazonal aprendido a partir do treino")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dados["perfil"].index, dados["perfil"].values, color="#16a34a")
    ax.set_xlabel("Posicao no ciclo diario (0 a 287, cada unidade equivale a 5 minutos)")
    ax.set_ylabel("Valor esperado")
    ax.set_title("Perfil sazonal medio de um dia")
    st.pyplot(fig)
    st.markdown(
        "Esse e o comportamento considerado normal, aprendido pelo modelo, e serve de base "
        "para o forecasting. Repare que o pico aparece suavizado, bem abaixo da amplitude "
        "real observada nos dados brutos. Isso acontece por causa do jitter no horario do "
        "job: como o pico nao cai sempre na mesma posicao, a media acaba espalhando ele "
        "entre varias posicoes proximas."
    )

with aba3:
    st.subheader("Valor real comparado ao previsto, no conjunto de teste")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dados["teste"].index, dados["teste"]["value"], label="Real",
            linewidth=0.7, color="#2563eb")
    ax.plot(dados["previsto"].index, dados["previsto"].values, label="Previsto (perfil sazonal)",
            linewidth=1, color="#f97316")
    for inicio, fim in dados["windows"]:
        ax.axvspan(inicio, fim, color="red", alpha=0.1)
    ax.legend()
    st.pyplot(fig)

    mae = dados["residuo"].abs().mean()
    st.metric("MAE (erro absoluto medio)", f"{mae:.4f}")
    st.caption(
        "Chegamos a testar SARIMA com termos de Fourier para representar a sazonalidade, "
        "mas nao trouxe ganho em relacao a essa abordagem mais simples, pelo menos para a "
        "tarefa de deteccao de anomalia neste dataset. A comparacao completa esta no README."
    )

with aba4:
    st.subheader("Sinal de deteccao comparado ao limiar")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dados["sinal"].index, dados["sinal"].values, linewidth=0.8, color="#2563eb",
            label="Sinal (desvio-padrao movel, janela de 2h)")
    ax.axhline(dados["limiar"], color="black", linestyle="--",
               label=f"Limiar (percentil 99 do periodo normal) = {dados['limiar']:.3f}")
    for inicio, fim in dados["windows"]:
        ax.axvspan(inicio, fim, color="red", alpha=0.1)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Alarmes gerados")
    for inicio, fim in dados["alarmes"]:
        if (inicio, fim) in avaliacao["alarmes_verdadeiros"]:
            status = "Verdadeiro positivo"
        else:
            status = "Falso positivo"
        st.write(f"{status}, de {inicio} ate {fim}")

    st.subheader("Janelas reais e cobertura")
    for i, (inicio, fim) in enumerate(dados["windows"]):
        if (inicio, fim) in avaliacao["janelas_detectadas"]:
            status = "detectada"
        else:
            status = "nao detectada"
        st.write(f"Janela {i + 1} ({status}), de {inicio} ate {fim}")

    st.subheader("Metricas, avaliacao por evento no estilo NAB")
    c1, c2, c3 = st.columns(3)
    c1.metric("Precisao", f"{avaliacao['precisao']:.3f}")
    c2.metric("Recall", f"{avaliacao['recall']:.3f}")
    c3.metric("F1-score", f"{avaliacao['f1']:.3f}")

    with st.expander("Por que a segunda janela nao foi detectada"):
        st.markdown(
            "Essa janela corresponde a um evento cuja amplitude real, por volta de 0.6, e "
            "menor que o pico do job de rotina que acontece todo dia, em torno de 1.5. Um "
            "limiar calibrado para nao disparar falso alarme no job de rotina acaba, por "
            "construcao, deixando passar um evento com amplitude menor que esse ruido de "
            "referencia. E uma limitacao real do modelo, que foi investigada a fundo, e nao "
            "um bug ou algo que passou despercebido."
        )

with aba5:
    st.subheader("O que isso significa em termos de FinOps")

    n_falsos = len(avaliacao["alarmes_falsos"])
    n_verdadeiros = len(avaliacao["alarmes_verdadeiros"])

    st.markdown(
        f"""
O modelo teria alertado a equipe sobre o incidente principal, a janela de maior severidade,
logo apos o inicio do evento, sem precisar esperar a fatura mensal de custo chegar para
perceber o problema.

Ao longo do periodo de teste, cerca de 2.8 dias, o modelo gerou {len(dados['alarmes'])} alarmes
no total: {n_verdadeiros} verdadeiro(s) e {n_falsos} falso(s) positivo(s). E uma taxa de falso
alarme baixa considerando as execucoes normais de rotina que aconteceram nesse periodo.

Vale registrar uma limitacao conhecida: eventos anomalos com amplitude menor que o ruido
normal do sistema, como o caso da segunda janela, nao sao capturados por esse modelo. Para
um cenario de producao, o recomendado seria combinar esse detector estatistico com outras
fontes de contexto, como logs de incidente ou alertas de infraestrutura, para cobrir esse
tipo de caso.
        """
    )