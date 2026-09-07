# FinOps Preditivo: Forecasting e Detecção de Anomalias em CPU EC2

## Sumário

- [Objetivo](#objetivo)
- [Dataset](#dataset)
- [Planejamento inicial](#planejamento-inicial)
- [Etapas realizadas](#etapas-realizadas)
- [Metodologia](#metodologia)
- [Resultados](#resultados)
- [Dashboard](#dashboard)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como reproduzir](#como-reproduzir)
- [Limitações](#limitações)
- [Próximos passos](#próximos-passos)

## Objetivo

Este projeto nasceu como um estudo aplicado de forecasting e detecção de anomalias em séries temporais, usando como pretexto um cenário real de FinOps: antecipar picos de utilização de recursos EC2 antes que virem custo desnecessário ou risco de indisponibilidade, em vez de descobrir o problema só quando a fatura mensal chega.

O foco principal foi consolidar, na prática, os seguintes tópicos de estudo:

- Decomposição de séries temporais (tendência, sazonalidade, resíduo)
- Estacionariedade e o teste de Dickey-Fuller Aumentado (ADF)
- Interpretação de ACF e PACF para identificação de parâmetros de modelo
- Modelos ARIMA e SARIMA, incluindo suas limitações práticas (como custo computacional de sazonalidade longa)
- Detecção de anomalias a partir de resíduos de forecasting
- Métricas de avaliação para forecasting (MAE, RMSE, MAPE) e para detecção de anomalias (precisão, recall, F1)

O projeto cobre o pipeline completo: análise exploratória, diagnóstico estatístico da série temporal, construção de um modelo de forecasting, detecção de anomalias a partir dos resíduos do modelo, validação contra rótulos reais e um dashboard interativo que traduz os resultados para linguagem de negócio.

## Dataset

Os dados vêm do NAB (Numenta Anomaly Benchmark), um benchmark usado como referência em pesquisa e na indústria para avaliação de métodos de detecção de anomalias em séries temporais. Especificamente, foi usada a série `ec2_cpu_utilization_24ae8d.csv`, parte do conjunto `realAWSCloudwatch`: métricas reais de CPU utilization coletadas de uma instância EC2 pelo CloudWatch, ao longo de 14 dias, com amostragem a cada 5 minutos.

O NAB fornece, junto com os dados, janelas de tempo rotuladas como anômalas por especialistas, o que permitiu validar objetivamente os modelos construídos aqui, em vez de depender só de inspeção visual.

Repositório do NAB: https://github.com/numenta/NAB

## Planejamento inicial

Antes de começar a implementação, defini um plano de trabalho estruturado em etapas, pensado para seguir uma progressão lógica: entender os dados primeiro, só depois modelar, e só depois avaliar e comunicar os resultados. A intenção original era a seguinte:

1. **Ingestão e exploração de dados (EDA):** carregar a série e os rótulos de anomalia, identificar visualmente padrões e comportamentos suspeitos antes de qualquer modelagem.
2. **Pré-processamento:** validar a qualidade dos dados (gaps, duplicatas, valores nulos) e dividir treino e teste respeitando a ordem cronológica.
3. **Diagnóstico estatístico:** decompor a série em tendência, sazonalidade e resíduo, testar estacionariedade (ADF) e interpretar ACF/PACF para embasar a escolha dos parâmetros do modelo.
4. **Modelagem de forecasting com SARIMA:** treinar um modelo estatístico clássico, capaz de capturar tanto a dependência de curto prazo quanto o padrão sazonal diário da série.
5. **Detecção de anomalias:** usar os resíduos do modelo de forecasting como sinal de anomalia, sinalizando desvios significativos do comportamento esperado.
6. **Validação:** comparar as anomalias detectadas pelo modelo contra os rótulos reais do NAB, usando métricas de precisão, recall e F1.
7. **Dashboard:** construir uma interface interativa em Streamlit para visualizar a série, o forecast, as anomalias detectadas e o impacto de negócio.
8. **Documentação:** consolidar todo o raciocínio, decisões e limitações em um README, deixando o projeto claro para qualquer pessoa que for lê-lo.

Nem tudo saiu exatamente como planejado. O SARIMA nativo com sazonalidade diária (288 pontos) se mostrou inviável na prática por consumo de memória, o que exigiu uma mudança de abordagem para regressão harmônica (termos de Fourier). Além disso, o SARIMA, mesmo ajustado dessa forma, não superou uma abordagem mais simples de decomposição sazonal para a tarefa de detecção de anomalias neste dataset específico. O raciocínio técnico completo por trás dessas mudanças está na seção de Metodologia.

## Etapas realizadas

A tabela abaixo resume o que foi feito em cada notebook, na ordem em que o projeto foi desenvolvido. Os notebooks estão na pasta `notebooks/`.

| Notebook | O que foi feito | Principal conclusão |
|---|---|---|
| `01_init_test.ipynb` | Carregamento inicial da série e dos rótulos de anomalia, primeira visualização dos dados. | Identificação de um pico diário recorrente, e das duas janelas de anomalia rotuladas pelo NAB. |
| `02_data_validation.ipynb` | Validação de qualidade: checagem de duplicatas, gaps temporais, valores nulos e fora do domínio esperado. | A série está limpa, sem necessidade de nenhum tratamento adicional. |
| `03_train_test_split.ipynb` | Divisão treino e teste respeitando a ordem cronológica (80/20). | As duas janelas de anomalia real caem inteiramente no conjunto de teste, permitindo treinar só com dados normais. |
| `04_seasonal_decomposition.ipynb` | Decomposição da série (tendência, sazonalidade, resíduo), implementada manualmente e comparada com a função do `statsmodels`. | O resíduo mantinha um padrão repetido, o que levou à descoberta de que o pico diário não acontece em horário fixo (jitter de até 35 minutos). |
| `05_rolling_std_anomaly_signal.ipynb` | Comparação da magnitude do resíduo bruto entre período normal e anômalo, e teste do desvio-padrão móvel como sinal alternativo, em diferentes tamanhos de janela. | O resíduo bruto não separa os grupos normal e anômalo. O desvio-padrão móvel em janela de 2 horas foi o primeiro a separar os grupos sem sobreposição. |
| `06_anomaly_detection_evaluation.ipynb` | Definição de um limiar objetivo (percentil 99 do período normal), aplicação como detector, e avaliação ponto a ponto e por janela (estilo NAB). | O modelo detecta bem a anomalia de maior magnitude, mas não detecta a segunda janela, cuja amplitude é menor que o próprio ruído normal do sistema. |
| `07_stationarity_acf_pacf.ipynb` | Teste de Dickey-Fuller Aumentado (ADF) e análise de ACF/PACF, em lags curtos e sazonais. | A série já é fortemente estacionária, sem necessidade de diferenciação. Os parâmetros do SARIMA foram definidos a partir dessa análise. |
| `08_sarima_modeling.ipynb` | Tentativa de treinar SARIMA nativo (inviável por consumo de memória), solução com termos de Fourier, comparação de candidatos por AIC e cálculo de métricas de forecasting. | O modelo `ARIMA(1,0,0)` com termos de Fourier foi o melhor por AIC, mas os coeficientes de Fourier ficaram próximos de zero, indicando que o pico diário não foi bem representado como sinal determinístico. |
| `09_sarima_anomaly_detection.ipynb` | Aplicação do mesmo pipeline de detecção de anomalias (desvio-padrão móvel, limiar, avaliação por janela) usando o resíduo do SARIMA, e comparação final com a decomposição simples. | O SARIMA não superou a decomposição simples para a tarefa de detecção de anomalias neste dataset. A decomposição simples foi mantida como modelo final do projeto. |

## Metodologia

### Análise exploratória e o principal desafio do projeto

A série de CPU utilization mostrou, logo na exploração inicial, um pico diário recorrente por volta das 3h da manhã, provavelmente de um job agendado (backup, processamento batch, ou similar). Investigando o horário exato de cada ocorrência, ficou claro que esse pico não acontece num horário fixo: ele varia entre 03:05 e 03:40, um jitter de até 35 minutos entre execuções.

Esse achado se tornou o fio condutor de boa parte das decisões do projeto: qualquer método baseado em posição fixa dentro do ciclo (decomposição sazonal clássica, regressão harmônica com poucos termos de Fourier) tem dificuldade em capturar totalmente esse tipo de evento, porque a média entre ciclos acaba espalhando a amplitude do pico entre várias posições próximas, em vez de concentrar tudo numa só.

### Pré-processamento e diagnóstico estatístico

A série passou por uma validação de qualidade (checagem de duplicatas, gaps temporais, valores nulos e fora do domínio esperado) e não precisou de nenhum tratamento: os dados vieram limpos, com frequência regular de 5 minutos ao longo de 14 dias.

A divisão treino/teste seguiu a ordem cronológica (80/20), com as duas janelas de anomalia rotuladas caindo inteiramente no conjunto de teste, o que permitiu treinar o modelo só com comportamento normal e validar a detecção exatamente no período com anomalias reais.

O teste de Dickey-Fuller Aumentado (ADF) mostrou que a série já era fortemente estacionária (estatística de -59.89, bem além do valor crítico de 1%), o que tornou desnecessária qualquer diferenciação. A análise de ACF e PACF, tanto em lags curtos quanto em lags sazonais (múltiplos de 288, correspondentes a 1 dia), confirmou esse comportamento e apontou para uma dependência de curto prazo bem fraca, com o principal padrão de autocorrelação concentrado no lag sazonal.

### Modelagem de forecasting

A primeira tentativa foi treinar um SARIMA nativo, com sazonalidade `s=288` representando o ciclo diário. Essa abordagem se mostrou computacionalmente inviável: a representação interna do componente sazonal do `statsmodels` cresce em função do quadrado do período sazonal, o que gerava uma matriz de mais de 2GB de alocação e esgotava a memória disponível.

A alternativa adotada foi representar a sazonalidade como variáveis exógenas, usando termos de Fourier (regressão harmônica dinâmica), mantendo o restante do modelo como um ARIMA simples.

**Por que Fourier resolve o problema de memória:** o SARIMAX, quando recebe um componente sazonal (`seasonal_order`), representa esse componente internamente como um modelo de espaço de estados cuja dimensão cresce em função do período sazonal `s`. Na prática, o tamanho das matrizes usadas no filtro de Kalman cresce proporcionalmente a `s²` multiplicado pelo número de observações. Com `s=288` e cerca de 3226 pontos de treino, isso gerou uma tentativa de alocação de aproximadamente 289×289×3226 posições, mais de 2GB só nessa alocação, o que estourou a memória disponível. Os termos de Fourier tratam a sazonalidade de forma diferente: em vez de representá-la internamente (crescendo com `s²`), ela entra como variáveis exógenas (pares de seno e cosseno), adicionando apenas `2 × número de harmônicos` colunas ao modelo (6 colunas, para 3 harmônicos, neste projeto). A dimensão do espaço de estados passa a depender só da ordem do ARIMA (`p`, `d`, `q`), não mais do período sazonal — a complexidade de memória deixa de ser `O(s²)` e passa a ser aproximadamente `O(1)` em relação a `s`. Essa é uma técnica padrão em forecasting (regressão harmônica dinâmica, descrita por Hyndman e Athanasopoulos) para lidar com esse tipo de limitação prática de escala.

Foram testadas três combinações de parâmetros não-sazonais (baseadas no diagnóstico de ACF/PACF), comparadas pelo critério AIC:

| Modelo | AIC |
|---|---|
| ARIMA(0,0,0) + Fourier | maior (pior) |
| **ARIMA(1,0,0) + Fourier** | **-4835.25 (melhor)** |
| ARIMA(0,0,1) + Fourier | maior (pior) |

O modelo escolhido foi `ARIMA(1,0,0)` com 3 harmônicos de Fourier.

### Detecção de anomalias

A estratégia inicial era usar a magnitude do resíduo de um único ponto (real menos previsto) como sinal de anomalia. Essa abordagem se mostrou ineficaz: como o próprio job de rotina já gera resíduos grandes todo dia (por causa do jitter de horário), a distribuição do resíduo em períodos normais e anômalos ficou praticamente indistinguível (médias de 0.038 e 0.039, respectivamente).

A solução foi trocar a métrica: em vez de olhar a magnitude de um ponto isolado, passamos a medir a variabilidade (desvio-padrão) do resíduo dentro de uma janela móvel de tempo. Foram testados tamanhos de janela de 30 minutos a 3 horas, e a janela de 2 horas foi a que produziu a primeira separação limpa entre os grupos normal e anômalo, sem sobreposição de valores extremos.

O limiar de detecção foi calibrado de forma objetiva, como o percentil 99 da distribuição desse sinal no período normal (sem usar os rótulos de anomalia para calibrar), e os pontos que ultrapassam esse limiar são agrupados em alarmes (eventos), não avaliados isoladamente ponto a ponto.

A avaliação foi feita de duas formas: ponto a ponto (mais rigorosa) e por janela/evento, seguindo o mesmo critério de pontuação usado oficialmente pelo NAB (um alarme conta como acerto se tocar qualquer parte de uma janela rotulada). A avaliação por evento é a mais alinhada com o caso de uso de negócio, já que o que importa, na prática, é alertar a tempo, não acertar cada minuto exato.

### Comparação: decomposição sazonal simples vs. SARIMA

O mesmo pipeline de detecção (resíduo, desvio-padrão móvel, limiar, avaliação por evento) foi aplicado tanto ao resíduo da decomposição sazonal simples quanto ao resíduo do SARIMA, para comparar os dois métodos:

| Métrica | Decomposição simples | SARIMA + Fourier |
|---|---|---|
| Precisão | 0.923 | 0.857 |
| Recall | 0.119 | 0.060 |
| F1-score | 0.211 | 0.112 |

Contra a expectativa inicial, o SARIMA não superou a decomposição simples. Investigando os coeficientes do modelo, ficou claro que os termos de Fourier aprenderam uma amplitude quase nula (entre 0.003 e 0.009), ou seja, o SARIMA também não conseguiu representar o pico diário como sinal determinístico, pelo mesmo motivo do jitter de horário. Por isso, a decomposição sazonal simples foi adotada como o modelo final do projeto.

## Resultados

O modelo final (decomposição sazonal simples + desvio-padrão móvel de 2 horas como sinal de anomalia) alcançou os seguintes resultados no conjunto de teste:

**Forecasting:**

| Métrica | Valor |
|---|---|
| MAE | 0.0381 |
| RMSE | 0.1247 |
| MAPE | 29.91% |

O MAPE elevado é esperado neste dataset: a série opera majoritariamente numa base baixa (~0.13), e o MAPE é sensível a valores reais próximos de zero, ampliando o erro percentual mesmo para desvios absolutos pequenos. MAE e RMSE são mais informativos neste caso.

**Detecção de anomalias, avaliação por evento (estilo NAB):**

| Métrica | Valor |
|---|---|
| Precisão | 0.667 |
| Recall | 0.500 |
| F1-score | 0.571 |

Na prática, isso significa que o modelo gerou 3 alarmes ao longo do período de teste (~2.8 dias): 2 verdadeiros positivos (cobrindo a janela de anomalia de maior severidade) e 1 falso positivo (um dia em que o job de rotina disparou um pouco mais forte que o normal). Das duas janelas de anomalia reais, 1 foi detectada e 1 não foi, pelo motivo detalhado na seção de Limitações.

## Dashboard

O projeto inclui um dashboard interativo em Streamlit (`app/streamlit_app.py`), organizado em 5 abas:

1. **Visão geral:** a série completa, com opção de destacar as janelas de anomalia real.
2. **Diagnóstico:** o perfil sazonal aprendido a partir do treino.
3. **Forecast:** comparação entre valor real e previsto, com a métrica de erro do modelo.
4. **Anomalias detectadas:** o sinal de detecção comparado ao limiar, os alarmes gerados (com rótulo de verdadeiro ou falso positivo), a cobertura de cada janela real, as métricas de avaliação, e uma explicação de por que a segunda janela não foi detectada.
5. **Impacto de negócio:** tradução dos resultados para linguagem de FinOps, incluindo a limitação conhecida e uma recomendação para cenário de produção.

## Estrutura do repositório

```
finops-anomaly-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── ec2_cpu_utilization_24ae8d.csv
│   └── labels/
│       └── combined_windows.json
├── notebooks/
│   ├── 01_init_test.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_train_test_split.ipynb
│   ├── 04_seasonal_decomposition.ipynb
│   ├── 05_rolling_std_anomaly_signal.ipynb
│   ├── 06_anomaly_detection_evaluation.ipynb
│   ├── 07_stationarity_acf_pacf.ipynb
│   ├── 08_sarima_modeling.ipynb
│   └── 09_sarima_anomaly_detection.ipynb
├── src/
│   ├── data_loader.py
│   ├── forecasting.py
│   └── anomaly_detection.py
└── app/
    └── streamlit_app.py
```

## Como reproduzir

1. Clone o repositório:
   ```bash
   git clone https://github.com/NikolasTola/finops-anomaly-forecasting.git
   cd finops-anomaly-forecasting
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Os dados não precisam ser baixados no link, eles já estão versionados no repositório (`data/raw/` e `data/labels/`)
4. Rode os notebooks, na ordem, dentro da pasta `notebooks/`, para reproduzir a análise completa.
5. Rode o dashboard:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Limitações

**Detecção de anomalias de baixa amplitude.** O modelo não detecta anomalias cuja amplitude é menor que o ruído normal do sistema, como foi o caso da segunda janela de anomalia deste dataset (pico real de aproximadamente 0.6, contra um pico normal de rotina de aproximadamente 1.5). Qualquer limiar calibrado para não disparar falso alarme no comportamento normal, por construção, deixa passar eventos mais discretos que esse próprio ruído de referência. Isso não é uma falha de implementação, é uma limitação estrutural de métodos de detecção baseados em magnitude estatística, quando não há contexto adicional (como logs de incidente) para complementar a análise.

**Sazonalidade com jitter de horário.** O principal padrão sazonal da série (o pico diário) não ocorre num horário fixo, variando em até 35 minutos entre execuções. Isso limita a eficácia de qualquer método baseado em posição fixa dentro do ciclo, seja decomposição sazonal clássica, seja regressão harmônica com poucos termos de Fourier. Ambos os métodos testados neste projeto convergiram para o mesmo problema: a amplitude real do pico é subestimada, porque a média entre ciclos espalha o pico entre posições próximas em vez de concentrá-lo numa só.

**Escopo de validação limitado a uma única série.** Todo o desenvolvimento e validação foi feito sobre uma única série do NAB (`ec2_cpu_utilization_24ae8d.csv`). Não há garantia de que os mesmos parâmetros (janela de 2 horas, percentil 99) generalizem bem para outras séries com padrões de ruído ou sazonalidade diferentes, sem um novo processo de calibração.

**Threshold estático.** O limiar de detecção foi calibrado uma única vez, a partir de um período fixo de treino. Em um cenário real de produção, o comportamento "normal" de um sistema tende a mudar ao longo do tempo (crescimento orgânico de uso, mudanças de infraestrutura), o que exigiria uma estratégia de recalibração periódica do limiar, não implementada aqui.

## Próximos passos

- Testar uma estratégia de detecção que combine múltiplos sinais (por exemplo, desvio-padrão móvel e frequência de picos numa janela), em vez de depender de uma única métrica de variabilidade.
- Avaliar o mesmo pipeline em outras séries do NAB (`realAWSCloudwatch` tem outras séries de CPU utilization), para checar se os parâmetros escolhidos generalizam ou precisam de ajuste por série.
- Explorar uma estratégia de recalibração periódica do limiar, simulando um cenário de produção onde o comportamento normal do sistema muda ao longo do tempo.
- Investigar se a inclusão de contexto adicional (por exemplo, logs de eventos de infraestrutura, se disponíveis) ajudaria a capturar anomalias de baixa amplitude, como a que passou despercebida neste projeto.
- Considerar bibliotecas mais recentes de forecasting probabilístico (como Prophet ou modelos baseados em redes neurais para séries temporais) como alternativa ao SARIMA, especialmente para lidar melhor com sazonalidade de fase variável.