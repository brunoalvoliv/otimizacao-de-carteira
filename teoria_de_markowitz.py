#Importando bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt

plt.style.use('ggplot')

#Importando dados das empresas

tickers = 'TOTS3.SA MGLU3.SA WEGE3.SA'.split()
inicio = dt.datetime(2019, 1, 1)
fim = dt.datetime(2021, 11, 23)

#Inputs para criar as carteiras

quantidade_de_carteiras = 5
ativo_livre_de_risco = 0

retornos = pd.DataFrame()
for ticker in tickers:
    dados = yf.download(ticker, start=inicio, end=fim)
    dados = pd.DataFrame(dados)
#Calculando os dados das empresas
    dados[ticker] = dados['Adj Close'].pct_change()

    if retornos.empty:
        retornos = dados[[ticker]]
    else:
        retornos = retornos.join(dados[[ticker]], how='outer')

carteira_retornos = []
carteira_riscos = []
sharpe_ratios = []
carteira_pesos = []

for carteiras in range(quantidade_de_carteiras):
    #Definindo os pesos
    pesos = np.random.random_sample(len(tickers))
    pesos = np.round((pesos / np.sum(pesos)), 3)
    carteira_pesos.append(pesos)
    #Retorno anualizado
    retorno_anual =  np.sum(retornos.mean() * pesos) * 252
    carteira_retornos.append(retorno_anual)
    #Matriz de covariância e risco da carteira
    matriz_cov =  retornos.cov() * 252
    carteira_variancias = np.dot(pesos.T, np.dot(matriz_cov, pesos.T))
    carteira_desvipadrao = np.sqrt(carteira_variancias)
    carteira_riscos.append(carteira_desvipadrao)
    #Sharpe ratio
    sharpe = (retorno_anual + ativo_livre_de_risco) / carteira_desvipadrao
    sharpe_ratios.append(sharpe)

carteira_retornos = np.array(carteira_retornos)
carteira_riscos = np.array(carteira_riscos)
carteira_pesos = np.array(carteira_pesos)
sharpe_ratios = np.array(sharpe_ratios)

carteira_metricas = [carteira_retornos, carteira_riscos, sharpe_ratios, carteira_pesos]
carteiras_df = pd.DataFrame(carteira_metricas).T
carteiras_df.columns = ['Retornos', 'Riscos', 'Índices de Sharpe', 'Pesos']

minimo_risco = carteiras_df.iloc[carteiras_df['Riscos'].astype(float).idxmin()]
maximo_retorno = carteiras_df.iloc[carteiras_df['Retornos'].astype(float).idxmax()]
maximmo_sharpe = carteiras_df.iloc[carteiras_df['Índices de Sharpe'].astype(float).idxmax()]

print('Carteira de mínimo risco')
print(minimo_risco)
print(tickers)
print('')

print('Carteira de máximo retorno')
print(maximo_retorno)
print(tickers)
print('')

print('Carteira de máximo Índice de Sharpe')
print(minimo_risco)
print(tickers)
print('')
