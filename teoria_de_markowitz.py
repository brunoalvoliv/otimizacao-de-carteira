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
    pesos = np.round(pesos / pesos.sum(), 3)
    carteira_pesos.append(pesos)
    #Retorno anualizado
    retorno_anual =  np.sum(retornos * pesos) * 252
    carteira_retornos.append(retorno_anual)
    #Matriz de covariância e risco da carteira
    matriz_cov =  retornos.cov() * 252
    carteira_variancias = np.dot(pesos.T, np.dot(matriz_cov, pesos.T))
    carteira_desvipadrao = np.sqrt(carteira_variancias)
    carteira_riscos.append(carteira_desvipadrao)
    #Sharpe ratio
    sharpe = (retorno_anual + ativo_livre_de_risco) / carteira_desvipadrao
    sharpe_ratios.append(sharpe)

print(carteira_retornos)