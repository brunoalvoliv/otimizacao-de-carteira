#Importando bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt

#np.set_printoptions(threshold=13)
plt.style.use('ggplot')

#Importando dados das empresas

tickers = 'PETR3.SA MGLU3.SA ITSA4.SA RDOR3.SA LCAM3.SA BTCR11.SA CSAN3.SA SMFT3.SA ARZZ3.SA TRPL3.SA B3SA3.SA WEGE3.SA'.split()
inicio = dt.datetime(2019, 1, 1)
fim = dt.datetime(2021, 11, 30)

#Inputs para criar as carteiras

quantidade_de_carteiras = 10000
ativo_livre_de_risco = 0

retornos = pd.DataFrame()
for ticker in tickers:
    dados = yf.download(ticker, start=inicio, end=fim)
    dados = pd.DataFrame(dados)
#Calculando os retornos das empresas
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
    sharpe = ((retorno_anual + ativo_livre_de_risco) / carteira_desvipadrao)
    sharpe_ratios.append(sharpe)

carteira_retornos = np.array(carteira_retornos)
carteira_riscos = np.array(carteira_riscos)
carteira_pesos = np.array(carteira_pesos)
sharpe_ratios = np.array(sharpe_ratios)

carteira_metricas = [carteira_retornos, carteira_riscos, sharpe_ratios, carteira_pesos]
carteiras_df = pd.DataFrame(carteira_metricas).T
carteiras_df.columns = ['Retorno', 'Risco', 'Índice de Sharpe', 'Peso']

minimo_risco = carteiras_df.iloc[carteiras_df['Risco'].astype(float).idxmin()]
maximo_retorno = carteiras_df.iloc[carteiras_df['Retorno'].astype(float).idxmax()]
maximo_sharpe = carteiras_df.iloc[carteiras_df['Índice de Sharpe'].astype(float).idxmax()]

print('Carteira de mínimo risco')
print(minimo_risco)
print(minimo_risco.Peso[:12])
print(tickers)
print('')

print('Carteira de máximo retorno')
print(maximo_retorno)
print(maximo_retorno.Peso[:12])
print(tickers)
print('')

print('Carteira de máximo Índice de Sharpe')
print(maximo_sharpe)
print(maximo_sharpe.Peso[:12])
print(tickers)
print('')

#Visualização

plt.figure(figsize=(12, 6));
plt.title('Otimização de carteira', fontsize=25);
plt.scatter(carteira_riscos, carteira_retornos, c = carteira_retornos / carteira_riscos);
plt.scatter(minimo_risco.Risco, minimo_risco.Retorno, color='red', label='Mínima variância', marker='*');
plt.scatter(maximo_retorno.Risco, maximo_retorno.Retorno, color='blue', label='Máximo retorno', marker='p');
plt.scatter(maximo_sharpe.Risco, maximo_sharpe.Retorno, color='black', label='Máximo Sharpe Ratio', marker='H');
plt.xlabel('Volatilidade', fontsize=20);
plt.ylabel('Retornos esperados', fontsize=20);
plt.xticks(fontsize=15);
plt.yticks(fontsize=15);
plt.colorbar(label='Índice de Sharpe');
plt.legend()
plt.show()
