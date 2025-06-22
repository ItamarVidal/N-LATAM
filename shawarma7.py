#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 00:36:15 2023

@author: itamarvidal@protonmail
"""

import time
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

caminho_atual = os.getcwd()
print("Caminho da pasta atual:", caminho_atual)

# Obter todos os arquivos .xlsx com o início do nome "data" na pasta
arquivos = glob.glob('data*.xlsx')

# Exibir a lista de arquivos disponíveis
print("Arquivos disponíveis:")
for i, arquivo in enumerate(arquivos):
    print(f"{i + 1}. {arquivo}")

# Solicitar ao usuário para escolher um arquivo
opcao = int(input("Escolha o número do arquivo desejado: ")) - 1

# Verificar se a opção escolhida é válida
if 0 <= opcao < len(arquivos):
    # Ler o arquivo selecionado
    arquivo_selecionado = arquivos[opcao]
    data = pd.read_excel(arquivo_selecionado)
    print(f"Arquivo '{arquivo_selecionado}' lido com sucesso.")
else:
    print("Opção inválida. Encerrando o programa.")

# Início da contagem do tempo
inicio = time.time()

# Parâmetros de treinamento
epoch = 200
ensemble = 1

# Extrair os dados da coluna C4 à C55 como um array numpy
semanas_sell_out = np.array(data.iloc[2:54, 2])

# Extrair os dados da coluna D4 à D55 e formatar os números
PY_str = data.iloc[2:54, 3].astype(str)
PY = np.array(PY_str, dtype=float)

# Extrair os dados da coluna F4 à F55 e formatar os números
CY_str = data.iloc[2:54, 5].astype(str)
CY = np.array(CY_str, dtype=float)
CY = CY[CY != 0]

concatenated = np.concatenate((PY, CY))

sell_out = concatenated

# Preparar os dados para treinamento
window_size = 10
X_train = []
y_train = []
for i in range(len(sell_out) - window_size):
    X_train.append(sell_out[i:i+window_size])
    y_train.append(sell_out[i+window_size])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Variável para armazenar as previsões
predictions_total = []

# Executar o loop 10 vezes
for _ in range(ensemble):
    # Construir o modelo de rede neural profunda
    model = Sequential()
    model.add(LSTM(512, activation='relu', input_shape=(window_size, 1)))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Treinar o modelo
    model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train, epochs=epoch, batch_size=8)

    # Preparar os dados de teste para previsão
    X_test = sell_out[-window_size:].reshape((1, window_size, 1))

    # Fazer a previsão para as próximas 10 semanas e adicionar à lista de previsões
    predictions = []
    for _ in range(10):
        predicted_value = model.predict(X_test)
        predictions.append(predicted_value[0][0])
        X_test = np.roll(X_test, -1)
        X_test[0][-1] = predicted_value[0][0]
    
    # Adicionar as previsões do loop atual à lista total de previsões
    predictions_total.append(predictions)

# Calcular a média aritmética das previsões
predictions_average = np.mean(predictions_total, axis=0)

# Criar um array com os valores totais de vendas e previsões
total_sales = np.concatenate((sell_out[-len(CY):], predictions_average))

# Verificar o tamanho de PY
py_size = len(PY)

# Calcula a diferença entre os tamanhos de total_sales e PY
diff_size = py_size - len(total_sales)

# Adiciona zeros ao final de total_sales para igualar o tamanho com PY
extended_total_sales = np.pad(total_sales, (0, diff_size), mode='constant')


# Calcular a diferença entre extended_total_sales e PY até o tamanho de CY
diff = extended_total_sales[:len(CY)] - PY[:len(CY)]


# Configurações do gráfico
fig, ax = plt.subplots()

# Definir os números das semanas no eixo x como 1, 2, 3, ...
ax.set_xticks(np.arange(1, len(PY) + 1))

# Plotar a linha preta para PY
ax.plot(semanas_sell_out, PY, color='#6699cc', label='PY', linewidth=1)

# Plotar as barras verdes para extended_total_sales até o tamanho de CY
ax.bar(semanas_sell_out[:len(CY)], extended_total_sales[:len(CY)], color=np.where(diff >= 0, '#669933', '#990000'), alpha=0.8, label='Sell Out')

# Plotar as barras cinza para extended_total_sales após o tamanho de CY
ax.bar(semanas_sell_out[len(CY):], extended_total_sales[len(CY):], color='#7c7c7c', alpha=0.8, label='Previsões')

# Medir o taamanho de total_sales
total_sales_size = len(total_sales)
# Ajustar tamanho de PY para que fique do tamanho de total_sales
PY_resized = np.resize(PY, total_sales_size)


# Calcular a diferença percentual entre total_sales e PY
diff_percent = ((total_sales - PY_resized) / PY_resized) * 100

# Arredondar sem decimais
diff_rounded = np.round(diff_percent).astype(int)

# Verificar o tamanho de diff_percent e semanas_sell_out
diff_size = len(diff_rounded)

semanas_size = len(semanas_sell_out)

# Completar diff_percent com zeros até o tamanho de semanas_sell_out
diff_extended = np.concatenate((diff_rounded, np.zeros(semanas_size - diff_size)))
diff_extended_int = np.round(diff_extended).astype(int)


# Adicionar legendas
#ax.legend()

# Configurar os números dos eixos x e y menores
ax.tick_params(axis='both', which='both', labelsize=5)

#############################################################################
# # Adicionar as anotações no topo do gráfico
# max_diff = np.max(diff_extended_int)
# offset = -5  # Valor negativo para posicionar os números abaixo da linha superior da caixa do gráfico

# for i in range(semanas_size):
#     if i < len(total_sales):
#         color = 'black'
#     else:
#         color = 'white'

#     ax.annotate(f'{diff_extended_int[i]}', (semanas_sell_out[i], max_diff + offset),
#                   xytext=(0, 295), textcoords='offset points', ha='center', va='bottom',
#                   color=color, fontsize=5)
# Adicionar as anotações no topo do gráfico II
max_diff = np.max(CY)
offset = 20

for i in range(semanas_size):
    if i < len(total_sales):
        color = 'black'
    else:
        color = 'white'

    ax.text(semanas_sell_out[i], max_diff + offset, str(diff_extended_int[i]),
              fontsize=5, color=color, ha='center', va='bottom')

############################################################################

# Ajustar o tamanho da figura para uma resolução maior
fig.set_size_inches(12, 6)

# Mostrar o gráfico
plt.show()

# Salvar a figura com resolução de 300 dpi
nome_arquivo = str(arquivo_selecionado) + '_SHAWARMA7.png'
fig.savefig(nome_arquivo, dpi=300)


# Fim da contagem do tempo
fim = time.time()

# Tempo de processamento em segundos
tempo_total = fim - inicio

print("Tempo de processamento:", tempo_total, "segundos")
print((diff_extended_int[len(CY):(len(CY)+10)]), ' => Diferença percetual das previsões de', nome_arquivo)

