#Autor: Ítalo Reis
#Data: 19/02/2025
#Objetivo: Usar Machine Learning para fazer o agrupamento de dados de clientes de uma empresa, 
# o agrupamento deve ser feito por similaridade dos clientes. Após o agrupamento devemos construir
# um relatório para o departamento de Marketing.


#imports
#limitando o uso de threads, para evitar o uso exessivo de memória
from email.mime import base
import os
os.environ["OMP_NUM_THREADS"] = '1' #resolução para um erro que aconteceu ao utilizar o KMEANS
#aparentemente se nao declararmos esse import, o Kmeans utiliza mais memoria doque o nescessário
#isso pode prejudicar o desempenho do Computador.

import pandas as pd #Com o Pandas, você pode carregar, limpar, manipular e 
#analisar dados de várias fontes (como arquivos CSV, bancos de dados SQL, etc.).
#sklearn é um biblioteca de machine learning para python, possui diversos algoritmos e etc...

from sklearn.cluster import KMeans #Algoritmo de Machine Learning chamado de "cluster", utilizado
#para separação de dados semelhantes em grupos

from sklearn.preprocessing import StandardScaler #É usado para a pré processagem de dados, ou seja,
#ele atua para padronizar os dados e facilitar o estudo do algoritmo de machine learning


#CARREGAMENTO DOS DADOS
baseDeDadosClientes = pd.read_csv('dados/dados_clientes.csv') #importa os dados do arquvio csv e os transformam em
#algo semelhante a uma tabela de excel
#print(type(leituraDeDados)) # verifica se o arquvio csv foi realmente convertido para DataFrame

#print(leituraDeDados.head(10)) # faz a leitura das 10 primeiras linhas do DataFrame
#aqui podemos verficar se os dados foram tratados/separados corretamente nas suas respectivas "colunas"

#ANÁLISE EXPLORATÓRIA
dadosColuna = ['idade', 'renda_anual', 'pontuacao_gastos'] #criei o array so para organizar o codigo
#print(baseDeDadosClientes[dadosColuna].describe()) #retorna estatisticas sobre os dados analisados, exemplo:
#media, desvio padrao, valor mínimo, valor máximo, mediana.....

#PRÉ PROCESSAMENTO DOS DADOS
#Criando o padronizador de dados
padronizador = StandardScaler()

#aplicando o padronizador nos dados de nosso interesse
dados_padronizados = padronizador.fit_transform(baseDeDadosClientes[dadosColuna]) #fit transforma = padronizar dados
#aqui passamos para função "fit_trasnform" a base de dados "baseDeDadosClientes" e as colunas de nosso interesse
#'idade', 'renda_atual' e 'pontuação_gastos' todas elas armazenadas na variavel "dados coluna".

#print(dados_padronizados) #vizualização dos dados após padronização


#CONSTRUÇÃO DO MODELO DE MACHINE LEARNING PARA SEGMENTAÇÃO DOS DADOS

numeroDeclusters = 3 #definção dos numeros de cluster(grupos) que os dados vão ser separados.

#criando o modelo
modeloMachineLearningClusterizacao = KMeans(n_clusters=numeroDeclusters)

#treinando o modelo
modeloMachineLearningClusterizacao.fit(dados_padronizados) #função "fit" responsável por "clusterizar" os dados
#pede como parâmetro os dados a serem "clusterizados", no nosso caso, os dados são os "dados_padronizados", que
#padronizamos anteriormente usando StandardScaler

#criando a "coluna" a onde sera indicado o cluster(grupo) daquele dado(linha da tabela)
baseDeDadosClientes['cluster'] = modeloMachineLearningClusterizacao.labels_ #estamos atribuando a coluna
#"cluster" os dados que foram gerados apartir do modelo de Machine Learning

#verificando o resultado....
print(baseDeDadosClientes.head(10))

#funcionou

#salvando o resultado em um arquivo...
baseDeDadosClientes.to_csv('dados/dados_clientes_segmentados.csv',index = False) 