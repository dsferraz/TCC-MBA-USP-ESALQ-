# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 10:18:26 2024

@author: dougl
"""

import pandas as pd
import requests
import numpy as np





caminho_arquivo_2 = 'C:/Users/dougl/OneDrive/Desktop/TCC PóS/dados/base_completa.xlsx'
df = pd.read_excel(caminho_arquivo_2)
df.set_index("data",inplace=True)


relacao =  pd.read_excel('C:/Users/dougl/OneDrive/Desktop/TCC PóS/dados/relacao.xlsx')
relacao["serie"] = relacao["serie"].astype(str)

relacao_final = relacao[relacao["serie"].isin(df.columns)].drop_duplicates()


len(relacao_final["serie"].drop_duplicates())
duplicadas = relacao_final["serie"][relacao_final["serie"].duplicated()].drop_duplicates()




 #PRÉ-PROCESSAMENTO
#Retirando variáveis com muitos nulos

# AJUSTANDO VARIÁVEIS COM DISPONIBILIDADE DEFASADA COM A VARIÁVEL ALVO
def contar_nulos_apos_ultimo_nao_nulo(df):
    colunas = []
    contagem_nulos = []
    for coluna in df.columns:
        ultima_observacao_nao_nula = df[coluna].last_valid_index()
        if ultima_observacao_nao_nula is not None:
            nulos_apos_ultimo_nao_nulo = len(df.loc[ultima_observacao_nao_nula:].iloc[1:])
        else:
            nulos_apos_ultimo_nao_nulo = 0
        colunas.append(coluna)
        contagem_nulos.append(nulos_apos_ultimo_nao_nulo)
    
    # Criar um DataFrame com os resultados
    resultado_df = pd.DataFrame({'Coluna': colunas, 'Nulos_Apos_Ultimo_Nao_Nulo': contagem_nulos})
    return resultado_df



base = df.loc[:"2025-01-01"]


contagem_nulos = contar_nulos_apos_ultimo_nao_nulo(base)

series_defasadas = contagem_nulos[contagem_nulos['Nulos_Apos_Ultimo_Nao_Nulo']>6]
series_disponíveis = contagem_nulos[contagem_nulos['Nulos_Apos_Ultimo_Nao_Nulo']<7]

base2 = base.drop(series_defasadas["Coluna"], axis=1)
base3 = base2.copy()


#Marca série defasada
relacao_final.loc[relacao_final["serie"].isin(series_defasadas['Coluna']), "Serie_defasa"] = 1


#Alinha séries com defasagem inferior as 6 meses
for s in series_disponíveis["Coluna"]:
    defasagem = int(series_disponíveis[series_disponíveis["Coluna"] == s]['Nulos_Apos_Ultimo_Nao_Nulo'])
    if defasagem > 0:
        base3[s] = base3[s].shift(defasagem)
        base3.rename(columns={s:s+"_"+str(defasagem)}, inplace=True)
    else:
        continue
    
 
    
tamanho_series = pd.DataFrame()
for column in base3.columns:
    coluna = column 
    series = base3[column].dropna()
    nova_linha = {'nome_variável': column, 'comprimento': len(series)}
    tamanho_series = pd.concat([tamanho_series, pd.DataFrame([nova_linha])], ignore_index=True)


series_curtas = tamanho_series[tamanho_series["comprimento"] < 12]["nome_variável"]

#Marca séries curtas
relacao_final.loc[relacao_final["serie"].isin(series_curtas), "Serie_curta"] = 1
relacao_final.loc[relacao_final["serie"]== "28866"] = 1

#Salva relação de variáveis
relacao_final = relacao_final.drop_duplicates(subset=["serie"])
relacao_final.to_excel('C:/Users/dougl/OneDrive/Desktop/TCC PóS/dados/relacao_variaveis.xlsx')


#Remove series muito curtas
base4 = base3.copy()
base4 = base4.drop(columns=list(series_curtas))
base4 = base4.drop(columns="28866") #só possui zeros 




#AJUSTE DE VARIÁVEL ALVO E INSERÇÃO DE LAGS
base5 = base4.copy()
alvo = "21033"



#Insere 12 lags para cada variável
cols = base5.columns
lag = 12
for i in range(1, lag):
    for col in cols:
        base5[f'{col}_lag{i}'] = base5[col].shift(i)



# FIM DO PRE-PROCESSAMENTO




#Estimação de modelos
# XGBOOST
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error

#RANDOM SEARCH
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Criando o modelo XGBoost para regressão
model = XGBRegressor()

# Definição do grid de hiperparâmetros com intervalos variados
param_dist = {
    'max_depth': np.arange(3, 10, 2),  
    'learning_rate': np.linspace(0.01, 0.3, 5),  
    'n_estimators': np.arange(100, 500, 100), 
    'subsample': np.linspace(0.5, 1.0, 4), 
    'colsample_bytree': np.linspace(0.3, 1.0, 4),  
    'min_child_weight': [1, 3, 5, 7],  
    'gamma': np.linspace(0, 0.5, 4)  
}



#Forecast movel XGBOOST

alv = "21033"

reg = xgb.XGBRegressor()
tscv = TimeSeriesSplit(n_splits=3) #validação cruzada  3 partições sequenciais para o Random Search

#Cria função para realizar forecast movel e validação cruzada de séries temporais e calculo dos erros.
def forecast_xgb(base, min_treino, horizonte):
    analise = pd.DataFrame()
    base = base.copy()
    base["Alvo"] = base[alv].shift((-1)*horizonte)
    base = base.loc[base["Alvo"].dropna().index]

    b_search = base.dropna(subset=["Alvo"])
    b_search= b_search.dropna(axis=1)
    b_search = b_search.iloc[:min_treino,:]
    x_search =  b_search.drop(["Alvo"], axis=1)
    y_search =  b_search["Alvo"]
    # RANDOM SEARCH
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,  # Número de combinações aleatórias a serem testadas
        cv=tscv,  # Validação cruzada com 5 folds
        scoring='neg_mean_squared_error',  # Métrica para regressão
        n_jobs=-1,  # Utiliza múltiplos núcleos para acelerar a busca
        random_state=42  # Garante resultados reprodutíveis
    )
    random_search.fit(x_search, y_search)
    parametros_grid = random_search.best_params_
    modelo_xgb = xgb.XGBRegressor(**parametros_grid)
    for i in range(len(base) - min_treino - horizonte-1):
        #BASE DE TREINAMENTO
        treinamento =  base.iloc[:min_treino+i]
        x_treinamento = treinamento.drop(["Alvo"], axis=1)
        y_treinamento = treinamento[["Alvo"]]

        #BASE TESTE
        base_teste  = base.iloc[[min_treino+i+horizonte+1],:]
        x_teste = base_teste.drop(["Alvo"], axis=1) 
        y_teste = base_teste[["Alvo"]]
        
        #Treinamento
        modelo_treinado = modelo_xgb.fit(x_treinamento, y_treinamento)
        #pred1 = modelo_xgb.predict(x_teste)
        pred1 = modelo_treinado.predict(x_teste)
        modelo = pd.DataFrame()
        modelo["Realizado"] = y_teste.copy()
        modelo["Previsão"] = pred1   
        modelo["Data"] = modelo.index
        modelo["Horizonte_prj"] = horizonte
        analise = pd.concat([analise,modelo])
    analise["Erro"] = analise["Previsão"] -  analise["Realizado"] 
    analise["Erro Absoluto"] = abs(analise["Erro"])
    analise["Erro Quadratico"] = (analise["Erro"])**2
    analise["Erro Percentual"] = (analise["Erro"]/analise["Realizado"])*100 
    analise["Erro Percentual Absoluto"] = abs(analise["Erro Percentual"])
    return analise
  



#Cria variáveis dummies mensais
x2 = base5.copy()
x2["mês"] = x2.index.month
x2["mês"] = pd.Categorical(x2["mês"])
x2 = pd.concat([x2, pd.get_dummies(x2["mês"], prefix="mes_")], axis=1)
x2= x2.drop("mês",axis=1)




#arquivos para salvamento
arquivo_excel = 'C:/Users/dougl/OneDrive/Desktop/TCC PóS/dados/XGBOOST_21033_FE.xlsx'
arquivo_covid = 'C:/Users/dougl/OneDrive/Desktop/TCC PóS/dados/XGBOOST_21033_covid_FE.xlsx'

# Iteração para realizar a estimação e calculo dos erros para multiplos horizontes
forecast = pd.DataFrame()
hrztes = [1,3,6,12,18]
for x in hrztes :
    horizonte = x
    Teste_estimacao = forecast_xgb(x2, 36, horizonte)
    Teste_estimacao.index = Teste_estimacao.index + pd.DateOffset(months=horizonte)
    desc_test= Teste_estimacao.describe()
    desc_test_covid= Teste_estimacao.loc[:"2019-12-01"].describe()
    tst = pd.DataFrame()
    tst[f'horizonte-{horizonte}'] = Teste_estimacao["Previsão"]
    forecast = pd.concat([forecast,tst],axis=1)
    #Salva em abas diferentes do excel
    if x == hrztes[0]:
        desc_test.to_excel(arquivo_excel, sheet_name = f"horizonte-{horizonte}")
        desc_test_covid.to_excel(arquivo_covid, sheet_name = f"horizonte-{horizonte}")
    else:
    # Abrindo o arquivo existente com o openpyxl e adicionando uma nova aba
        with pd.ExcelWriter(arquivo_excel, engine="openpyxl", mode="a") as writer:
            # Nome da nova aba
            desc_test.to_excel(writer, sheet_name = f"horizonte-{horizonte}")
            
        with pd.ExcelWriter(arquivo_covid , engine="openpyxl", mode="a") as writer:
            # Nome da nova aba
            desc_test_covid.to_excel(writer, sheet_name = f"horizonte-{horizonte}")



#salva previsões no excel
with pd.ExcelWriter(arquivo_excel, engine="openpyxl", mode="a") as writer:
    # Nome da nova aba
    forecast.to_excel(writer, sheet_name = "Previsões XGBoost")
    

#RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


#Parametros para o GRIDSEACH
param_grid = {
    'bootstrap': [True,False],
    'max_depth': [50, 80, 300],
    'max_features': ["log2",1],
    'min_samples_leaf': [2,5,8],
    'min_samples_split': [6,10,12],
    'n_estimators': [100,1000]}



#Cria função para treinamento e teste de modelo utilizando validação cruzada de séries temporais
regrf = RandomForestRegressor()
tscv = TimeSeriesSplit(n_splits=3)

alv = "21033"
def forecast_rf(base, min_treino, horizonte):
    analise = pd.DataFrame()
    base = base.copy()
    base["Alvo"] = base[alv].shift((-1)*horizonte)
    #base = base.dropna(subset=["Alvo"])
    # GRID SEARCH
    b_search = base.dropna(subset=["Alvo"])
    b_search= b_search.dropna(axis=1)
    b_search = b_search.iloc[:min_treino,:]
    x_search =  b_search.drop(["Alvo"], axis=1)
    y_search =  b_search["Alvo"]
    grid_search_rf = GridSearchCV(estimator = regrf , param_grid = param_grid, 
                             cv = tscv, n_jobs = -1, verbose = 0, scoring='neg_root_mean_squared_error')
    grid_search_rf.fit(x_search,y_search)
    parametros_grid = grid_search_rf.best_params_
    regrf1 = RandomForestRegressor(**parametros_grid) 
    for i in range(len(base) - min_treino - horizonte - 1):
        #BASE DE TREINAMENTO
        treinamento =  base.iloc[:min_treino+i]
        x_treinamento = treinamento.drop(["Alvo"], axis=1)
        y_treinamento = treinamento["Alvo"]
        
        #BASE TESTE
        base_teste  = base.iloc[[min_treino+i+horizonte+1],:]
        x_teste = base_teste.drop(["Alvo"], axis=1)
        y_teste = base_teste[["Alvo"]]
        
        #Treinamento
        modelo_treinado = regrf1.fit(x_treinamento, y_treinamento)
        modelo = pd.DataFrame()
        modelo["Realizado"] = y_teste.copy()
 #       modelo["Previsão"] = regrf1.predict(x_teste)
        modelo["Previsão"] = modelo_treinado.predict(x_teste)
        modelo["Data"] = modelo.index
        modelo["Horizonte_prj"] = modelo.reset_index().index + 1
        analise = pd.concat([analise,modelo])
    analise["Erro"] = analise["Previsão"] -  analise["Realizado"] 
    analise["Erro Absoluto"] = abs(analise["Erro"])
    analise["Erro Quadratico"] = (analise["Erro"])**2
    analise["Erro Percentual"] = (analise["Erro"]/analise["Realizado"])*100 
    analise["Erro Percentual Absoluto"] = abs(analise["Erro Percentual"])
    return analise

#base6 = base5.copy()
#base6.columns = base6.columns.astype(str)

#Elimina Nulos
# Pacote Random Forest não suporta nulos
x3 = x2.dropna(subset=[alv])
x3 = x3.dropna(axis=1)



#Arquivos para salvar no excel
arquivo_excel = 'C:/Users/dougl/OneDrive/Desktop/TCC PóS/dados/RF_21033_reteste.xlsx'
arquivo_covid = 'C:/Users/dougl/OneDrive/Desktop/TCC PóS/dados/RF_21033_covid_reteste.xlsx'


#Cria iteração para realizar a estimação e o calculo de erros para múltiplos horizontes
forecastrf = pd.DataFrame()
hrztes = [18]

for x in hrztes :
    horizonte = x
   
    #Estimação
    
    Teste_estimacao = forecast_rf(x3, 36,horizonte)
    Teste_estimacao.index = Teste_estimacao.index + pd.DateOffset(months=horizonte)
    desc_test= Teste_estimacao.describe()
    desc_test_covid= Teste_estimacao.loc[:"2019-12-01"].describe()
    tst = pd.DataFrame()
    tst[f'horizonte-{horizonte}'] = Teste_estimacao["Previsão"]
    forecastrf = pd.concat([forecastrf,tst],axis=1)
    
    if x == hrztes[0]:
        desc_test.to_excel(arquivo_excel, sheet_name = f"horizonte-{horizonte}")
        desc_test_covid.to_excel(arquivo_covid, sheet_name = f"horizonte-{horizonte}")
    else:
    # Abrindo o arquivo existente com o openpyxl e adicionando uma nova aba
        with pd.ExcelWriter(arquivo_excel, engine="openpyxl", mode="a") as writer:
            # Nome da nova aba
            desc_test.to_excel(writer, sheet_name = f"horizonte-{horizonte}")
            
        with pd.ExcelWriter(arquivo_covid , engine="openpyxl", mode="a") as writer:
            # Nome da nova aba
            desc_test_covid.to_excel(writer, sheet_name = f"horizonte-{horizonte}")




forecastrf.to_excel('C:/Users/dougl/OneDrive/Desktop/TCC PóS/dados/Previsões_RF_fe.xlsx')


relacao_final.to_excel('C:/Users/dougl/OneDrive/Desktop/TCC PóS/dados/relacao_variaveis.xlsx')






