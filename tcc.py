#!/usr/bin/env python
# coding: utf-8

# # Previsão da qualidade de vinhos através de técnicas de Machine Learning
# 
# 

# ## Importando os pacotes necessários e os dados
# 

# In[1]:


# Importando as bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Importando as bibliotecas para a construção dos modelos de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from imblearn.under_sampling import RandomUnderSampler

from yellowbrick.classifier import ROCAUC

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Configurando o notebook
%matplotlib inline
sns.set(style='white')



# Carregar os dados dos vinhos tintos
dados_vinhos_tintos = pd.read_csv("winequality-red.csv")
dados_vinhos_tintos["Tipo"] = 1  # Adicionar coluna "Tipo" para identificar o tipo de vinho 1 = Tinto

# Carregar os dados dos vinhos brancos
dados_vinhos_brancos = pd.read_csv("winequality-white.csv")
dados_vinhos_brancos["Tipo"] = 0  # Adicionar coluna "Tipo" para identificar o tipo de vinho 0 = Branco

print(dados_vinhos_tintos.head())
print(dados_vinhos_brancos.head())

# In[2]:


# Junção dos datasets de vinhos tintos e brancos
dados_vinhos = pd.concat([dados_vinhos_tintos, dados_vinhos_brancos], ignore_index=True)

# Verificar a quantidade de registros no dataset unificado
print("Quantidade de registros de vinhos:", dados_vinhos.shape[0])

# In[3]:


# Visualizar a estrutora do DataFrame
dados_vinhos.info()

# In[4]:


# Separar as colunas corretamente usando o ponto e vírgula como separador
vinhos_total = dados_vinhos["fixed acidity;\"volatile acidity\";\"citric acid\";\"residual sugar\";\"chlorides\";\"free sulfur dioxide\";\"total sulfur dioxide\";\"density\";\"pH\";\"sulphates\";\"alcohol\";\"quality\""].str.split(";", expand=True)

# Renomear as colunas
vinhos_total.columns = [
    "acidez fixa",
    "acidez volátil",
    "ácido cítrico",
    "açúcar residual",
    "cloretos",
    "dióxido de enxofre livre",
    "dióxido de enxofre total",
    "densidade",
    "pH",
    "sulfatos",
    "álcool",
    "qualidade"
]

# Adicionar a coluna "Tipo" aos dados separados
vinhos_total["tipo"] = dados_vinhos["Tipo"]

# In[5]:


# Exibir as primeiras linhas do DataFrame com as colunas renomeadas
vinhos_total.head()

# # Primeiras entradas e dimensões do conjunto de dados

# In[6]:


# Dimensões do dataset
print("Dimensões do conjunto de dados:\n{} linhas e {} colunas\n".format(vinhos_total.shape[0], vinhos_total.shape[1]))

# Primeiras entradas do dataset
print("Primeiras entradas:")
vinhos_total.head()

# # Análise Exploratória dos Dados
# 
# Nesta seção, vou verificar a integridade e a usabilidade do conjunto de dados, verificando diferentes características do conjunto. Para isso, irei mostrar o nome dos atributos, se há valores ausentes e o tipo de cada coluna.

# In[7]:


print("Nome dos atributos:\n{}".format(vinhos_total.columns.values))
print("\nQuantidade de valores ausentes por atributo:\n{}".format(vinhos_total.isnull().sum()))
print("\nTipo de cada atributo:\n{}".format(vinhos_total.dtypes))

# In[8]:


#Verificar presença de nulos
vinhos_total.isnull().any()

# Não há valores ausentes no conjunto de dados, com isso não será necessário fazer um tratamento nesse sentido.
# Porem as features (variáveis independentes) e a variável classe (qualidade) são do tipo object, sendo necessário tratá-las, convertendo-as em tipo int64.

# In[9]:


# Converter colunas relevantes para tipos numéricos
colunas_numericas = ['acidez fixa', 'acidez volátil', 'ácido cítrico', 'açúcar residual', 'cloretos', 'dióxido de enxofre livre', 'dióxido de enxofre total', 'densidade', 'pH', 'sulfatos', 'álcool', 'qualidade']
vinhos_total[colunas_numericas] = vinhos_total[colunas_numericas].astype(float)

print("\nTipo de cada atributo:\n{}".format(vinhos_total.dtypes))

# In[10]:


#Analisar o dataset
#O objetivo é entender se podemos considerar o dataset como um todo ou se devemos observá-los por tipo de vinho para isso iremos agregar os dados por tipo de vinho e ver como as variáveis se comportam

agrupado_tipo = vinhos_total.groupby('tipo').std()
print(agrupado_tipo)

# Existem diferenças significativas considerando o "desvio padrão" agrupados por tipo de vinhos, isto é, existe diferenças nas caracteristicas de cada tipo de vinho, visto pela analise fisico-quimica de cada tipo, ou seja, a qualidade são determinadas por composições fisico-quimica diefrentes sem interferir na qualidade. Observações:
# 
#  - Acidez fixa é quase o dobro em vinhos Tintos;
#  - Acidez Volatil é maior 0.7 desvios em Tintos;
#  - Ácido Cítrico é quase 4 desvios maior em Brancos;
#  - Cloretos maior que 2 desvios em Tintos;
#  - Dióxido de enxofre livre e Dióxido de enxofre total são maiores em Brancos;
#  - Densidade é maior em Brancos.

# # Resumo estatístico

# In[12]:


vinhos_total.describe()

# Pelo resumo estatístico, podemos ver que a variável qualidade varia de 3 a 9, mesmo ela podendo variar de 0 a 10, ou seja, nenhum vinho foi tão ruim a ponto de receber 0 ou tão bom a ponto de receber 10. Também percebemos que, possivelmente, há outliers nas variáveis: 'acidez volátil', 'açúcar residual', 'cloretos', 'dióxido de enxofre livre', 'dióxido de enxofre total' e 'sulfatos'.
# 
# Com isso, vamos separar a variável qualidade em apenas duas categorias, sendo que os vinhos que receberam notas menor ou igual a 6 serão considerados ruins e serão definidos como 0, já os que receberam nota maior que 6 serão considerados bons e serão definidos como 1.

# In[13]:


# Categorizando a variável qualidade e criando a variável qualidade_cat
vinhos_total['qualidade_cat'] = pd.cut(vinhos_total['qualidade'], bins=(2, 6.5, 8), labels = [0, 1])

# In[14]:


# Verificando quais as entradas únicas da variável qualidade_cat
vinhos_total['qualidade_cat'] = vinhos_total['qualidade_cat'].astype('category')
vinhos_total['qualidade_cat'].unique()

# Verificação e tratamento dos *outliers*
# Nesta seção vou fazer a verificação e tratamento 

# In[15]:


# Criar um histograma das colunas numéricas
vinhos_total.hist(bins=10, figsize=(12, 8))
plt.tight_layout()
plt.show()

# A maioria dos histogramas apresenta uma distribuição normal entretanto não centralizado o que pode indicar a presença de outliers, tais como os campos 'acidez volátil', 'açúcar residual', 'cloretos', 'dióxido de enxofre livre' e 'dióxido de enxofre total' possuem um desvio padrão acima das demais variaveis. 
# Gerar boxplot para cada uma das variáveis para verificar  

# In[16]:


# Configurando o plot
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))

sns.boxplot(vinhos_total['acidez volátil'], ax=ax[0, 0])
sns.boxplot(vinhos_total['açúcar residual'], ax=ax[0, 1])
sns.boxplot(vinhos_total['cloretos'], ax=ax[1, 0])
sns.boxplot(vinhos_total['dióxido de enxofre livre'], ax=ax[1, 1])
sns.boxplot(vinhos_total['dióxido de enxofre total'], ax=ax[2, 0])
sns.boxplot(vinhos_total['sulfatos'], ax=ax[2, 1])

plt.tight_layout()

# Pelos boxplots, as variáveis com a maior quantidade de outliers são "açúcar residual", cloretos e sulfatos.Sendo assim, vamos calcular o  "intervalo interquartil" (Interquartile Range - IQR) para essas três variáveis e definir o limite de corte para eliminar os outliers.

# In[17]:


# Identificando os outliers para a variável residual sugar
q1_rsugar = vinhos_total['açúcar residual'].quantile(0.25)
q3_rsugar = vinhos_total['açúcar residual'].quantile(0.75)
IQR_rsugar = q3_rsugar - q1_rsugar

print("IQR da variável açúcar residual: {}\n".format(round(IQR_rsugar, 2)))

# Definindo os limites para a variável residual sugar
sup_rsugar = q3_rsugar + 1.5 * IQR_rsugar
inf_rsugar = q1_rsugar - 1.5 * IQR_rsugar

print("Limite superior de açúcar residual: {}".format(round(sup_rsugar, 2)))
print("Limite inferior de açúcar residual: {}".format(round(inf_rsugar, 2)))

# In[18]:


cut_rsugar = len(vinhos_total[vinhos_total['açúcar residual'] < 0.85]) + len(vinhos_total[vinhos_total['açúcar residual'] > 3.65])

print("As entradas da variável açúcar residual fora dos limites representam {} % do dataset.\n".format(round((cut_rsugar / vinhos_total.shape[0]) * 100, 2)))

# In[19]:


# Identificando os outliers para a variável cloretos
q1_chlo = vinhos_total['cloretos'].quantile(0.25)
q3_chlo = vinhos_total['cloretos'].quantile(0.75)
IQR_chlo = q3_chlo - q1_chlo

print("IQR da variável cloretos: {}\n".format(round(IQR_chlo, 2)))

# Definindo os limites para a variável cloretos
sup_chlo = q3_chlo + 1.5 * IQR_chlo
inf_chlo = q1_chlo - 1.5 * IQR_chlo

print("Limite superior de cloretos: {}".format(round(sup_chlo, 2)))
print("Limite inferior de cloretos: {}".format(round(inf_chlo, 2)))

# In[20]:


cut_chlo = len(vinhos_total[vinhos_total['cloretos'] < 0.04]) + len(vinhos_total[vinhos_total['cloretos'] > 0.12])

print("As entradas da variável cloretos fora dos limites representam {} % do dataset.\n".format(round((cut_chlo / vinhos_total.shape[0]) * 100, 2)))

# In[21]:


# Identificando os outliers para a variável sulfatos
q1_sulp = vinhos_total['sulfatos'].quantile(0.25)
q3_sulp = vinhos_total['sulfatos'].quantile(0.75)
IQR_sulp = q3_sulp - q1_sulp

print("IQR da variável sulfatos: {}\n".format(round(IQR_sulp, 2)))

# Definindo os limites para a variável sulfatos
sup_sulp = q3_sulp + 1.5 * IQR_sulp
inf_sulp = q1_sulp - 1.5 * IQR_sulp

print("Limite superior de sulfatos: {}".format(round(sup_sulp, 2)))
print("Limite inferior de sulfatos: {}".format(round(inf_sulp, 2)))

# In[22]:


cut_sulp = len(vinhos_total[vinhos_total['sulfatos'] < 0.28]) + len(vinhos_total[vinhos_total['sulfatos'] > 1.0])

print("As entradas da variável sulfatos fora dos limites representam {} % do dataset.\n".format(round((cut_sulp / vinhos_total.shape[0]) * 100, 2)))

# Remover as entradas que estão fora dos limites de corte das variáveis residual sugar, cloretos e sulfatos:

# In[23]:


vinhos_total_clean = vinhos_total.copy()

vinhos_total_clean.drop(vinhos_total_clean[vinhos_total_clean['açúcar residual'] > 3.65].index, axis=0, inplace=True)
vinhos_total_clean.drop(vinhos_total_clean[vinhos_total_clean['açúcar residual'] < 0.85].index, axis=0, inplace=True)
vinhos_total_clean.drop(vinhos_total_clean[vinhos_total_clean['cloretos'] > 0.12].index, axis=0, inplace=True)
vinhos_total_clean.drop(vinhos_total_clean[vinhos_total_clean['cloretos'] < 0.04].index, axis=0, inplace=True)
vinhos_total_clean.drop(vinhos_total_clean[vinhos_total_clean['sulfatos'] > 1.0].index, axis=0, inplace=True)
vinhos_total_clean.drop(vinhos_total_clean[vinhos_total_clean['sulfatos'] < 0.28].index, axis=0, inplace=True)

# Plotando o boxplot para as variáveis:

# In[24]:


# Configurando o plot
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
fig.delaxes(ax[1,1])

sns.boxplot(vinhos_total_clean['açúcar residual'], ax=ax[0, 0])
sns.boxplot(vinhos_total_clean['cloretos'], ax=ax[0, 1])
sns.boxplot(vinhos_total_clean['sulfatos'], ax=ax[1, 0])

plt.tight_layout()

# Os boxplots mostram alguns outliers, no entanto, esses estão sendo calculados em relação ao novo dataset e a limpeza levou em conta os dados originais.

# # Verificação do balanceamento dos dados
# Como é citado na descrição do conjunto de dados, a variável classe (qualidade) está desbalanceada. Utilizando um gráfico, fica mais fácil percebemos este desbalanceamento.

# In[25]:


# Construindo o gráfico
fig, ax = plt.subplots(figsize=(10,8))

sns.countplot(vinhos_total_clean['qualidade'], color='b', ax=ax)
sns.despine()

plt.tight_layout()

# Pelo gráfico, fica fácil perceber que a maioria dos notas ficou em 5 ou 6.
# 
# Vamos analisar o balanceamento para a variável qualidade_cat.

# In[26]:


# Porcentagem de cada classe
print("A classe 0 representa {} % de todas as entradas.".format(round((len(vinhos_total_clean[vinhos_total_clean['qualidade_cat'] == 0]) / vinhos_total_clean.shape[0]) * 100, 2)))
print("A classe 1 representa {} % de todas as entradas.".format(round((len(vinhos_total_clean[vinhos_total_clean['qualidade_cat'] == 1]) / vinhos_total_clean.shape[0]) * 100, 2)))

# Pela porcentagem de cada classe, percebemos que a classe 1 representa apenas 14.69% do total, isso já é um desbalanceamento considerável. Para uma melhor visualização do desbalanceamento farei um gráfico de barras da variável qualidade_cat.

# In[27]:


# Cronstruindo o gráfico de barras
fig, ax = plt.subplots(figsize=(6,6))

sns.countplot(vinhos_total_clean['qualidade_cat'], color='b', ax=ax)
sns.despine()

ax.set_title("Distribuição das Classes", fontsize=16)
ax.set_xlabel("Classes", fontsize=14)

plt.tight_layout()

# Pela Distribuição de Classes, percebemos claramente o desbalanceamento, tendo muito mais entradas da classe 0 do que da classe 1.

# # Preparação dos dados

# ### Separação dos dados em treino e teste
# 
# Uma etapa de grande importância para a construção de modelos de Machine Learning é separação do conjunto de dados em treino e teste, pois se não fizermos isso é, bem provável, que ocorra overfitting (sobreajuste).

# In[28]:


# Separando os dados entre feature matrix e target vector
X = vinhos_total_clean.drop(['qualidade', 'qualidade_cat'], axis=1)
y = vinhos_total_clean['qualidade_cat'] # Pois usaremos apenas a separação entre "ruim" ou "bom" (0 ou 1)

# Dividindo os dados entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# #### Balanceamento dos dados
# 
# Para realizar o balanceamento dos dados, existem vários métodos complexos tais como Recognition-based Learning e Cost-sensitive Learning e métodos mais simples que vem sendo amplamente utilizados com ótimos resultados como Over-sampling e Under-sampling.
# 
# Neste projeto, será utilizado o método Under-sampling que foca na classe majoritário para balancear o conjunto de dados, ou seja, elimina aleatoriamente entradas da classe com maior quantidade de ocorrências.
# 
# Assim:

# In[29]:


# Definindo o modelo para balancear
und = RandomUnderSampler()

X_und, y_und = und.fit_resample(X_train, y_train)

# Verificando o balanceamento dos dados
print(pd.Series(y_und).value_counts(), "\n")

# Plotando a nova Distribuição de Classes
fig, ax = plt.subplots(figsize=(6,6))

sns.countplot(pd.Series(y_und), color='b', ax=ax)

sns.despine()

ax.set_title("Distribuição das Classes", fontsize=16)
ax.set_xlabel("Classes", fontsize=14)

plt.tight_layout()

# Após o balanceamento, percebemos as classes agora representam 50% cada uma do conjunto de dados, ou seja, não há mais a diferença como havia anteriormente.

# ### Correlação entre os atributos
# 
# Farei um mapa de calor (heatmap) para verificar a relação entre as variáveis utilizando o método de Pearson, essa correlação será feita com os dados antes e depois do balanceamento.

# In[30]:


# Construindo o heatmap
fig, ax = plt.subplots(figsize=(16, 6), nrows=1, ncols=2)
fig.suptitle("Matriz de Correlação")

sns.heatmap(X_train.corr().abs(), cmap='YlGn', linecolor='#eeeeee', linewidths=0.1, ax=ax[0])
ax[0].set_title("Desbalanceado")

sns.heatmap(X_und.corr().abs(), cmap='YlGn', linecolor='#eeeeee', linewidths=0.1, ax=ax[1])
ax[1].set_title("Balanceado")

plt.tight_layout()


# Como podemos perceber pela Matriz de Correlação, a correlação entre as variáveis parece não ter sofrido grandes alterações após o balanceamento.

# Poderia ser utilizado algoritmos supervisionados como o K-means pra predizer em qual categoria um vinho se encontra. Porém, consideramos que isso não faria sentido para rodar os modelos não supervisionados.
# 
# Pode fazer sentido analisar o vinho de maneira separada por tipo já que muitas variáveis tendem a se comportar de forma diferente vamos iniciar a preparação dos dados, separando o dataset em 2.
# 

# In[31]:


dados_base_tinto = vinhos_total[vinhos_total['tipo'] == 0].iloc[:, :15].copy()
dados_base_branco = vinhos_total[vinhos_total['tipo'] != 0].iloc[:, :15].copy()

# # Modelos de *Machine Learning*
# Após a preparação dos dados, vamos construir três modelos de Machine Learning e comparar o desempenho de cada uma deles e em cada base de tipo de vinho (Tinto e Branco). Os modelos utilizados serão:
# 
#  - Random Forest (Floresta Aleatória);
#  - XGBoost;
#  - Regressão Logística.

# ### *Random Forest*
# 
# O Random Forest é um modelo baseado em árvores de decisão e para esses tipos de modelos não é necessário fazer uma padronização dos dados, sendo assim, vamos utilizar o modelo diretamente nos dados balanceados.

# #### Para base dos vinhos Tinto: dados_base_tinto

# In[32]:


# Separando os dados entre feature matrix e target vector
dados_base_tinto_clean = vinhos_total_clean.copy()

X = dados_base_tinto_clean.drop(['qualidade', 'qualidade_cat'], axis=1)
y = dados_base_tinto_clean['qualidade_cat'] # Pois usaremos apenas a separação entre "ruim" ou "bom" (0 ou 1)

# Dividindo os dados entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# In[33]:


# Modelo Random Forest
rf_model = RandomForestClassifier()

# Definindo o melhor parâmetro
parameters = {'n_estimators': range(25, 1000, 25)}

kfold = StratifiedKFold(n_splits=5, shuffle=True)

rf_clf = GridSearchCV(rf_model, parameters, cv=kfold)
rf_clf.fit(X_und, y_und)

# Visualizar o melhor parâmetro
print("Melhor parâmetro: {}".format(rf_clf.best_params_))

# Agora, para o número de estimadores igual a 375, vamos analisar o desempenho do modelo.

# In[34]:


# Definindo o modelo com n_estimators igual a 375
rf_model = RandomForestClassifier(n_estimators = 375)

# Fit do modelo
rf_model.fit(X_und, y_und)

# Testando o modelo
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)


# In[35]:


# Relatório de classificação
print("Relatório de classificação para o Random Forest:\n", classification_report(y_test, y_pred_rf, digits=4))

# Área sob a curva
print("Área sob a curva (AUC):\t{}%".format(round(roc_auc_score(y_test, y_pred_rf) * 100, 2)))

# In[36]:


# Matriz de confusão
fig, ax = plt.subplots()

sns.heatmap(confusion_matrix(y_test, y_pred_rf, normalize='true'), annot=True, ax=ax)

ax.set_title('Matriz de Confusão Normalizada')
ax.set_ylabel('Verdadeiro')
ax.set_xlabel('Previsto')

plt.tight_layout()

# Como para conjunto de dados desbalanceados, a acurácia não é um bom indicador de desempenho. Então, utilizei a Área sob a curva (AUC ROC) que é uma indicador interessante e apresentou um valor de 88.42%.
# 
# Ainda, na Matriz de Confusão Normalizada pode ser visualizada a taxa de acertos.
# 
# Além disso, vou plotar a curva ROC:

# In[44]:


# Converter y_test em valores numéricos
y_test_numeric = y_test.cat.codes

# Treinar o modelo de classificação
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Calcular a probabilidade das classes positivas
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Calcular a curva ROC
fpr, tpr, thresholds = roc_curve(y_test_numeric, y_pred_proba)

# Plotar a curva ROC
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.title('Curva ROC')
plt.show()

# ## XGBoost
# 
# O XGBoost é um framework que utiliza gradient boosting que é uma técnica de aprendizado de máquina para problemas de regressão e classificação, que produz um modelo de previsão na forma de um ensemble de modelos de previsão fracos, geralmente árvores de decisão. Ela constrói o modelo em etapas, como outros métodos de boosting, e os generaliza, permitindo a otimização de uma função de perda diferenciável arbitrária.
# 
# Primeiro, vou ajustar o número de estimadores e para isso vou definir o parâmetro learning_rate = 0.1 e verbosity=0 (O parâmetro verbosity=0 é utilizado para não mostrar mensagens na tela enquanto o código é utilizado).

# In[84]:


# Modelo XGBoost
xgb_model = XGBClassifier(learning_rate=0.1)

# Definindo os melhores parâmetros
param_gs = {'n_estimators': range(0, 1000, 50)}

# Identificando os melhores parâmetros
kfold = StratifiedKFold(n_splits=5, shuffle=True)

xgb_clf = GridSearchCV(xgb_model, param_gs, cv=kfold)
xgb_clf.fit(X_und, y_und)

# Visualizar o melhor parâmetro
print("Melhores parâmetros: {}".format(xgb_clf.best_params_))

# Com o número de estimadores definidos como 100, vou ajustar os parâmetros max_depth e min_child_weight.

# In[88]:


# Modelo XGBoost
xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=100, verbosity=0)

# Definindo os melhores parâmetros
param_gs = {
    'max_depth': range(1, 8, 1), 
    'min_child_weigth': range(1, 5, 1), 
    }

# Identificando os melhores parâmetros
kfold = StratifiedKFold(n_splits=5, shuffle=True)

xgb_clf = GridSearchCV(xgb_model, param_gs, cv=kfold)
xgb_clf.fit(X_und, y_und)

# Visualizar o melhor parâmetro
print("Melhores parâmetros: {}".format(xgb_clf.best_params_))

# Tendo os valores de max_depth = 1 e min_child_weigth = 1, vou oajustar o parâmetro gamma.

# In[89]:


# Modelo XGBoost
xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, min_child_weigth=1, verbosity=0)

# Definindo os melhores parâmetros
param_gs = {'gamma': [i/10.0 for i in range(0,5)]}

# Identificando os melhores parâmetros
kfold = StratifiedKFold(n_splits=5, shuffle=True)

xgb_clf = GridSearchCV(xgb_model, param_gs, cv=kfold)
xgb_clf.fit(X_und, y_und)

# Visualizar o melhor parâmetro
print("Melhores parâmetros: {}".format(xgb_clf.best_params_))

# Como o gamma=0.0, por fim, irei testar quatro valores para learning_rate e qual apresenta o melhor resultado.

# In[91]:


# Modelo XGBoost
xgb_model = XGBClassifier(n_estimators=100, max_depth=3, min_child_weigth=1, gamma=0.0, verbosity=0)

# Definindo os melhores parâmetros
param_gs = {'learning_rate': [0.001, 0.01, 0.1, 1]}

# Identificando os melhores parâmetros
kfold = StratifiedKFold(n_splits=5, shuffle=True)

xgb_clf = GridSearchCV(xgb_model, param_gs, cv=kfold)
xgb_clf.fit(X_und, y_und)

# Visualizar o melhor parâmetro
print("Melhores parâmetros: {}".format(xgb_clf.best_params_))

# Com os parâmetros definidos, vou avaliar o desempenho do modelo final.

# In[92]:


# Modelo XGBoost final
xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, min_child_weigth=1, gamma=0.0, verbosity=0)

# Treinando o modelo
xgb_model.fit(X_und, y_und)

# Fazendo previsões
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)

# In[190]:


# Relatório de classificação
print("Relatório de classificação para o XGBClassifier:\n", classification_report(y_test, y_pred_xgb, digits=4))

# Área sob a curva
print("Área sob a curva (AUC):\t{}%".format(round(roc_auc_score(y_test, y_pred_xgb) * 100, 2)))

# In[93]:


# Matriz de confusão
fig, ax = plt.subplots()

sns.heatmap(confusion_matrix(y_test, y_pred_xgb, normalize='true'), annot=True, ax=ax)

ax.set_title('Matriz de Confusão Normalizada')
ax.set_ylabel('Verdadeiro')
ax.set_xlabel('Previsto')

plt.tight_layout()

# Como para conjunto de dados desbalanceados, a acurácia não é um bom indicador de desempenho. Então, utilizei a Área sob a curva (AUC ROC) que é uma indicador interessante e apresentou um valor de 81.53%.
# 
# Ainda, na Matriz de Confusão Normalizada pode ser visualizada a taxa de acertos.

# In[94]:


# Ajustar o modelo aos dados de treinamento
xgb_model.fit(X_und, y_und)

# Converter y_test para valores numéricos
y_test_numeric = y_test.cat.codes

# Calcular as probabilidades de previsão
y_pred_proba = xgb_model.predict_proba(X_test)

# Calcular a pontuação ROC
fpr, tpr, _ = roc_curve(y_test_numeric, y_pred_proba[:, 1])

# Plotar a curva ROC
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(fpr, tpr, label="XGBClassifier")
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax.set_xlabel("Taxa de Falso Positivo")
ax.set_ylabel("Taxa de Verdadeiro Positivo")
ax.set_title("Curva ROC para o XGBClassifier")
ax.legend()
plt.show()


# ## Regressão Logística
# 
# A regressão logística é uma técnica estatística que tem como objetivo produzir, a partir de um conjunto de observações, um modelo que permita a predição de valores tomados por uma variável categórica, frequentemente binária, a partir de uma série de variáveis explicativas contínuas e/ou binárias.
# 
# Diferentemente dos modelos Random Forest e XGBoost, para a Regressão Logística é necessário padronizar os dados antes de treiná-los no modelo. Primeiro, ajustarei o parâmetro C da Regressão Logísitca.

# In[95]:


# Regressão logística
rl_model = LogisticRegression() 
# Padronizando os dados de treino
scaler = StandardScaler().fit(X_und)
X_und_std = scaler.transform(X_und)

# Definindo o melhor parâmetro
param_rl = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Identificando os melhor parâmetro
kfold = StratifiedKFold(n_splits=5, shuffle=True)

rl_clf = GridSearchCV(rl_model, param_rl, cv=kfold)
rl_clf.fit(X_und_std, y_und)

# Visualizar o melhor parâmetro
print("Melhor parâmetro: {}".format(rl_clf.best_params_))

# Com o parâmetro ajustado, vou avaliar o desmepenho do modelo e para auxiliar vou criar um fluxo de trabalho utilizando uma pipeline.

# In[96]:


# Regressão Logística com pipeline
rl_model = make_pipeline(StandardScaler(), LogisticRegression(C=1))

# Treinando o modelo
rl_model.fit(X_und, y_und)

# Fazendo previsões
y_pred_rl = rl_model.predict(X_test)
y_prob_rl = rl_model.predict_proba(X_test)

# In[98]:


# Relatório de classificação
print("Relatório de classificação para a Regressão Logística:\n", classification_report(y_test, y_pred_rl, digits=4))

# Área sob a curva
print("Área sob a curva (AUC):\t{}%".format(round(roc_auc_score(y_test, y_pred_rl) * 100, 2)))


# In[99]:


# Matriz de confusão
fig, ax = plt.subplots()

sns.heatmap(confusion_matrix(y_test, y_pred_rl, normalize='true'), annot=True, ax=ax)

ax.set_title('Matriz de Confusão Normalizada')
ax.set_ylabel('Verdadeiro')
ax.set_xlabel('Previsto')

plt.tight_layout()

# Como para conjunto de dados desbalanceados, a acurácia não é um bom indicador de desempenho. Então, utilizei a Área sob a curva (AUC ROC) que é uma indicador interessante e apresentou um valor de 76.13%.
# 
# Ainda, na Matriz de Confusão Normalizada pode ser visualizada a taxa de acertos.

# In[100]:


# Ajustar o modelo aos dados de treinamento
rl_model.fit(X_und, y_und)

# Converter y_test para valores numéricos
y_test_numeric = y_test.cat.codes

# Calcular as probabilidades de previsão
y_pred_proba = rl_model.predict_proba(X_test)

# Calcular a pontuação ROC
fpr, tpr, _ = roc_curve(y_test_numeric, y_pred_proba[:, 1])

# Plotar a curva ROC
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(fpr, tpr, label="Regressão Logística")
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax.set_xlabel("Taxa de Falso Positivo")
ax.set_ylabel("Taxa de Verdadeiro Positivo")
ax.set_title("Curva ROC para a Regressão Logística")
ax.legend()
plt.show()


# # Random Forest

# ### Para base dos vinhos Branco: dados_base_branco

# In[101]:


dados_base_branco_clean = vinhos_total_clean.copy()

X = dados_base_branco_clean.drop(['qualidade', 'qualidade_cat'], axis=1)
y = dados_base_branco_clean['qualidade_cat'] # Pois usaremos apenas a separação entre "ruim" ou "bom" (0 ou 1)

# Dividindo os dados entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
rf_model = RandomForestClassifier()
# Definindo o melhor parâmetro
parameters = {'n_estimators': range(25, 1000, 25)}

kfold = StratifiedKFold(n_splits=5, shuffle=True)

rf_clf = GridSearchCV(rf_model, parameters, cv=kfold)
rf_clf.fit(X_und, y_und)

# Visualizar o melhor parâmetro
print("Melhor parâmetro: {}".format(rf_clf.best_params_))

# Agora, para o número de estimadores igual a 750, vamos analisar o desempenho do modelo.

# In[102]:


# Definindo o modelo com n_estimators igual a 750
rf_model = RandomForestClassifier(n_estimators = 750)

# Fit do modelo
rf_model.fit(X_und, y_und)

# Testando o modelo
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)

# In[165]:


# Relatório de classificação
print("Relatório de classificação para o Random Forest:\n", classification_report(y_test, y_pred_rf, digits=4))

# Área sob a curva
print("Área sob a curva (AUC):\t{}%".format(round(roc_auc_score(y_test, y_pred_rf) * 100, 2)))

# In[103]:


# Matriz de confusão
fig, ax = plt.subplots()

sns.heatmap(confusion_matrix(y_test, y_pred_rf, normalize='true'), annot=True, ax=ax)

ax.set_title('Matriz de Confusão Normalizada')
ax.set_ylabel('Verdadeiro')
ax.set_xlabel('Previsto')

plt.tight_layout()

# Como para conjunto de dados desbalanceados, a acurácia não é um bom indicador de desempenho. Então, utilizei a Área sob a curva (AUC ROC) que é uma indicador interessante e apresentou um valor de 84.91%.
# 
# Ainda, na Matriz de Confusão Normalizada pode ser visualizada a taxa de acertos.
# 
# Além disso, vou plotar a curva ROC:

# In[104]:


# Converter y_test para valores numéricos
y_test_numeric = y_test.cat.codes

# Calcular a pontuação com os dados de teste
vis_rf.score(X_test, y_test_numeric)

# Mostrar a figura
vis_rf.show()

# ## XGBoost
# 
# O XGBoost é um framework que utiliza gradient boosting que é uma técnica de aprendizado de máquina para problemas de regressão e classificação, que produz um modelo de previsão na forma de um ensemble de modelos de previsão fracos, geralmente árvores de decisão. Ela constrói o modelo em etapas, como outros métodos de boosting, e os generaliza, permitindo a otimização de uma função de perda diferenciável arbitrária.
# 
# Primeiro, vou ajustar o número de estimadores e para isso vou definir o parâmetro learning_rate = 0.1 e verbosity=0 (O parâmetro verbosity=0 é utilizado para não mostrar mensagens na tela enquanto o código é utilizado).

# In[105]:


xgb_model = XGBClassifier(learning_rate=0.1)

# Definindo os melhores parâmetros
param_gs = {'n_estimators': range(0, 1000, 50)}

# Identificando os melhores parâmetros
kfold = StratifiedKFold(n_splits=5, shuffle=True)

xgb_clf = GridSearchCV(xgb_model, param_gs, cv=kfold)
xgb_clf.fit(X_und, y_und)

# Visualizar o melhor parâmetro
print("Melhores parâmetros: {}".format(xgb_clf.best_params_))

# Com o número de estimadores definidos como 400, vou ajustar os parâmetros max_depth e min_child_weight.

# In[106]:


xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=400, verbosity=0)

# Definindo os melhores parâmetros
param_gs = {
    'max_depth': range(1, 8, 1), 
    'min_child_weigth': range(1, 5, 1), 
    }

# Identificando os melhores parâmetros
kfold = StratifiedKFold(n_splits=5, shuffle=True)

xgb_clf = GridSearchCV(xgb_model, param_gs, cv=kfold)
xgb_clf.fit(X_und, y_und)

# Visualizar o melhor parâmetro
print("Melhores parâmetros: {}".format(xgb_clf.best_params_))

# Tendo os valores de max_depth = 2 e min_child_weigth = 1, vou oajustar o parâmetro gamma.

# In[107]:


xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=400, max_depth=2, min_child_weigth=1, verbosity=0)

# Definindo os melhores parâmetros
param_gs = {'gamma': [i/10.0 for i in range(0,5)]}

# Identificando os melhores parâmetros
kfold = StratifiedKFold(n_splits=5, shuffle=True)

xgb_clf = GridSearchCV(xgb_model, param_gs, cv=kfold)
xgb_clf.fit(X_und, y_und)

# Visualizar o melhor parâmetro
print("Melhores parâmetros: {}".format(xgb_clf.best_params_))

# Com o melhor resultado.

# In[108]:


xgb_model = XGBClassifier(n_estimators=400, max_depth=2, min_child_weigth=1, gamma=0.4, verbosity=0)

# Definindo os melhores parâmetros
param_gs = {'learning_rate': [0.001, 0.01, 0.1, 1]}

# Identificando os melhores parâmetros
kfold = StratifiedKFold(n_splits=5, shuffle=True)

xgb_clf = GridSearchCV(xgb_model, param_gs, cv=kfold)
xgb_clf.fit(X_und, y_und)

# Visualizar o melhor parâmetro
print("Melhores parâmetros: {}".format(xgb_clf.best_params_))

# In[109]:


xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=400, max_depth=2, min_child_weigth=1, gamma=0.4, verbosity=0)

# Treinando o modelo
xgb_model.fit(X_und, y_und)

# Fazendo previsões
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)

# In[110]:


# Relatório de classificação
print("Relatório de classificação para o XGBClassifier:\n", classification_report(y_test, y_pred_xgb, digits=4))

# Área sob a curva
print("Área sob a curva (AUC):\t{}%".format(round(roc_auc_score(y_test, y_pred_xgb) * 100, 2)))

# In[111]:


fig, ax = plt.subplots()

sns.heatmap(confusion_matrix(y_test, y_pred_xgb, normalize='true'), annot=True, ax=ax)

ax.set_title('Matriz de Confusão Normalizada')
ax.set_ylabel('Verdadeiro')
ax.set_xlabel('Previsto')

plt.tight_layout()

# Como para conjunto de dados desbalanceados, a acurácia não é um bom indicador de desempenho. Então, utilizei a Área sob a curva (AUC ROC) que é uma indicador interessante e apresentou um valor de 82.65%.
# Ainda, na Matriz de Confusão Normalizada pode ser visualizada a taxa de acertos.

# In[112]:


# Ajustar o modelo aos dados de treinamento
xgb_model.fit(X_und, y_und)

# Converter y_test para valores numéricos
y_test_numeric = y_test.cat.codes

# Calcular as probabilidades de previsão
y_pred_proba = xgb_model.predict_proba(X_test)

# Calcular a pontuação ROC
fpr, tpr, _ = roc_curve(y_test_numeric, y_pred_proba[:, 1])

# Plotar a curva ROC
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(fpr, tpr, label="XGBClassifier")
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax.set_xlabel("Taxa de Falso Positivo")
ax.set_ylabel("Taxa de Verdadeiro Positivo")
ax.set_title("Curva ROC para o XGBClassifier")
ax.legend()
plt.show()

# ### Regressão Logística

# In[113]:


rl_model = LogisticRegression() 
# Padronizando os dados de treino
scaler = StandardScaler().fit(X_und)
X_und_std = scaler.transform(X_und)

# Definindo o melhor parâmetro
param_rl = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Identificando os melhor parâmetro
kfold = StratifiedKFold(n_splits=5, shuffle=True)

rl_clf = GridSearchCV(rl_model, param_rl, cv=kfold)
rl_clf.fit(X_und_std, y_und)

# Visualizar o melhor parâmetro
print("Melhor parâmetro: {}".format(rl_clf.best_params_))

# Com o parâmetro ajustado, vou avaliar o desmepenho do modelo e para auxiliar vou criar um fluxo de trabalho utilizando uma pipeline.

# In[114]:


rl_model = make_pipeline(StandardScaler(), LogisticRegression(C=10))

# Treinando o modelo
rl_model.fit(X_und, y_und)

# Fazendo previsões
y_pred_rl = rl_model.predict(X_test)
y_prob_rl = rl_model.predict_proba(X_test)

# In[185]:


# Relatório de classificação
print("Relatório de classificação para a Regressão Logística:\n", classification_report(y_test, y_pred_rl, digits=4))

# Área sob a curva
print("Área sob a curva (AUC):\t{}%".format(round(roc_auc_score(y_test, y_pred_rl) * 100, 2)))

# In[115]:


# Matriz de confusão
fig, ax = plt.subplots()

sns.heatmap(confusion_matrix(y_test, y_pred_rl, normalize='true'), annot=True, ax=ax)

ax.set_title('Matriz de Confusão Normalizada')
ax.set_ylabel('Verdadeiro')
ax.set_xlabel('Previsto')

plt.tight_layout()

# Como para conjunto de dados desbalanceados, a acurácia não é um bom indicador de desempenho. Então, utilizei a Área sob a curva (AUC ROC) que é uma indicador interessante e apresentou um valor de 77.54%.
# 
# Ainda, na Matriz de Confusão Normalizada pode ser visualizada a taxa de acertos.

# In[116]:


# Ajustar o modelo aos dados de treinamento
rl_model.fit(X_und, y_und)

# Converter y_test para valores numéricos
y_test_numeric = y_test.cat.codes

# Calcular as probabilidades de previsão
y_pred_proba = rl_model.predict_proba(X_test)

# Calcular a pontuação ROC
fpr, tpr, _ = roc_curve(y_test_numeric, y_pred_proba[:, 1])

# Plotar a curva ROC
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(fpr, tpr, label="Regressão Logística")
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax.set_xlabel("Taxa de Falso Positivo")
ax.set_ylabel("Taxa de Verdadeiro Positivo")
ax.set_title("Curva ROC para a Regressão Logística")
ax.legend()
plt.show()


# ## Comparativo grafico das tecnicas para cada tipo de vinho

# In[129]:


# Dados para o primeiro gráfico
tecnicas = ["Random Forest", "XGBClassifiero", "Regressão Logísitca"]
acuracias_tinto = [88.42, 81.53, 76.13]

# Dados para o segundo gráfico
acuracias_branco = [84.91, 82.65, 77.54]

# Configuração das figuras e dos eixos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Primeiro gráfico
ax1.bar(tecnicas, acuracias_tinto, color='red')
ax1.set_xlabel('Técnica aplicada para vinho Tinto')
ax1.set_ylabel('Acurácia')
ax1.set_title('Gráfico de Barras - Acurácia por Técnica')

# Segundo gráfico
ax2.bar(tecnicas, acuracias_branco, color='green')
ax2.set_xlabel('Técnica aplicada para vinho Branco')
ax2.set_ylabel('Acurácia')
ax2.set_title('Gráfico de Barras - Acurácia por Técnica')

# Ajusta o espaçamento entre as subplots
plt.tight_layout()

# Exibe os gráficos
plt.show()


# # Considerações finais
# Após a realização da análise e construção do modelo, podemos inferir que:
# 
#  - O modelo que apresentou o melhor desempenho para estimar se o vinho Tinto era 'bom' ou 'ruim' foi o Random Forest apresentou AUC de 88.42%, ainda o XGBClassifier com um AUC de 81.53% e a Regressão Logísitca apresentou AUC de 76.13%;
#  - O modelo que apresentou o melhor desempenho para estimar se o vinho Branco era 'bom' ou 'ruim' foi o o Random Forest apresentou AUC de  84.91%, ainda XGBClassifier com um AUC de 82.65% e a Regressão Logísitca apresentou AUC de 77.54%; 
#  - É interessante notar que o Random Forest e o XGBClassifier apresentaram um desempenho bem próximo;
#  - O conjunto de dados realmente apresentava-se com um desbalanceamento considerável, mesmo após a categorização da variável classe (qualidade) em "ruim" e "bom" (0 e 1).
#  - A necessidade de fazer um estudo dos vinhos separados, pois as caracteristicas fisico-quimicas dos tintos e dos brancos gerava caracteristicas unicas na qualidade.
