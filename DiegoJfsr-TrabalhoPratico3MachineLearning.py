
###Trabalho_PraticoIII_Machine_Learning.ipynb


#instalando as bibliotecas

!pip install pycaret == 2.1.2
!pip install yfinance

# Esse codigo permite que o pycaret seja usado dentro do colab

from pycaret.utils import enable_colab
enable_colab()

#inportando as bibliotecas

import yfinance as yf
import pandas as pd

# usei esse codigo para forçar a instalação do  Jinja2 
# e importação do pycaret.regression mais adiante no modelo

# mas pode ser retirado do modelo final caso nao haja problemas na execução do restante dos  comandos
pip install pycaret --user

# usei esse codigo para forçar a instalação do  Jinja2 
# e importação do pycaret.regression mais adiante no modelo

# mas pode ser retirado do modelo final caso nao haja problemas na execução do restante dos  comandos
pip install markupsafe==2.0.1

import jinja2
from pycaret.classification import *

#Escolhendo a ação que sera analizada / dataset usado

#df = yf.Ticker('ITUB4.SA') # Acção de teste Itaú Unibanco Holding S.A.
df = yf.Ticker('RADL3.SA') # Acção de teste Raia Drogasil S.A.

#escolher o intervalo de dados
#raia = df.history(period='2y') # escolhendo os 2 utimos anos
#raia

dataRad = df.history(period='2y') # escolhendo os 2 utimos anos
dataRad

# retirando campos que não utilizarei: Dividends,Stock Splits. Pois não precisarei deles

#raia = raia.drop(['Dividends','Stock Splits'], axis=1)
#raia

dataRad = dataRad.drop(['Dividends','Stock Splits'], axis=1)
dataRad

#criando novos campos, adicionando mais features para testar o comportamento da biblioteca
# os novos campos, media movel de 7 dias e de 30 dias

#raia['MM7d'] = raia['Close'].rolling(window=7).mean().round(2)
#raia['MM30d'] = raia['Close'].rolling(window=30).mean().round(2)
#raia

dataRad['MM7d'] = dataRad['Close'].rolling(window=7).mean().round(2) # media movel no campo close, com a media na janela de 7 dias, arrendondando as casas decimais para 2
dataRad['MM30d'] = dataRad['Close'].rolling(window=30).mean().round(2) # media movel no campo close, com a media na janela de 30 dias, arrendondando as casas decimais para 2
dataRad

# nova tabela com os 2 novos campos.
# no inicio e apresentado NaN pois o comando não consegue calcular o inicio dos dados.

# Separação para retiradas de dados / dias / linhas  para previsao

#raia_prever = raia.tail(5)
#raia_prever

#dataRad_prever = dataRad.tail(100)  # previsao de 100 dias
#dataRad_prever = dataRad.tail(50)  # previsao de 50 dias

dataRad_prever = dataRad.tail(5) # previsao de 5 dias
dataRad_prever

# retiradas de dados / dias / linhas  para previsao

#retirar os ultimos 100 dias do df
#dataRad.drop(dataRad.tail(100).index, inplace=True)
#dataRad

#retirar os ultimos 50 dias do df
#dataRad.drop(dataRad.tail(50).index, inplace=True)
#dataRad

#retirar os ultimos 5 dias do df
dataRad.drop(dataRad.tail(5).index, inplace=True) #usei o implace, para forçar a mudança diretamente no dataframe utilizado
dataRad

# puxando / empurrando os valores para frente
# baseado nos valores de fechamento do dia, eu faço com que o modelo consiga prever o valor de fechamento do proximo dia

#raia['Close'] = raia['Close'].shift(-1)
#raia

dataRad['Close'] = dataRad['Close'].shift(-1)
dataRad

#retirar valores nulos
#raia.dropna(inplace=True)
#raia

dataRad.dropna(inplace=True)
dataRad

#drop id
#raia.reset_index(drop=True, inplace=True)
#raia_prever.reset_index(drop=True, inplace=True)

dataRad.reset_index(drop=True, inplace=True) # df. dataRad com drop no id
dataRad_prever.reset_index(drop=True, inplace=True) # df. dataRad_prever com drop no id

dataRad

#importando a biblioteca do pycaret
#from pycaret.regression import *
#setup(data= raia, target='Close', session_id=123)

#importar a biblioteca e passar os parametros que seram utilizados
#dataset utilizado, o que eu quero prever/campo utilizado, session para manter as informações durante a execulçao

from pycaret.regression import *
setup(data= dataRad, target='Close', session_id=123)

#escolha dos melhores modelos
top3 = compare_models(n_select=3)

print(top3)

lar = create_model('lar', fold=10)

ridge = create_model('ridge', fold=10)

br = create_model('br', fold=10)

#Tunning
ridge_params = { 'alpha':[0.02, 0.024, 0.025, 0.026, 0.03]} #recebe o alpha com alguns parametros
tunne_ridge = tune_model(ridge, n_iter=1000, optimize='RMSE', custom_grid=ridge_params) # vai tunar o modelo # vai testar mil vezes, vai melhorar o RMSE, vai receber o reidge_params

tunne_lar = tune_model(lar, n_iter=1000, optimize = 'RMSE')

tunne_br = tune_model(br, n_iter=1000, optimize = 'RMSE')

#Grafico erros
plot_model(tunne_ridge, plot='error')

plot_model(tunne_ridge, plot='feature')

#Rodando o modelo com os dados de treinamento
predict_model(tunne_ridge)

#Rodando o modelo com os dados de teste
final_ridge_model = finalize_model(tunne_ridge)

#Rodando o modelo com os dados de teste e fazendo a previsao
previsao = predict_model(final_ridge_model, data=dataRad_prever)
previsao

# previsao feita apartir da base de dados separada do df principal

#Salvando o modelo para utilizar com os novos dados
save_model(final_ridge_model, 'Modelo Final Ridge')