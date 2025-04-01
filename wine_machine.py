import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import warnings

# Ignorar o FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# download dataset "https://www.kaggle.com/datasets/dell4010/wine-dataset"

file = pd.read_csv('')
#pd.set_option('display.max_rows', None)  # Mostra todas as linhas
#print(file.head(100))
print('Hello pandas:')
#print(file.shape)

file['style'] = file['style'].replace('red',0) # TRANSFORMA OS RED EM 0 E WHITE EM 1 NA COLUNA STYLE
file['style'] = file['style'].replace('white',1)
#print(file)

#SEPARANDO AS VARIAVEIS ENTRE PREDITORAS E VARIAVEL ALVO
y = file['style'] # ARMAZENA A COLUNA STYLE NA VARIAVEL Y = ALVO
x = file.drop('style', axis =1) # EXCLUI A COLUNA STYLE E ARMAZENA O DATAFRAME NA VARIAVEL X = PREDITORAS
#print(y)

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.30) # CHAMADA PARA TREINO (X=ALVO E Y=PREDITORAS, TAMANHO DO DADOS PARA TESTE = 0.30 = 30%) 30% DOS DADOS SERAO ESCOLHIDOS ALEATORIAMENTE PARA OS TESTES
# AS VARIAVEIS ACIMA REFEREN-SE AOS DADOS DE TREINO E TESTE, EX: X_TREINO=DADOS DE X PARA TREINO
#print(x_treino.shape)

# CRIAÇAO DO MODELO
model = ExtraTreesClassifier(n_estimators=100)
model.fit(x_treino,y_treino)

# IMPRIMIR O RESULTADO
result = model.score(x_teste, y_teste)
print('Acuracia:',result)
print(x_teste[400:410])

predict = model.predict(x_teste[400:410]) # PREDICT PARA TESTAR O MACHINE LARNING 
print(y_teste[400:410]) # GABARITO
print(predict) # RESPOSTA