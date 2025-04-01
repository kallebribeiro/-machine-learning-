# VIDEO URL: https://www.youtube.com/watch?v=cX0rc20NG-I&list=PLyqOvdQmGdTR46HUxDA6Ymv4DGsIjvTQ-&index=27

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier


# download dataset "https://www.kaggle.com/datasets/benjibb/sp500-since-1950"

# Carregar o DataFrame principal a partir do arquivo CSV
df = pd.read_csv('')

# Excluir a coluna 'Date' (não é necessária para o treinamento)
df = df.drop('Date', axis=1)

# Mostrar as primeiras linhas para garantir que os dados foram carregados corretamente
print(df.head())

# Preparar os dados de treino
base = df.drop(df[-1::].index, axis=0)  # Excluir a última linha (dados futuros)
base['target'] = base['Close'][1:len(base)].reset_index(drop=True)  # Criar a coluna target

# Preparar os dados para treino e teste
treino = base.drop(base[-1::].index, axis=0)  # Dados de treino (sem a última linha)
treino.loc[treino['target'] > treino['Close'], 'target'] = 1  # Se a variação for positiva, target = 1
treino.loc[treino['target'] != 1, 'target'] = 0  # Se a variação for negativa ou neutra, target = 0

y = treino['target']  # Target (variável dependente)
x = treino.drop('target', axis=1)  # Características (variáveis independentes)

# Dividir os dados em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# Treinar o modelo ExtraTreesClassifier
modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)

# Exibir a acurácia do modelo
print("Acurácia:", modelo.score(x_teste, y_teste))

# Salvar o modelo treinado
import joblib
joblib.dump(modelo, 'modelo_treinado.pkl')

# Salvar os dados preparados para produção (última linha para previsão)
prev = base[-1::].drop('target', axis=1)
prev.to_csv('C:/Users/ribei/Downloads/GSPC.csv/Prev.scv', index=False)

# Salvar o arquivo 'Hoje.csv' para uso na produção
base.to_csv('C:/Users/ribei/Downloads/GSPC.csv/Hoje.csv', index=False)

print("Modelo treinado e dados salvos com sucesso.")


    # PRODUCAO


import pandas as pd
import joblib

# Carregar o modelo treinado
modelo = joblib.load('modelo_treinado.pkl')

# Carregar os dados de produção
prev = pd.read_csv('C:/Users/ribei/Downloads/GSPC.csv/Prev.scv')
base = pd.read_csv('C:/Users/ribei/Downloads/GSPC.csv/Hoje.csv')

# Exibir o fechamento anterior e a previsão
print('Fechamento anterior:', prev['Close'][0])

# Tentar carregar os dados de 'Futuro'
try:
    amanha = pd.read_csv('C:/Users/ribei/Downloads/GSPC.csv/Futuro.scv')
    print('Fechamento atual:', amanha['Close'][0])
    base = base.append(amanha[:1], sort=True)
    amanha = amanha.drop(amanha[:1].index, axis=0)
    base.to_csv('C:/Users/ribei/Downloads/GSPC.csv/Hoje.csv', index=False)
    amanha.to_csv('C:/Users/ribei/Downloads/GSPC.csv/Futuro.scv', index=False)
except Exception:
    print('O fechamento ainda não ocorreu.')

# Adicionar a coluna 'target' para previsão
base['target'] = base['Close'][1:len(base)].reset_index(drop=True)

# Criar o DataFrame 'prev' com a última linha da base (excluindo 'target')
prev = base[-1::].drop('target', axis=1)

# Realizar a previsão
prev['target'] = modelo.predict(prev)

# Exibir a previsão
if prev['target'].iloc[0] == 1:
    print('VAI SUBIR !!!!')
else:
    print('VAI CAIR !!!!')

# Exibir a previsão do fechamento anterior
print('Previsão anterior:', prev['target'].iloc[0])

# Salvar a previsão
prev.to_csv('C:/Users/ribei/Downloads/GSPC.csv/Prev.scv', index=False)

print("Previsão realizada e salva com sucesso.")
