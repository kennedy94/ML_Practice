import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score

#importando dados
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#dados do url transformados em um dataframe do pandas
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
classes = ['target']

df = pd.read_csv(url, names=features + classes)

#imprimindo 5 primeiras linhas
df.head()

#preprocessando dados categoricos
#inicializando codificação
#para classes
#enc = preprocessing.OrdinalEncoder()
enc = preprocessing.LabelEncoder()

#Para características
#enc = preprocessing.OneHotEncoder()
#y = pd.DataFrame(enc.transform(df[['target']]).toarray())

#ajustando codificação para os dados categóricos
enc.fit(df['target'])
#classes são atualizadas para valor numerico para o SVM no y
y = pd.DataFrame(enc.transform(df['target']))

#guardando valores categoricos e númericos para tradução
unique_y = list(enc.classes_)
unique_y_enc = list(enc.transform(unique_y))


# Separando em exemplos e classes
X = df.loc[:, features].values

# Padronizando os dados para seguir uma normal com média 0 desvp 1, para utilizado o PCA
X = StandardScaler().fit_transform(X)
#PCA para 2 componentes
pca = PCA(n_components=2)
#ajustando modelo do pca para os dados X
X = pca.fit_transform(X)

#dando nome aos bois
X = pd.DataFrame(data = X,
                 columns = ['x1', 'x2'])

finalDf = pd.concat([X, y], axis = 1)

#criando modelos

C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))


#computar média do 10fold da validação cruzada
scores = []

for clf in models:
    scores.append(np.mean
                  (np.array
                   (cross_val_score(clf, X, y.values.ravel(), cv=10))))

print(scores)

scores = np.array(scores)

#pega melhor modelo e treina 
melhor_modelo = np.argmax(scores)
modelo_fodao = models[melhor_modelo];
modelo_fodao.fit(X,y.values.ravel())

#ajustando modelos
#models = (clf.fit(X, y) for clf in models)
#clf.fit(X,y)

#prever valores
previsto = modelo_fodao.predict(X)
#previsto = enc.inverse_transform(previsto)

#calcular acurácia para todos os exemplos
#clf.score(X,y)

import matplotlib.pyplot as plt

colors = ['r', 'g', 'b'];

plt.scatter(X['x1'], X['x2'], c = previsto)
plt.show()
































