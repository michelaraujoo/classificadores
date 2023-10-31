
######################################
# Importando as libraries
######################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sklearn.model_selection as ms
import seaborn as sns
from collections import Counter
from sklearn.model_selection import cross_val_score


######################################
# Importando o dataset
######################################
#dataset = pd.read_csv("triagem_online.csv")
dataset = pd.read_csv("dataset_ciasc_375180_desbalanceado.csv")
dataset = dataset.dropna()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

######################################
# Separar dados em Desenvolvimento e Testes
######################################

X_desenvolvimento, X_testes, y_desenvolvimento, y_testes = ms.train_test_split(X, y, test_size=0.1, random_state=1)

# Gravando a tabela de testes em arquivo para testar o desempenho do Triagem On-Line posteriormente.
panda_va1 = pd.DataFrame(data=X_testes)
panda_va1.to_csv("/home/guest/PycharmProjects/regressao_logistica/arquivo1.csv", index=False)
panda_va2 = pd.DataFrame(data=y_testes)
panda_va2.to_csv("/home/guest/PycharmProjects/regressao_logistica/arquivo2.csv", index=False)


########  A PARTIR DE AGORA INICIA A PARTE DO TREINAMENTO COM OS DADOS  SOMENTE ANTES DE ABRIL DE 2020  #############
""" Descomentar as linhas a seguir para fazer o treinamento com os dados até o dia 2 de abril de 2020 """

"""
dataset = pd.read_csv("triagem_online_ate_abril.csv")
dataset = dataset.dropna()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#faço esse pulo do gato
X_desenvolvimento = X
y_desenvolvimento = y
"""
################  FIM DA PARTE DO TREINAMENTO COM OS DADOS  SOMENTE ANTES DE ABRIL DE 2020  #########################


# Undersample and plot imbalanced dataset with the neighborhood cleaning rule
from collections import Counter
from matplotlib import pyplot
from numpy import where

sns.countplot(x=dataset['gravidade'])

# Importação da biblioteca do NearMiss
from imblearn.under_sampling import NearMiss

# Criando uma instância da Classe NearMiss
nm = NearMiss()

# Fazendo uma reamostragem dos dados, utilizando a lógica do NearMiss
X_nm, y_nm = nm.fit_resample(X_desenvolvimento, y_desenvolvimento)

#y_nm.value_counts()
counter = Counter(y_nm)
print(counter)

for label, _ in counter.items():
  row_ix = where(y_nm == label)[0]
  pyplot.scatter(X_nm[row_ix, 0], X_nm[row_ix, 1], label=str(label))



######################################
# Treinando o modelo com a Regressão Logística usando o Cross Validation
######################################
print()
print("A partir daqui serão apresentados os dados do Logistic Regression usando validação cruzada:")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Instanciar o modelo de regressão logística
clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')

# Obter as previsões através da validação cruzada
y_train_pred = cross_val_predict(clf, X_nm, y_nm, cv=10)

# Métricas para as previsões de validação cruzada
accuracy_val = accuracy_score(y_nm, y_train_pred)
precision_val = precision_score(y_nm, y_train_pred, average='macro')
recall_val = recall_score(y_nm, y_train_pred, average='macro')
f1_val = f1_score(y_nm, y_train_pred, average='macro')

print(f'Métricas da validação cruzada:')
print(f'Acurácia: {accuracy_val}')
print(f'Precisão: {precision_val}')
print(f'Sensibilidade (Recall): {recall_val}')
print(f'F1-Score: {f1_val}')

# Treinar o modelo no conjunto de treinamento
clf.fit(X_nm, y_nm)

# Prever no conjunto de teste
y_test_pred = clf.predict(X_testes)

# Métricas para o conjunto de teste
accuracy_test = accuracy_score(y_testes, y_test_pred)
precision_test = precision_score(y_testes, y_test_pred, average='macro')
recall_test = recall_score(y_testes, y_test_pred, average='macro')
f1_test = f1_score(y_testes, y_test_pred, average='macro')

print('\nMétricas no conjunto de teste:')
print(f'Acurácia: {accuracy_test}')
print(f'Precisão: {precision_test}')
print(f'Sensibilidade (Recall): {recall_test}')
print(f'F1-Score: {f1_test}')







######################################
# Treinando o modelo com a Árvore de Decisisão
######################################

print()
print("A partir daqui serão apresentados os dados da Árvore de Decisão usando validação cruzada:")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

clf = DecisionTreeClassifier(min_samples_leaf=7, max_features=16)
# Obter as previsões através da validação cruzada
y_train_pred = cross_val_predict(clf, X_nm, y_nm, cv=10)

# Métricas para as previsões de validação cruzada
accuracy_val = accuracy_score(y_nm, y_train_pred)
precision_val = precision_score(y_nm, y_train_pred, average='macro')
recall_val = recall_score(y_nm, y_train_pred, average='macro')
f1_val = f1_score(y_nm, y_train_pred, average='macro')

print(f'Métricas da validação cruzada:')
print(f'Acurácia: {accuracy_val}')
print(f'Precisão: {precision_val}')
print(f'Sensibilidade (Recall): {recall_val}')
print(f'F1-Score: {f1_val}')

# Treinar o modelo no conjunto de treinamento
clf.fit(X_nm, y_nm)

# Prever no conjunto de teste
y_test_pred = clf.predict(X_testes)

# Métricas para o conjunto de teste
accuracy_test = accuracy_score(y_testes, y_test_pred)
precision_test = precision_score(y_testes, y_test_pred, average='macro')
recall_test = recall_score(y_testes, y_test_pred, average='macro')
f1_test = f1_score(y_testes, y_test_pred, average='macro')

print('\nMétricas no conjunto de teste:')
print(f'Acurácia: {accuracy_test}')
print(f'Precisão: {precision_test}')
print(f'Sensibilidade (Recall): {recall_test}')
print(f'F1-Score: {f1_test}')




######################################
# Treinando o modelo com SVM
######################################

print()
print("A partir daqui serão apresentados os dados do SVM usando validação cruzada:")
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


# Instanciar o modelo SVM
clf = SVC(kernel='linear', C=1, probability=True)

# Obter as previsões através da validação cruzada
y_train_pred = cross_val_predict(clf, X_nm, y_nm, cv=10)

# Métricas para as previsões de validação cruzada
accuracy_val = accuracy_score(y_nm, y_train_pred)
precision_val = precision_score(y_nm, y_train_pred, average='macro')
recall_val = recall_score(y_nm, y_train_pred, average='macro')
f1_val = f1_score(y_nm, y_train_pred, average='macro')

print(f'Métricas da validação cruzada:')
print(f'Acurácia: {accuracy_val}')
print(f'Precisão: {precision_val}')
print(f'Sensibilidade (Recall): {recall_val}')
print(f'F1-Score: {f1_val}')

# Treinar o modelo no conjunto de treinamento
clf.fit(X_nm, y_nm)

# Prever no conjunto de teste
y_test_pred = clf.predict(X_testes)

# Métricas para o conjunto de teste
accuracy_test = accuracy_score(y_testes, y_test_pred)
precision_test = precision_score(y_testes, y_test_pred, average='macro')
recall_test = recall_score(y_testes, y_test_pred, average='macro')
f1_test = f1_score(y_testes, y_test_pred, average='macro')

print('\nMétricas no conjunto de teste:')
print(f'Acurácia: {accuracy_test}')
print(f'Precisão: {precision_test}')
print(f'Sensibilidade (Recall): {recall_test}')
print(f'F1-Score: {f1_test}')