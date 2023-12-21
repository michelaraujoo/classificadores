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
#dataset = pd.read_csv("https://raw.githubusercontent.com/michelaraujoo/classificadores/main/content/sample_data/triagem_online.csv")
dataset = pd.read_csv("https://raw.githubusercontent.com/michelaraujoo/classificadores/main/triagem_online.csv")
#dataset = pd.read_csv("https://raw.githubusercontent.com/michelaraujoo/classificadores/main/dataset_ciasc_375180_desbalanceado.csv")
dataset = dataset.dropna()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

######################################
# Separar dados em Desenvolvimento e Testes
######################################

X_desenvolvimento, X_testes, y_desenvolvimento, y_testes = ms.train_test_split(X, y, test_size=0.1, random_state=1)

# Gravando a tabela de testes em arquivo para testar o desempenho do Triagem On-Line posteriormente.
panda_va1 = pd.DataFrame(data=X_testes)
panda_va1.to_csv("/content/sample_data/arquivo1.csv", index=False)
panda_va2 = pd.DataFrame(data=y_testes)
panda_va2.to_csv("/content/sample_data/arquivo2.csv", index=False)



########  A PARTIR DE AGORA INICIA A PARTE DO TREINAMENTO COM OS DADOS  SOMENTE ANTES DE ABRIL DE 2020  #############
'''
dataset = pd.read_csv("triagem_online_ate_abril.csv")
dataset = dataset.dropna()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#faço esse pulo do gato
X_desenvolvimento = X
y_desenvolvimento = y
'''
# E depois corre tudo como antes - a diferença é que não sacrifiquei os 10% dos dados anteriores a abril
################  FIM DA PARTE DO TREINAMENTO COM OS DADOS  SOMENTE ANTES DE ABRIL DE 2020  #########################


counter = Counter(y_testes)
print(counter)

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
# Separar dados em Treino e Teste
######################################
X_train, X_test, y_train, y_test = ms.train_test_split(X_nm, y_nm, test_size=0.1, random_state=1)




##ESSA PARTE É USADA APENAS PARA AJUSTAR OS HIPERPARAMETROS USANDO A BIBLIOTECA
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# Carregar seus dados e dividir entre features (X) e rótulos (y)
X = X_train
y = y_train

# Configurando a validação cruzada com k=10
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Logistic Regression
#logistic_params = {'C': np.logspace(-4, 4, 9)}
#logistic_params = {'C': np.logspace(-5, 8, 15), 'penalty': ['l1', 'l2']}
logistic_params = {'C': np.logspace(-4, 4, 100)}


logistic_grid = GridSearchCV(LogisticRegression(max_iter=1000), logistic_params, cv=kf)
logistic_grid.fit(X, y)

print("Logistic Regression - Melhores hiperparâmetros:")
print(logistic_grid.best_params_)
print("Acurácia média durante a validação cruzada:")
print(logistic_grid.best_score_)

# Decision Tree
tree_params = {'max_depth': np.arange(1, 11)}
tree_grid = GridSearchCV(DecisionTreeClassifier(), tree_params, cv=kf)
tree_grid.fit(X, y)

print("\nDecision Tree - Melhores hiperparâmetros:")
print(tree_grid.best_params_)
print("Acurácia média durante a validação cruzada:")
print(tree_grid.best_score_)

# SVM (Support Vector Machine)
print("\n Tentando encontrar os Melhores hiperparâmetros...\n")
svm_params = {'C': np.logspace(-3, 3, 7), 'gamma': np.logspace(-3, 3, 7)}
svm_grid = GridSearchCV(SVC(), svm_params, cv=kf)
svm_grid.fit(X, y)

print("\nSVM - Melhores hiperparâmetros:")
print(svm_grid.best_params_)
print("Acurácia média durante a validação cruzada:")
print(svm_grid.best_score_)
# FIM DA PARTE DE AJUSTE DE HIPERPARAMETROS





######################################
# Treinando o modelo com a Regressão Logística
######################################
classifier = LogisticRegression(max_iter=500)
classifier.fit(X_train, y_train)


######################################
# Previsao com dados não utilizados no desenvolvimento
######################################

y_pred = classifier.predict(X_testes)
y_pred_prob = classifier.predict_proba(X_testes)

y_pred_prob = y_pred_prob[:, 1]
y_result_prob = np.concatenate((y_pred.reshape(len(y_pred), 1), y_pred_prob.reshape(len(y_pred_prob), 1)), 1)

######################################
# Matriz de confusao com os dados não utilizados no desenvolvimento
######################################

cm = confusion_matrix(y_testes, y_pred)
print(cm)
y_result = np.concatenate((y_pred.reshape(len(y_pred), 1), y_testes.reshape(len(y_testes), 1)), 1)
print("Acurácia: ",accuracy_score(y_testes, y_pred))
print("Precisão: ",precision_score(y_testes, y_pred))
print("Sensibilidade: ",recall_score(y_testes, y_pred))
print("F1-score: ",f1_score(y_testes, y_pred))




####### NESSA PARTE QUERO REPETIR OS PASSOS ACIMA, MAS UTILIZANDO A VALIDAÇÃO CRUZADA COM LOGISTIC REGRESSION  ######################
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
#clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')     # Hiperparâmetros configurados manualmente
#clf = LogisticRegression(C=0.01)       # Hiperparâmetros para somente sintomas
clf = LogisticRegression(C=17.886495290574352, max_iter=1000)         # Hiperparâmetros para sinais sintomas e fatores de risco
#clf = LogisticRegression(C=1.9179102616724888)         # Hiperparâmetros para o início da pandemia

# Obter as previsões através da validação cruzada
y_train_pred = cross_val_predict(clf, X_nm, y_nm, cv=10)

# Métricas para as previsões de validação cruzada
print('Nos dados de TREINO do Logistic Regression:')
print('---' * 20)
print('Modelo:    Regressão Logística \n')
print(f"accuracy:  {accuracy_score(y_nm, y_train_pred)}")
print(f"precision: {precision_score(y_nm, y_train_pred)}")
print(f"recall:    {recall_score(y_nm, y_train_pred)}")
print(f"f1:        {f1_score(y_nm, y_train_pred)}")




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



# Obter os pesos (coeficientes) e o termo de interceptação
pesos = clf.coef_
interceptacao = clf.intercept_
print("Pesos (Coeficientes):", pesos)
print("Interceptação:", interceptacao)


# Avaliar o Modelo
y_pred = clf.predict(X_testes)

# Apresentar a Matriz de Confusão
conf_matrix = confusion_matrix(y_testes, y_test_pred)

# Visualizar a Matriz de Confusão usando Seaborn
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Classe Negativa', 'Classe Positiva'],
            yticklabels=['Classe Negativa', 'Classe Positiva'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()





######################################
# Treinando o modelo com a Árvore de Decisisão
######################################
from sklearn import tree

classifier = tree.DecisionTreeClassifier()
classifier.fit(X_nm, y_nm)


######################################
# Previsao com dados não utilizados no desenvolvimento
######################################

y_pred = classifier.predict(X_testes)

######################################
# Matriz de confusao com os dados não utilizados no desenvolvimento
######################################
"""
cm = confusion_matrix(y_testes, y_pred)
print(cm)
y_result = np.concatenate((y_pred.reshape(len(y_pred), 1), y_testes.reshape(len(y_testes), 1)), 1)

print("Acurácia: ",accuracy_score(y_testes, y_pred))
print("Precisão: ",precision_score(y_testes, y_pred))
print("Sensibilidade: ",recall_score(y_testes, y_pred))
print("F1-score: ",f1_score(y_testes, y_pred))
"""

# Tentando mostrar graficamente em modo texto
#text_representation = tree.export_text(classifier, feature_names=dataset['feature_names'], max_depth=10, spacing=3, decimals=2, show_weights=False)
#print(text_representation)


####### NESSA PARTE QUERO REPETIR OS PASSOS ACIMA, MAS UTILIZANDO A VALIDAÇÃO CRUZADA COM ÁRVORE DE DECISÃO  ######################
######################################
# Treinando o modelo com a Árvore de Decisão usando o Cross Validation
######################################

print()
print("A partir daqui serão apresentados os dados da Árvore de Decisão usando validação cruzada:")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Instanciar o modelo de árvore de decisão
#clf = DecisionTreeClassifier()

#clf = DecisionTreeClassifier(min_samples_leaf=7, max_features=16)  #Hiperparâmetros configurados manualmente
#clf = DecisionTreeClassifier(max_depth=9)   #Hiperparâmetros para somente sintomas
clf = DecisionTreeClassifier(max_depth=10)   #Hiperparâmetros para o Sinais sintomas e fatores de risco
#clf = DecisionTreeClassifier(max_depth=10)   #Hiperparâmetros para o início da pandemia


# Obter as previsões através da validação cruzada
y_train_pred = cross_val_predict(clf, X_nm, y_nm, cv=10)

# Métricas para as previsões de validação cruzada

print('Nos dados de TREINO da Árvore de Decisão:')
print('---' * 20)
print('Modelo:    Árvore de Decisão \n')
print(f"accuracy:  {accuracy_score(y_nm, y_train_pred)}")
print(f"precision: {precision_score(y_nm, y_train_pred)}")
print(f"recall:    {recall_score(y_nm, y_train_pred)}")
print(f"f1:        {f1_score(y_nm, y_train_pred)}")




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
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)


######################################
# Previsao
######################################
y_pred = classifier.predict(X_testes)
y_result = np.concatenate((y_pred.reshape(len(y_pred), 1), y_testes.reshape(len(y_testes), 1)), 1)



######################################
# Matriz de confusao com os dados não utilizados no desenvolvimento
######################################
"""
cm = confusion_matrix(y_testes, y_pred)
print(cm)
y_result = np.concatenate((y_pred.reshape(len(y_pred), 1), y_testes.reshape(len(y_testes), 1)), 1)
print("Acurácia: ",accuracy_score(y_testes, y_pred))
print("Precisão: ",precision_score(y_testes, y_pred))
print("Sensibilidade: ",recall_score(y_testes, y_pred))
print("F1-score: ",f1_score(y_testes, y_pred))
"""

####### NESSA PARTE QUERO REPETIR OS PASSOS ACIMA, MAS UTILIZANDO O SVM COM VALIDAÇÃO CRUZADA  ######################
######################################
# Treinando o modelo com o SVM usando o Cross Validation
######################################

print()
print("A partir daqui serão apresentados os dados do SVM usando validação cruzada:")
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


# Instanciar o modelo SVM
#clf = SVC(kernel='linear', C=1, probability=True)      # Hiperparâmetros configurados manualmente
#clf = SVC(C=0.1, gamma=10)     # Hiperparâmetros somente sintomas
clf = SVC(kernel='linear', C=1, gamma=10)     # Hiperparâmetros sintomas, dados de perfil e fatores de risco
#clf = SVC(C=0.1, gamma=1.0)     # Hiperparâmetros para o início da pandemia

# Obter as previsões através da validação cruzada
y_train_pred = cross_val_predict(clf, X_nm, y_nm, cv=10)

# Métricas para as previsões de validação cruzada

print('Nos dados de TREINO do SVM:')
print('---' * 20)
print('Modelo:    SVM\n')
print(f"accuracy:  {accuracy_score(y_nm, y_train_pred)}")
print(f"precision: {precision_score(y_nm, y_train_pred)}")
print(f"recall:    {recall_score(y_nm, y_train_pred)}")
print(f"f1:        {f1_score(y_nm, y_train_pred)}")




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
