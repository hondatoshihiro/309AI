#O'REILLY Pythonではじめる機械学習 p.20～p.24
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

#k-最近傍法(クラス分類の一種)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)
print("knn:\n{}".format(knn))

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))
#予測を行う
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

#予測結果の評価 
#テストデータの評価を行う
Y_pred = knn.predict(X_test)
print("Test set predictions:\n{}".format(Y_pred))
#テストデータの実際の判定結果はY_testに存在するので、Y_testとY_predを比較し、
#予測の精度を判定する。
print("Test set score: {:.2f}".format(np.mean(Y_pred == Y_test)))
#以下のようにテストデータから予測の精度を判定できる
print("Test set score: {:.2f}".format(knn.score(X_test, Y_test)))
