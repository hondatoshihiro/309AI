#O'REILLY Pythonではじめる機械学習 p.18～p.20
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("Y_train shape: {}".format(Y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("Y_test shape: {}".format(Y_test.shape))

fig, ax = plt.subplots()
#X_trainのデータからDataFrameを作る
#iris_dataset.feature_namesの文字列を使ってカラムに名前をつける
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
#DataFrameからscatter matrixを作成し、Y_trainに従って色を付ける
grr = pd.plotting.scatter_matrix(iris_dataframe, c=Y_train, figsize=(15,15), ax=ax, marker='o', hist_kwds={'bins':20}, s=60, alpha=0.8, cmap=mglearn.cm3)

plt.show()
