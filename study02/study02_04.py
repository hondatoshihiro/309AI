#O'REILLY Pythonではじめる機械学習 p.35～
from sklearn.datasets import load_boston
import mglearn

boston = load_boston()
print("Data shape: {}".format(boston.data.shape))
#13の測定結果の特徴量だけでなく、
#特徴量間の積も特徴量としてみます。
x,y = mglearn.datasets.load_extended_boston()
print("x.shape: {}".format(x.shape))
