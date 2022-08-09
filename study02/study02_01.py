#O'REILLY Pythonではじめる機械学習 p.32
import mglearn
import matplotlib.pyplot as plt

#データセットの生成(２つの特徴量を持つデータセットを生成する)
x,y = mglearn.datasets.make_forge()
#データセットをプロット
mglearn.discrete_scatter(x[:,0], x[:,1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
#26のデータで、2つの特徴量を持つことを確認
print("x.shape: {}".format(x.shape))
plt.show()