#O'REILLY Pythonではじめる機械学習 p.9
import numpy as np
import matplotlib.pyplot as plt

#-10から10までを100ステップに区切った列を配列として生成
x = np.linspace(-10,10,100)
#sin関数を用いて2つ目の配列yを作成
y = np.sin(x)
#plot関数は、一方の配列に対して他方の配列をプロットする
plt.plot(x,y, marker="x")
plt.show()
