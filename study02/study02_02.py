#O'REILLY Pythonではじめる機械学習 p.33
import mglearn
import matplotlib.pyplot as plt

x,y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(x, y, 'o')
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()