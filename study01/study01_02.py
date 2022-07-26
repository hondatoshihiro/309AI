import numpy as np
from scipy import sparse

#対角成分が1でそれ以外が0の2次元配列を作る
eye = np.eye(4)
print("Numpy array:\n{}".format(eye))

#Numpy配列をScipyのCSR形式の疎行列に変換する
#非ゼロ要素だけが格納される
sparse_matrix = sparse.csr_matrix(eye)
print("\nScipy sparse CSR matrix:\n{}".format(sparse_matrix))
