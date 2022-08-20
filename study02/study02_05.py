#O'REILLY Pythonではじめる機械学習 p.38～
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.neighbors import KNeighborsClassifier

x, y = mglearn.datasets.make_forge()
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)
print("Test set predictions: {}".format(clf.predict(x_test)))
