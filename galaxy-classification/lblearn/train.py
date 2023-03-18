import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from knn import KNN
from linear import LinearRegression

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

def mse(y_test, y_pred):
    return np.mean((y_test - y_pred)**2)
# ====================== KNN =========================
# cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
# iris = datasets.load_iris()
# X, y = iris.data, iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# clf = KNN(k=5)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# plt.figure()
# plt.scatter(X[:,2], X[:, 3], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()
# ==================== LinearRegression ==================
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], y, color='b', marker='o', s=30)
# plt.show()


clf = LinearRegression(lr=1e-2)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
error = mse(y_test, predictions)
y_pred_line = clf.predict(X)

cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5))
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()
print(error)
