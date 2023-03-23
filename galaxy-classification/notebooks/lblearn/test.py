import unittest
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN


class TestStringMethods(unittest.TestCase):

    def test_knn(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        clf = KNN(k=5)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        true_results = [1, 2, 2, 0, 1, 0, 0, 0, 1, 2, 1, 0, 2, 1, 0, 1, 2, 0, 2, 1, 1, 1, 1, 1, 2, 0, 2, 1, 2, 0]
        self.assertEqual(true_results, predictions)



if __name__ == '__main__':
    unittest.main()