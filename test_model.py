import pandas as pd

from KNN import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
data = iris.data    
target = iris.target

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)#, random_state=3)

# # Test my KNN algorithm (targets are numerical)
# clf = KNN(num_neighbors=7)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# print(predictions)
# print(f"My model's accuracy: {accuracy_score(y_test, predictions)}")
# y_pred = clf.predict([X_test[0]])
# print(y_pred)
# neighbors = clf.show_neighbors(X_test[0])
# print(neighbors)

# # Test Scikit-learn KNN algorithm to compare (targets are numerical)
# neigh = KNeighborsClassifier(n_neighbors=7)
# neigh.fit(X_train, y_train)
# predictions = neigh.predict(X_test)
# print(predictions)
# print(f"Sklearn model's accuracy: {accuracy_score(y_test, predictions)}")
# y_pred = neigh.predict([[4, 6, 1, 9]])
# print(y_pred)
# neighbors = neigh.show_neighbors([[4, 6, 1, 9]])
# print(neighbors)



# Test my algorithm after reading in with Pandas and doing train_test_split
df = pd.read_csv('data/iris.csv', names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
train, test = train_test_split(df, random_state=3)
X_train = train[['sepal length', 'sepal width', 'petal length', 'petal width']].values.tolist()
y_train = train['class'].values.tolist()
X_test = test[['sepal length', 'sepal width', 'petal length', 'petal width']].values.tolist()
y_test = test['class'].values.tolist()
knn = KNN(num_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
# print(predictions)
print(f"My model's accuracy: {accuracy_score(y_test, predictions)}")
y_pred = knn.predict([[141, 6, 1, 9]])
print(y_pred)
print(knn.show_neighbors([141, 6, 1, 9]))

# Test Sklearn's algorithm after reading in with Pandas and doing train_test_split
df = pd.read_csv('data/iris.csv', names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
train, test = train_test_split(df, random_state=3)
X_train = train[['sepal length', 'sepal width', 'petal length', 'petal width']].values.tolist()
y_train = train['class'].values.tolist()
X_test = test[['sepal length', 'sepal width', 'petal length', 'petal width']].values.tolist()
y_test = test['class'].values.tolist()
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
predictions = neigh.predict(X_test)
# print(predictions)
print(f"My model's accuracy: {accuracy_score(y_test, predictions)}")
y_pred = knn.predict([[141, 6, 1, 9]])
print(y_pred)
print(neigh.kneighbors([[141, 6, 1, 9]]))



# # See my KNN algorithm with data loaded where targets are numbers and fit on
# #. entire dataset
# clf = KNN(num_neighbors=7)
# clf.fit(data, target)
# y_pred = clf.predict([[4.3, 2.7, 1, 1.3]])
# print(y_pred)
# neighbors = clf.show_neighbors([4.3, 2.7, 1, 1.3])
# print(neighbors)



# # See my KNN algorithm with data loaded in with pandas (targets are strings)
# #. and fit on entire dataset
# df = pd.read_csv('data/iris.csv', names=['PW', 'PH', 'SW', 'SH', 'target'])
# X_train = df[['PW', 'PH', 'SW', 'SH']].values.tolist()
# y_train = df['target'].values.tolist()

# clf = KNN(num_neighbors=7)
# clf.fit(X_train, y_train)
# y_pred = clf.predict([[4, 6, 1, 9],
#                       [4, .8, .1, 3]])
# print(y_pred)
