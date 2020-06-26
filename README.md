# K Nearest Neighbors Classifier

## Usage

```python
from KNN import KNN
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data    
target = iris.target

# Instantiate
clf = KNN(num_neighbors=7)
# Fit the model
clf.fit(data, target)
# Predict a target
clf.predict([[4.3, 2.7, 1, 1.3]])
# Show its 7 nearest neighbors
clf.show_neighbors([[4.3, 2.7, 1, 1.3]])
```

### For more information on how this algoithm was implemented, check out this [blog post](https://dougscohen.github.io/2020-06-25-K-Nearest-Neighbors-From-Scratch/)

### Check out my [portfolio website](https://dougscohen.github.io/) as well!