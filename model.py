
import numpy as np
import scipy as stats

from helpers.functions import read_csv, convert_str_col_to_float, convert_str_col_to_int, encode_strings, most_frequent

class KNN:
    """
    Docstring
    
    Attributes
    ----------
    

    Methods
    -------
    
    """
    def __init__(self):
        """
        Docstring
        """
        pass
        

    # @staticmethod
    def find_distance(self, row_A, row_B):
        """
        Returns Euclidian distance between 2 rows of data.
        
            Parameters:
                    row_A (list): vector of numerical data
                    row_B (list): vector of numerical data
                    
            Returns:
                    Euclidian Distance (float)
        """
        # row_A = np.array(row_A)
        # row_B = np.array(row_B)
        
        dist = 0.0
        for i in range(len(row_A) - 1):
            dist += (row_A[i] - row_B[i])**2
        
        return np.sqrt(dist)
    
    
    # def fit(self, X_train, y_train):
    #     return
        
    def find_neighbors(self, X, y, num_neighbors=5):
        """
        docstring
        """
        euclidian_distances = []
        
        for row in X:
            eu_dist = self.find_distance(row, y)
            euclidian_distances.append((row, eu_dist))
        
        euclidian_distances.sort(key=lambda tup: tup[1])
        
        neighbors = []
        for i in range(num_neighbors):
            neighbors.append(euclidian_distances[i])
            
        return neighbors


    def predict(self, X, y, num_neighbors=5):
        neighbors = self.find_neighbors(X, y, num_neighbors)
        preds = [tup[0][-1] for tup in neighbors]
        prediction = most_frequent(preds)
        return prediction

data = read_csv('data/iris.csv')
for i in range(len(data[0]) - 1):
    convert_str_col_to_float(data, i)
# encode_strings(data, 4)

# data = read_csv('data/abalone.csv')
# for i in range(1, len(data[0]) - 1):
#     convert_str_col_to_float(data, i)
    
# encode_strings(data, 0)
# convert_str_col_to_int(data, len(data[0]) - 1)


knn = KNN()
print(knn.predict(data, data[100]))

# a = [2, 2, 2, 2, 2]
# print(most_frequent(a))