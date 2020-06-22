
import numpy as np
import scipy as stats

from helpers.functions import read_csv, convert_str_col_to_float, convert_str_col_to_int, encode_strings

class KNN:
    """
    Docstring
    
    Attributes
    ----------
    

    Methods
    -------
    
    """
    def __init__(self, num_neighbors=5):
        """
        Docstring
        """
        self.num_neighbors = num_neighbors
        

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
        row_A = np.array(row_A)
        row_B = np.array(row_B)
        
        dist = 0.0
        for i in range(len(row_A) - 1):
            dist += (row_A[i] - row_B[i])**2
        
        return np.sqrt(dist)
    
    
    # def fit(self, X_train, y_train):
    #     return
        
    def find_neighbors(self, X, y):
        """
        docstring
        """
        euclidian_distances = []
        
        for row in X:
            eu_dist = self.find_distance(row, y)
            euclidian_distances.append((row[:-1], eu_dist))
        
        euclidian_distances.sort(key=lambda tup: tup[1])
        
        neighbors = []
        for i in range(self.num_neighbors):
            neighbors.append(euclidian_distances[i])
            
        return neighbors
    
    # def predict(self, X, y)
        

# data = read_csv('data/iris.csv')
# for i in range(len(data[0]) - 1):
#     convert_str_col_to_float(data, i)
# encode_strings(data, 4)

data = read_csv('data/abalone.csv')
for i in range(1, len(data[0]) - 1):
    convert_str_col_to_float(data, i)
    
encode_strings(data, 0)
convert_str_col_to_int(data, len(data[0]) - 1)


knn = KNN(5)
print(knn.find_neighbors(data, data[74]))