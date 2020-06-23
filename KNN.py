import numpy as np

class KNN():
    
    def __init__(self, num_neighbors=5):
        self.num_neighbors = num_neighbors
        
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
        
        # set distance to start at 0
        dist = 0.0
        # iterate through row_A
        for i in range(len(row_A)):
            # subtract the rows element wise, square the difference, then add
            #. it to the distance
            dist += (row_A[i] - row_B[i])**2
        
        # return the sqyare root of the total distance
        return np.sqrt(dist)
    
    def fit(self, X_train, y_train):
        """
        Fit the model to the training data

        Parameters: 
                X_train (list): 2D list/array of numerical values
                y_train (list): list/array holding the target values for X_train
        """
        self.X_train = X_train
        self.y_train = y_train
        
    
    def predict(self, X):
        """
        Returns a list of predictions for the inputed X matrix.
        
        Parameters:
                X (list): 2D list/array of numerical values
                
        Returns:
                predictions (list): list of predictions for each of the vectors
                in X. 
        """
        # set predictinos to an empty list
        predictions = []
        
        # iterate (len(X)) number of times through 
        for i in range(len(X)):
            
            # list containing euclidian distances
            euclidian_distances = []
            
            # for each row in X_train, find its euclidian distance with the
            #. current 'X' row we are iterating through
            for row in self.X_train:
                eu_dist = self.find_distance(row, X[i])
                # append each euclidian distance to the list above
                euclidian_distances.append(eu_dist)
            
            # sort the euclidian distances from smallest to largest and grab
            #. the first K distances where K is the num_neigbors we want
            euclidian_distances_sorted = np.array(euclidian_distances).argsort()[:self.num_neighbors]
            
            # empty dictionary
            neighbor_count = {}
            
            for j in euclidian_distances_sorted:
                if self.y_train[j] in neighbor_count:
                    neighbor_count[self.y_train[j]] += 1
                else:
                    neighbor_count[self.y_train[j]] = 1
            
            # get the most common class label and append it to predictions
            predictions.append(max(neighbor_count, key=neighbor_count.get))
            
        return predictions


    def show_neighbors(self, X):
        """
        docstring
        """
        # set predictinos to an empty list
        neighbors = []
        
        # iterate (len(X)) number of times through 
        for i in range(len(X)):
            
            # list containing euclidian distances
            euclidian_distances = []
            
            # for each row in X_train, find its euclidian distance with the
            #. current 'X' row we are iterating through
            for row in self.X_train:
                eu_dist = self.find_distance(row, X[i])
                # append each euclidian distance to the list above
                euclidian_distances.append((row, eu_dist))
            
            # sort the euclidian distances from smallest to largest and grab
            #. the first K distances where K is the num_neigbors we want
            # euclidian_distances_sorted = np.array(euclidian_distances).argsort()[:self.num_neighbors]
            euclidian_distances.sort(key=lambda tup: tup[1])
            
            neighbors.append(euclidian_distances)
            
        return neighbors[0][:self.num_neighbors]