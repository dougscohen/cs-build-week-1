import numpy as np

class KNN():
    """
    A class for a K Nearest Neighbors Classifier.
    
    ...
    
    Attributes
    ----------
    num_neighbors : int
        The number of neighbors the model will find. Default is 5. 
        
    Methods
    -------
    find_distance(row_A, row_B):
        Returns Euclidean distance between 2 rows of data.
    
    fit(X_train, y_train):
        Fits the model to the training data.
        
    predict(X):
        Returns a list of predictions for the inputed X matrix.
    
    show_neighbors(x_instance):
        Returns a list of tuples, where each tuple contains a neighbor and its
        euclidean distance for the given row of data.
    
    """
    def __init__(self, num_neighbors=5):
        """
        Initialization of algorithm
        """
        self.num_neighbors = num_neighbors
        
    def find_distance(self, row_A, row_B):
        """
        Returns Euclidean distance between 2 rows of data.
        
            Parameters:
                    row_A (list): vector of numerical data
                    row_B (list): vector of numerical data
                    
            Returns:
                 Euclidean Distance (float)
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
        
        # return the square root of the total distance
        return np.sqrt(dist)
    
    def fit(self, X_train, y_train=[]):
        """
        Fits the model to the training data.

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
            
            # list containing euclidean distances
            euclidean_distances = []
            
            # for each row in X_train, find its euclidean distance with the
            #. current 'X' row we are iterating through
            for row in self.X_train:
                eu_dist = self.find_distance(row, X[i])
                # append each euclidean distance to the list above
                euclidean_distances.append(eu_dist)
            
            # sort the euclidean distances from smallest to largest and grab
            #. the first K indeces where K is the num_neigbors we want
            neighbor_indices = np.array(euclidean_distances).argsort()[:self.num_neighbors]
            
            # empty dictionary for class count
            neighbor_count = {}
            
            # for each neighbor, find its class
            for j in neighbor_indices:
                if self.y_train[j] in neighbor_count:
                    neighbor_count[self.y_train[j]] += 1
                else:
                    neighbor_count[self.y_train[j]] = 1
            
            # get the most common class label and append it to predictions
            predictions.append(max(neighbor_count, key=neighbor_count.get))
            
        return predictions


    def show_neighbors(self, x_instance):
        """
        Returns a list of tuples, where each tuple contains a neighbor and its
        euclidean distance for the given row of data.
        
        Parameters:
                x_instance (list): vector of numerical data.
                
        Returns:
                neighbors_and_distances (list): a list of tuples, each tuple
                containing a neighbor and its euclidean distance to the given
                vector.
        """
            
        # list containing euclidean distances
        euclidean_distances = []
        
        # for each row in X_train, find its euclidean distance with the
        #. current 'X' row we are iterating through
        for row in self.X_train:
            eu_dist = self.find_distance(row, x_instance)
            # append each row and the euclidean distance to the list above
            euclidean_distances.append(eu_dist)
        
        
        # sort from smallest distance to largest distance and grab the first K
        #. indeces where K is the number of neigbors
        neighbor_indices = np.array(euclidean_distances).argsort()[:self.num_neighbors]
        
        euclidean_distances.sort()
        
        # list containg tuples of neighbor indeces and its euclidian distance
        #. to x_instance
        neighbors_and_distances = []
        
        for i in range(len(neighbor_indices)):
            val1 = neighbor_indices[i]
            val2 = euclidean_distances[i]
            neighbors_and_distances.append((val1, val2))
            
        
        return neighbors_and_distances