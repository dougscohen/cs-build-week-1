
    import numpy as np
    
    def find_distance(row_A, row_B):
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