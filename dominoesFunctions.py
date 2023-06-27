import numpy as np
import itertools

def numberDominoes(highestDominoe):
        return int((highestDominoe+1)*(highestDominoe+2)/2)
    
def listDominoes(highestDominoe):
    # given a standard rule for how to organize the list of dominoes as one-hot arrays, list the dominoes present in a one hot array
    return np.array([np.array(quake) for quake in itertools.combinations_with_replacement(np.arange(highestDominoe+1), 2)])
