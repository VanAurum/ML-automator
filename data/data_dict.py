#Standard Python Libary imports
from sklearn.datasets import (
    load_boston,
    load_iris,
    load_diabetes,
    load_digits,
    load_linnerud,
    load_wine,
    load_breast_cancer,
)

def get_data(key):
    '''
    Accepts a text string as a key and calls the appropriate data loading function from 
    sklearn.  

    Args:
        key (str) : The description of the dataset to load. 

    Returns:
        library[key] : functional call to the appropriate dataset from sklearn.    
    '''
    library = {
        'boston': load_boston(),
        'iris' : load_iris(),
        'diabetes' : load_diabetes(), 
        'digits' : load_digits(), 
        'linnerud' : load_linnerud(),
        'wine' : load_wine(), 
        'breast_cancer' : load_breast_cancer(),
    }

    return library[key]