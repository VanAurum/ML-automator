#Standard Python libary imports
import pandas as pd
import numpy as np


def get_data(filename=None):
    
    '''
    '''
    
    seed=np.random.seed(1985)
    
    #Load Numerai data from disk    
    data_directory='datasets/'
    training_data = pd.read_csv(data_directory+filename, header=0)
    
    # Transform the loaded CSV data into numpy arrays
    columns = list(training_data)
    features=columns[:-1]
    X = training_data[features]
    Y = training_data[columns[-1]]
         
    x_train=X.values
    y_train=Y.values

    return x_train, y_train