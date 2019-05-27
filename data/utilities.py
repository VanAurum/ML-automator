'''

'''

#Standard Python libary imports
import pandas as pd

#Local imports
from data.data_dict import get_data


def clf_prep(filename=None, target_columns=1, ignore_columns=None):
    '''A utility method for preparing training and target data from a csv file.

    Args:
        filename (string):  the file name of the dataset to be imported.  This function assumes the
            dataset resides in 'data/datasets/'
        target_columns (int, optional, default=1): The number of target columns there (in case it 
            might be one-hot encoded, for example). Method assumes that target columns reside at 
            the end of the dataframe. The default value takes the last column as the target.
        ignore_columns (list, optional, default=None): A list of column names or indices that 
            you want dropped or ignored during preparation. These columns will not be included in the output.

    Returns:
        x_train (numpy ndarray): Training data for the model.
        y_train (numpy ndarray): Target data for the model.   
    '''

    #Load data from csv file.  
    data_directory = 'data/datasets/'
    training_data = pd.read_csv(data_directory+filename, header=0)

    if ignore_columns:
        training_data = training_data.drop(ignore_columns, axis=1)
    
    # Transform the loaded CSV data into numpy arrays
    columns = list(training_data)
    features = columns[:-1]
    feature_data = training_data[features]
    target_data = training_data[columns[-target_columns:]]
         
    x_train = feature_data.values
    y_train = target_data.values.ravel()


    return x_train, y_train


def from_sklearn(key):
    '''Imports and processes data from sklearn's library of sample datasets into 
    a format digestible by MLAutomator.

    Args:
        key (string): The key descripion of the dataset to pull from sklearn. See data_dict module for info
            on available datasets and their keys.

    Returns:
        x_train (numpy ndarray): Training data for the model.
        y_train (numpy ndarray): Target data for the model.   
    '''
    data = get_data(key)
    dataframe = pd.DataFrame(data.data)
    dataframe.columns = data.feature_names
    feature_data = dataframe[data.feature_names]
    x_train = feature_data.values
    y_train = data.target

    
    return x_train, y_train
