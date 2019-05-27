'''A module for retrieving a list of all optimization parameters unique to each algorithm.
'''

ALGORITHM_KEYS = {
    'xgboost_classifier': (
        'max_depth',
        'min_child_weight',
        'gamma',
        'learning_rate',
        'subsample',
        'colsample_bylevel', 
        'colsample_bytree',
        'n_estimators',
        ),
    'xgboost_regressor': (
        'max_depth',
        'min_child_weight',
        'gamma',
        'learning_rate',
        'subsample',
        'colsample_bylevel', 
        'colsample_bytree',
        'n_estimators',
        ),            
    'SGDClassifier': (
        'loss',
        'penalty',
        'alpha',
        'max_iter',
        ),
    'SGDRegressor': (
        'loss',
        'penalty',
        'alpha',
        'max_iter',
        ),            
    'RandomForestClassifier': (
        'n_estimators',
        'max_depth',
        'max_features',
        'criterion',
        'min_samples_split',
        'min_samples_leaf',
        'min_impurity_decrease',
        'n_jobs',
        ),
    'RandomForestRegressor': (
        'n_estimators',
        'max_depth',
        'max_features',
        'criterion',
        'min_samples_split',
        'min_samples_leaf',
        'min_impurity_decrease',
        'n_jobs',
        ),            
    'SVC': (
        'C',
        'gamma',
        'kernel',
        'degree',
        ),
    'SVR': (
        'C',
        'gamma',
        'kernel',
        'degree',
        ),            
    'LogisticRegression' : (
        'C',
        'solver',
        'pentalty',   
        ),
    'KNeighborClassifier': (
        'n_neighbors',
        'weights',
        'algorithm',
        ),
    'KNeighborRegressor': (
        'n_neighbors',
        'weights',
        'algorithm',
        ),    
    'GaussianNB': (),                            
}


def get_keys(algorithm):
    '''
    Returns the dictionary of comprehensive search key parameters that Hyperopt will attempt
    to optimize on.  

    Args:
        algorithm (string): The key for the appropriate algorithm search parameters. 
            (i.e, 'xgboost_classifier')

    Returns:
        ALGORITHM_KEYS[algorithm] (dict): dictionary of search parameter keys.

    '''

    
    return ALGORITHM_KEYS[algorithm]