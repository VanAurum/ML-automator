

def get_keys(algorithm):
    '''
    '''
    keys={
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

    }

    return keys[algorithm]