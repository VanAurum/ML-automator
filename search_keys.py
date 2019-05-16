

def get_keys(algorithm):
    '''
    '''
    keys={
        'xgboost': (
            'max_depth',
            'min_child_weight',
            'gamma',
            'learning_rate',
            'subsample',
            'colsample_bylevel', 
            'colsample_bytree',
            'n_estimators',
            )
    }

    return keys[algorithm]