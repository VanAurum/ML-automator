#Standard Python libary imports
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer

#3rd party imports
from hyperopt import hp

SCALER_LIST = [
        StandardScaler(copy=True), 
        RobustScaler(copy=True), 
        MinMaxScaler(copy=True), 
        PowerTransformer(), 
        None
        ]

N_COMPONENTS = np.round(np.arange(0.1, 0.99, 0.01), 5)        


def classifiers(**kwargs):

    # If there variable parameters to set they get passed through kwargs.
    k_best=kwargs.get('k_best', [3])

    CLASSIFIER_SPACES = {
                #XGBoost Classifier search space.
                '01':      {
                            'max_depth': hp.choice('x_max_depth', [3, 4, 5, 6]),
                            'min_child_weight': hp.choice('x_min_child_weight', np.round(np.arange(0.0001, 0.3, 0.0001), 5)),
                            'gamma': hp.choice('x_gamma', np.round(np.arange(0.0,40.0,0.005),5)),
                            'learning_rate': hp.choice('x_learning_rate',np.round(np.arange(0.0005,0.3,0.0005),5)),
                            'subsample': hp.choice('x_subsample',np.round(np.arange(0.01,1.0,0.01),5)),
                            'colsample_bylevel': hp.choice('x_colsample_bylevel', np.round(np.arange(0.1,1.0,0.01),5)),
                            'colsample_bytree': hp.choice('x_colsample_bytree', np.round(np.arange(0.1,1.0,0.01),5)),
                            'n_estimators': hp.choice('x_n_estimators', np.arange(25,850,1)),
                            'k_best': hp.choice('x_k_best', k_best),
                            'scaler': hp.choice('x_scale', SCALER_LIST),
                            },        
            
                #SGD Classifier search space    
                '02':       {
                            'loss': hp.choice('x_loss', ['log','modified_huber']),
                            'penalty': hp.choice('x_penalty', ['none', 'l2', 'l1', 'elasticnet']),
                            'alpha': hp.choice('x_alpha', np.arange(0.0,0.5,0.00001)),
                            'max_iter': hp.choice('x_max_iter', np.arange(5,1000,1)),
                            'k_best': hp.choice('x_k_best', k_best),
                            'n_components': hp.choice('x_n_comps', N_COMPONENTS),
                            'scaler': hp.choice('x_scale', SCALER_LIST),
                            },
                        
                #Random Forest search space        
                '03':       {
                            'n_estimators': hp.choice('x_loss', np.arange(50,750,5)),
                            'max_depth': hp.choice('x_max_depth', np.arange(1,8,1)),
                            'max_features': hp.choice('x_max_features', ['sqrt','log2',None]),
                            'criterion': hp.choice('x_crit', ['gini','entropy']),
                            'min_samples_split': hp.choice('x_mss', np.arange(0.000001,0.3,0.00005)),
                            'min_samples_leaf': hp.choice('x_msl', np.arange(0.000001,0.3,0.00005)),
                            'min_impurity_decrease': hp.choice('min_id', np.arange(0.000001,0.3,0.00005)),
                            'n_jobs': hp.choice('x_njobs',[-1]),
                            'k_best': hp.choice('x_k_best',k_best),
                            'n_components': hp.choice('x_n_comps', N_COMPONENTS),
                            'scaler': hp.choice('x_scale', SCALER_LIST)                    
                            },
                        
                #Support Vector Machine search space        
                '04':       {
                            'C': hp.choice('C', np.arange(0.0,1.0,0.00005)),
                            'gamma': hp.choice('x_gamma', ['auto']),
                            'kernel': hp.choice('x_kernel', ['poly', 'rbf']),
                            'degree': hp.choice('x_degree', [2,3,4,5]),
                            'probability': hp.choice('x_probability', [True]),
                            'n_estimators': hp.choice('x_n_estimators', np.arange(4, 25, 1)),
                            'k_best': hp.choice('x_k_best', k_best),
                            'scaler': hp.choice('x_scale', SCALER_LIST)                    
                            },

                #GaussianNB   
                '05':       {
                            'k_best': hp.choice('x_k_best', k_best),
                            'scaler': hp.choice('x_scale', SCALER_LIST)                    
                            },      

                #LogisticRegression
                '06':       {
                            'penalty': hp.choice('x_penalty', ['l1','l2']),
                            'C': hp.choice('C', np.round(np.arange(0.0, 1.0, 0.00001), 5)),
                            'solver': hp.choice('x_solver', [ 'lbfgs', 'liblinear', 'sag', 'saga']),
                            'k_best': hp.choice('x_k_best', k_best),
                            'scaler': hp.choice('x_scale', SCALER_LIST)                                
                            },

                #KNearestNeighbor
                '07':       {
                            'n_neighbors': hp.choice('x_n_neighbors', np.arange(1,10,1)),
                            'weights': hp.choice('x_weights', ['uniform','distance']),
                            'algorithm': hp.choice('x_algorithm', ['ball_tree','kd_tree','brute','auto']),  
                            'k_best': hp.choice('x_k_best', k_best),
                            'scaler': hp.choice('x_scale', SCALER_LIST),
                            },   
                        
                }


    return CLASSIFIER_SPACES        


def regressors(**kwargs):
    '''
    A method for return a dictionary of parameters from a selected search space. 
    kwargs is passed because some search spaces have a variable parameter space that is data and
    algorithm dependent.  
    '''

    # If there variable parameters to set they get passed through kwargs.
    k_best=kwargs.get('k_best', [3])

    REGRESSION_SPACES={
                #XGBoost Regressor search space.
                '01':      {
                            'max_depth': hp.choice('x_max_depth', [3,4,5,6]),
                            'min_child_weight': hp.choice('x_min_child_weight', np.round(np.arange(0.0001, 0.3, 0.0001), 5)),
                            'gamma': hp.choice('x_gamma', np.round(np.arange(0.0,40.0,0.005), 5)),
                            'learning_rate': hp.choice('x_learning_rate', np.round(np.arange(0.0005,0.3,0.0005), 5)),
                            'subsample': hp.choice('x_subsample', np.round(np.arange(0.01,1.0,0.01),5)),
                            'colsample_bylevel': hp.choice('x_colsample_bylevel', np.round(np.arange(0.1,1.0,0.01), 5)),
                            'colsample_bytree': hp.choice('x_colsample_bytree', np.round(np.arange(0.1,1.0,0.01), 5)),
                            'n_estimators': hp.choice('x_n_estimators', np.arange(25,850,1)),
                            'k_best': hp.choice('x_k_best', k_best),
                            'n_components': hp.choice('x_n_comps', N_COMPONENTS),
                            'scaler': hp.choice('x_scale', SCALER_LIST)
                            },        
            
                #SGD Regressor search space    
                '02':       {
                            'loss': hp.choice('x_loss', ['squared_loss','huber']),
                            'penalty': hp.choice('x_penalty', ['none', 'l2', 'l1', 'elasticnet']),
                            'alpha': hp.choice('x_alpha', np.arange(0.0,0.5,0.00001)),
                            'max_iter': hp.choice('x_max_iter', np.arange(5,1000,1)),
                            'k_best': hp.choice('x_k_best', k_best),
                            'n_components': hp.choice('x_n_comps', N_COMPONENTS),
                            'scaler': hp.choice('x_scale', SCALER_LIST)
                            },
                        
                #Random Forest Regressor search space        
                '03':       {
                            'n_estimators': hp.choice('x_loss', np.arange(50, 750, 5)),
                            'max_depth': hp.choice('x_max_depth', np.arange(1, 8, 1)),
                            'max_features': hp.choice('x_max_features', ['sqrt','log2',None]),
                            'criterion': hp.choice('x_crit', ['mse']),
                            'min_samples_split': hp.choice('x_mss', np.arange(0.000001, 0.3, 0.00005)),
                            'min_samples_leaf': hp.choice('x_msl', np.arange(0.000001, 0.3, 0.00005)),
                            'min_impurity_decrease': hp.choice('min_id', np.arange(0.000001, 0.3, 0.00005)),
                            'n_jobs': hp.choice('x_njobs', [-1]),
                            'k_best': hp.choice('x_k_best', k_best),
                            'n_components': hp.choice('x_n_comps', N_COMPONENTS),
                            'scaler': hp.choice('x_scale', SCALER_LIST)                    
                            },
                        
                #Support Vector Machine Regressor search space        
                '04':       {
                            'C': hp.choice('C', np.arange(0.0, 1.0, 0.00005)),
                            'gamma': hp.choice('x_gamma', ['auto']),
                            'kernel': hp.choice('x_kernel', ['poly', 'rbf']),
                            'degree': hp.choice('x_degree', [2, 3, 4]),
                            'n_estimators': hp.choice('x_n_estimators', np.arange(4, 25, 1)),
                            'k_best': hp.choice('x_k_best', k_best),
                            'scaler': hp.choice('x_scale', SCALER_LIST)                    
                            },

                #KNearestNeighbor Regressor search space
                '05':       {
                            'n_neighbors': hp.choice('x_n_neighbors', np.arange(1,10,1)),
                            'weights': hp.choice('x_weights', ['uniform','distance']),
                            'algorithm': hp.choice('x_algorithm', ['ball_tree','kd_tree','brute','auto']),  
                            'k_best': hp.choice('x_k_best', k_best),
                            'scaler': hp.choice('x_scale', SCALER_LIST),
                            },                      
                }


    return REGRESSION_SPACES            


def get_space(automator, space):
    '''
    Get space checks the algo_type property in the automator object and returns the search space that corresponds 
    with the provided key. 

    Args:
        automator (object): an MLAutomator object.
        space (str): The key value for the search space to return.   

    Returns:    
        <search space> (dict) :A selection of hyperparameters chosen for analysis in the optimization engine.   
    '''
    
    # include any parameters that are data-dependent here
    algo_dependent_params = {
        'k_best' : np.arange(2, automator.num_features-1, 1),
        }
 
    if automator.type == 'classifier':
        Space_List = classifiers(**algo_dependent_params)

    elif automator.type == 'regressor':     
        Space_List = regressors(**algo_dependent_params)


    return Space_List[space]
