#Standard Python libary imports
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

#3rd party imports
from hyperopt import hp



def get_space(space):
    '''

    '''
    
    scaler_list=[StandardScaler(copy=True), RobustScaler(copy=True), None]
    k_best=np.arange(2,8,1)
    n_components=np.round(np.arange(0.1,0.99,0.01),5)
    mcw=np.round(np.arange(0.0001,0.3,0.0001),5)
    
    Space_List={
                 #XGBoost Classifier search space.
                 '01':      {
                            'max_depth': hp.choice('x_max_depth',[3,4,5,6]),
                            'min_child_weight':hp.choice('x_min_child_weight',mcw),
                            'gamma':hp.choice('x_gamma',np.round(np.arange(0.0,40.0,0.005),5)),
                            'learning_rate':hp.choice('x_learning_rate',np.round(np.arange(0.0005,0.3,0.0005),5)),
                            'subsample':hp.choice('x_subsample',np.round(np.arange(0.01,1.0,0.01),5)),
                            'colsample_bylevel':hp.choice('x_colsample_bylevel',np.round(np.arange(0.1,1.0,0.01),5)),
                            'colsample_bytree':hp.choice('x_colsample_bytree',np.round(np.arange(0.1,1.0,0.01),5)),
                            'n_estimators':hp.choice('x_n_estimators',np.arange(25,850,1)),
                            'k_best':hp.choice('x_k_best',np.arange(2,8,1)),
                            'n_components': hp.choice('x_n_comps',n_components),
                            'scaler':hp.choice('x_scale',scaler_list)
                             
                            },        

                    
                #SGD Classifier search space    
                '02':       {
                            'loss': hp.choice('x_loss',['log','modified_huber']),
                            'penalty':hp.choice('x_penalty',['none', 'l2', 'l1', 'elasticnet']),
                            'alpha':hp.choice('x_alpha',np.arange(0.0,0.5,0.00001)),
                            'max_iter':hp.choice('x_max_iter',np.arange(5,1000,1)),
                            'k_best':hp.choice('x_k_best',np.arange(2,8,1)),
                            'n_components': hp.choice('x_n_comps',n_components),
                            'scaler':hp.choice('x_scale',scaler_list)
                            },
                        
                #Random Forest search space        
                '03':       {
                            'n_estimators': hp.choice('x_loss',np.arange(50,750,5)),
                            'max_depth':hp.choice('x_max_depth',np.arange(1,8,1)),
                            'max_features':hp.choice('x_max_features',['sqrt','log2',None]),
                            'criterion':hp.choice('x_crit',['gini','entropy']),
                            'min_samples_split':hp.choice('x_mss',np.arange(0.000001,0.3,0.00005)),
                            'min_samples_leaf':hp.choice('x_msl',np.arange(0.000001,0.3,0.00005)),
                            'min_impurity_decrease':hp.choice('min_id',np.arange(0.000001,0.3,0.00005)),
                            'n_jobs':hp.choice('x_njobs',[-1]),
                            'k_best':hp.choice('x_k_best',np.arange(2,8,1)),
                            'n_components': hp.choice('x_n_comps',n_components),
                            'scaler':hp.choice('x_scale',scaler_list)                    
                            },
                        
                #Support Vector Machine search space        
                '04':       {
                            'C': hp.choice('C', np.arange(0.0,1.0,0.00005)),
                            'kernel': hp.choice('x_kernel',['poly', 'rbf']),
                            'degree':hp.choice('x_degree',[2,3,4,5]),
                            'probability':hp.choice('x_probability',[True]),
                            'k_best':hp.choice('x_k_best',k_best),
                            'n_estimators':hp.choice('x_n_estimators',np.arange(5,200,1)),
                            'scaler':hp.choice('x_scale',scaler_list)                    
                            },
                        
                }

    return Space_List[space]