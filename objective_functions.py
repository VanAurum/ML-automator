# Standard Python Library imports
from random import shuffle
import pprint
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, RepeatedKFold, KFold, cross_val_score)
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2,f_classif
from sklearn.pipeline import FeatureUnion

#Local Imports

#3rd party imports
from xgboost import XGBClassifier

class Classifiers:

    @staticmethod
    def objective01(space, X,Y):
        
        #Define the subset of dictionary keys that should get passed to the machine learning
        #algorithm.
        
        keys=('max_depth','min_child_weight','gamma','learning_rate','subsample','colsample_bylevel', \
            'colsample_bytree','n_estimators')
        
        subspace={k:space[k] for k in set(space).intersection(keys)}
        
        #Extract the remaining keys that are pertinent to data preprocessing.
        
        model = XGBClassifier(n_jobs=-1,**subspace)     
        scaler=space.get('scaler')
        num_features=space.get('k_best')
        
        #Assemble a data pipeline with the extracted data preprocessing keys.
        pipeline=[]
        pipeline=Pipeline([('scaler', scaler),
                        ('select_best', SelectKBest(k=num_features)),
                        ('classifier',model)])
        
        #perform two passes of 10-fold cross validation and return the mean score.
        kfold = RepeatedKFold(n_splits=10, n_repeats=1)
        scores = -cross_val_score(pipeline, X, Y, cv=kfold, scoring='accuracy',verbose=False).mean()
        
        #best is a global variable that will be defined later.  By adding this as a threshold, 
        #we only train models that beat the baseline benchmark, and then the subsequent best score
        #thereafter.  This reduces the time complexity of this algorithm significantly.
        
        return scores


    @staticmethod
    def objective02(space, X,Y):
        
        #Define the subset of dictionary keys that should get passed to the machine learning
        #algorithm.
        
        keys=('loss','penalty','alpha','max_iter')    
        subspace={k:space[k] for k in set(space).intersection(keys)}
        
        #Extract the remaining keys that are pertinent to data preprocessing.
        
        model = SGDClassifier(n_jobs=-1,**subspace)   
        scaler=space.get('scaler')
        num_features=space.get('k_best')
        
        #Assemble a data pipeline with the extracted data preprocessing keys.
        pipeline=[]
        pipeline=Pipeline([('scaler', scaler),
                        ('select_best', SelectKBest(k=num_features)),
                        ('classifier',model)])
        
        #perform two passes of 10-fold cross validation and return the mean score.
        kfold = RepeatedKFold(n_splits=10, n_repeats=1)
        scores = -cross_val_score(pipeline, X, Y, cv=kfold, scoring='accuracy',verbose=False).mean()
        
        return scores


    @staticmethod
    def objective03(space, X,Y):
        
        #Define the subset of dictionary keys that should get passed to the machine learning
        #algorithm.
        
        keys=('n_estimators','max_depth','max_features','criterion','min_samples_split',\
            'min_samples_leaf','min_impurity_decrease','n_jobs')
        
        subspace={k:space[k] for k in set(space).intersection(keys)}
        
        #Extract the remaining keys that are pertinent to data preprocessing.
        
        model = RandomForestClassifier(**subspace)   
        scaler=space.get('scaler')
        num_features=space.get('k_best')
        
        #Assemble a data pipeline with the extracted data preprocessing keys.
        pipeline=[]
        pipeline=Pipeline([('scaler', scaler),
                        ('select_best', SelectKBest(k=num_features)),
                        ('classifier',model)])
        
        #perform two passes of 10-fold cross validation and return the mean score.
        kfold = RepeatedKFold(n_splits=10, n_repeats=1)
        scores = -cross_val_score(pipeline, X, Y, cv=kfold, scoring='accuracy',verbose=False).mean()
        
        return scores


    @staticmethod
    def objective04(space, X,Y):
        
        #Define the subset of dictionary keys that should get passed to the machine learning
        #algorithm.
        
        keys=('C','kernel','degree','probability')     
        subspace={k:space[k] for k in set(space).intersection(keys)}
        
        #Extract the remaining keys that are pertinent to data preprocessing.
        n_estimators=space.get('n_estimators')
        
        #Build a bagging model with the parameters from our Hyperopt search space.
        model = BaggingClassifier(SVC(**subspace),
                                max_samples=len(X)//n_estimators,
                                n_estimators=n_estimators,
                                n_jobs=-1)
        
        scaler=space.get('scaler')
        num_features=space.get('k_best')
        
        #Assemble a data pipeline with the extracted data preprocessing keys.
        pipeline=[]
        pipeline=Pipeline([('scaler', scaler),
                        ('select_best', SelectKBest(k=num_features)),
                        ('classifier',model)])
        
        #perform two passes of 10-fold cross validation and return the mean score.
        kfold = RepeatedKFold(n_splits=10, n_repeats=1)
        scores = -cross_val_score(pipeline, X, Y, cv=kfold, scoring='accuracy',verbose=False).mean()
        
        return scores


class Regressors:
    pass