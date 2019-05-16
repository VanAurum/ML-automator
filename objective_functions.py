# Standard Python Library imports
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, RepeatedKFold, KFold, cross_val_score)
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, chi2,f_classif
import warnings



#Local Imports
from search_keys import get_keys

#3rd party imports
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

class Classifiers:

    @staticmethod
    def objective01(automator, space):
        algo='XGBoost Classifier'
        X=automator.x_train
        Y=automator.y_train
        
        #Define the subset of dictionary keys that should get passed to the machine learning
        #algorithm.
        
        keys=get_keys('xgboost')
        
        subspace={k:space[k] for k in set(space).intersection(keys)}
        
        #Extract the remaining keys that are pertinent to data preprocessing.
        
        model = XGBClassifier(n_jobs=-1,**subspace)     
        scaler=space.get('scaler')
        num_features=space.get('k_best')
        
        #Assemble a data pipeline with the extracted data preprocessing keys.
        pipeline=[]
        pipeline=Pipeline([
            ('scaler', scaler),
            ('select_best', SelectKBest(k=num_features)),
            ('classifier',model),
        ])
        
        #perform cross validation and return the mean score.
        kfold = RepeatedKFold(n_splits=automator.num_cv_folds, n_repeats=automator.repeats)
        scores = -cross_val_score(pipeline, X, Y, cv=kfold, scoring=automator.score_metric,verbose=False).mean()   
        return scores, algo


    @staticmethod
    def objective02(automator, space):
        algo='SGD Classifier'
        X=automator.x_train
        Y=automator.y_train
        #Define the subset of dictionary keys that should get passed to the machine learning
        #algorithm.
        
        keys=get_keys('SGDClassifier')   
        subspace={k:space[k] for k in set(space).intersection(keys)}
        
        #Extract the remaining keys that are pertinent to data preprocessing.
        model = SGDClassifier(n_jobs=-1,**subspace)   
        scaler=space.get('scaler')
        num_features=space.get('k_best')
        
        #Assemble a data pipeline with the extracted data preprocessing keys.
        pipeline=[]
        pipeline=Pipeline([
            ('scaler', scaler),
            ('select_best', SelectKBest(k=num_features)),
            ('classifier',model),
        ])
        
        #perform cross validation and return the mean score.
        kfold = RepeatedKFold(n_splits=automator.num_cv_folds, n_repeats=automator.repeats)
        scores = -cross_val_score(pipeline, X, Y, cv=kfold, scoring=automator.score_metric,verbose=False).mean()     
        return scores, algo


    @staticmethod
    def objective03(automator,space):
        algo='Random Forest Classifier'
        X=automator.x_train
        Y=automator.y_train
        #Define the subset of dictionary keys that should get passed to the machine learning
        #algorithm.
        
        keys=get_keys('RandomForestClassifier')  
        subspace={k:space[k] for k in set(space).intersection(keys)}
        
        #Extract the remaining keys that are pertinent to data preprocessing.
        model = RandomForestClassifier(**subspace)   
        scaler=space.get('scaler')
        num_features=space.get('k_best')
        
        #Assemble a data pipeline with the extracted data preprocessing keys.
        pipeline=[]
        pipeline=Pipeline([
            ('scaler', scaler),
            ('select_best', SelectKBest(k=num_features)),
            ('classifier',model),
        ])
        
        #perform two passes of 10-fold cross validation and return the mean score.
        kfold = RepeatedKFold(n_splits=10, n_repeats=1)
        scores = -cross_val_score(pipeline, X, Y, cv=kfold, scoring='accuracy',verbose=False).mean()
        return scores, algo


    @staticmethod
    def objective04(automator,space):
        algo='Support Vector Machine'
        X=automator.x_train
        Y=automator.y_train

        #Define the subset of dictionary keys that should get passed to the machine learning
        #algorithm.
        
        keys=get_keys('SVC')    
        subspace={k:space[k] for k in set(space).intersection(keys)}
 
        #Build a model with the parameters from our Hyperopt search space.
        model = SVC(probability=True,**subspace)
        scaler=space.get('scaler')
        num_features=space.get('k_best')
        
        #Assemble a data pipeline with the extracted data preprocessing keys.
        pipeline=[]
        pipeline=Pipeline([
            ('scaler', scaler),
            ('select_best', SelectKBest(k=num_features)),
            ('classifier',model),
        ])
        
        #perform cross validation and return the mean score.
        kfold = RepeatedKFold(n_splits=automator.num_cv_folds, n_repeats=automator.repeats)
        scores = -cross_val_score(pipeline, X, Y, cv=kfold, scoring=automator.score_metric,verbose=False).mean()   
        return scores, algo


    @staticmethod
    def objective05(automator,space):
        algo='GaussianNB'
        X=automator.x_train
        Y=automator.y_train

        #Build a model with the parameters from our Hyperopt search space.
        model = GaussianNB()
        scaler=space.get('scaler')
        num_features=space.get('k_best')
        
        #Assemble a data pipeline with the extracted data preprocessing keys.
        pipeline=[]
        pipeline=Pipeline([
            ('scaler', scaler),
            ('select_best', SelectKBest(k=num_features)),
            ('classifier',model),
        ])
        
        #perform cross validation and return the mean score.
        kfold = RepeatedKFold(n_splits=automator.num_cv_folds, n_repeats=automator.repeats)
        scores = -cross_val_score(pipeline, X, Y, cv=kfold, scoring=automator.score_metric,verbose=False).mean()   
        return scores, algo


    @staticmethod
    def objective06(automator,space):
        algo='Logistic Regression'
        X=automator.x_train
        Y=automator.y_train

        #Define the subset of dictionary keys that should get passed to the machine learning
        #algorithm.
        keys=get_keys('LogisticRegression')    
        subspace={k:space[k] for k in set(space).intersection(keys)}      

        #Build a model with the parameters from our Hyperopt search space.
        model = LogisticRegression()
        scaler=space.get('scaler')
        num_features=space.get('k_best')
        
        #Assemble a data pipeline with the extracted data preprocessing keys.
        pipeline=[]
        pipeline=Pipeline([
            ('scaler', scaler),
            ('select_best', SelectKBest(k=num_features)),
            ('classifier',model),
        ])
        
        #perform cross validation and return the mean score.
        kfold = RepeatedKFold(n_splits=automator.num_cv_folds, n_repeats=automator.repeats)
        scores = -cross_val_score(pipeline, X, Y, cv=kfold, scoring=automator.score_metric,verbose=False).mean()   
        return scores, algo        

class Regressors:
    pass