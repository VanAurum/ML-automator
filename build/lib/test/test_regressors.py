#Standard python imports
import unittest
import warnings

#Local imports
from mlautomator.mlautomator import MLAutomator
from data.utilities import from_sklearn
from mlautomator.objectives import *

warnings.filterwarnings("ignore")

class TestRegressors(unittest.TestCase):
    '''
    Integrations tests for regressor objective functions. Checks to see that at least 
    one iteration can be executed on the given dataset and algorithm.

    To increase test speed:
        - iterations set to 1
        - num_cv_folds=2
    '''

    iters=1
    folds=2
    x, y = from_sklearn('boston')
    print('Data prepped')


    def test_objective01_xgboost_regressor(self):
        automator=MLAutomator(
            self.x, 
            self.y, 
            iterations=self.iters, 
            algo_type='regressor', 
            specific_algos=['01'], 
            num_cv_folds=self.folds, 
            score_metric = 'neg_mean_squared_error',
            )
        automator.find_best_algorithm()
        self.assertEqual(automator.best_algo, 'xgboost_regressor')

    def test_objective02_sgd_regressor(self):
        automator=MLAutomator(
            self.x, 
            self.y, 
            iterations=self.iters, 
            algo_type='regressor', 
            specific_algos=['02'], 
            num_cv_folds=self.folds, 
            score_metric = 'neg_mean_squared_error',
            )        
        automator.find_best_algorithm()
        self.assertEqual(automator.best_algo, 'SGDRegressor')

    def test_objective03_sgd_regressor(self):
        automator=MLAutomator(
            self.x, 
            self.y, 
            iterations=self.iters, 
            algo_type='regressor', 
            specific_algos=['03'], 
            num_cv_folds=self.folds, 
            score_metric = 'neg_mean_squared_error',
            )        
        automator.find_best_algorithm()
        self.assertEqual(automator.best_algo, 'RandomForestRegressor')

    def test_objective04_svr_regressor(self):
        automator=MLAutomator(
            self.x, 
            self.y, 
            iterations=self.iters, 
            algo_type='regressor', 
            specific_algos=['04'], 
            num_cv_folds=self.folds, 
            score_metric = 'neg_mean_squared_error',
            )        
        automator.find_best_algorithm()
        self.assertEqual(automator.best_algo, 'SVR')                

    def test_objective05_knn_regressor(self):
        automator=MLAutomator(
            self.x, 
            self.y, 
            iterations=self.iters, 
            algo_type='regressor', 
            specific_algos=['05'], 
            num_cv_folds=self.folds, 
            score_metric = 'neg_mean_squared_error',
            )        
        automator.find_best_algorithm()
        self.assertEqual(automator.best_algo, 'KNeighborRegressor')          



        