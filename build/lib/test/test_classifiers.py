#Standard python imports
import unittest

#Local imports
from mlautomator.mlautomator import MLAutomator
from data.utilities import clf_prep
from mlautomator.objectives import *

class TestClassifiers(unittest.TestCase):
    '''
    Integrations tests for classifier objective functions. Checks to see that at least 
    one iteration can be executed on the given dataset and algorithm.

    To increase test speed:
        - iterations set to 1
        - num_cv_folds=2
    '''

    iters=1
    folds=2
    x, y = clf_prep('pima-indians-diabetes.csv')
    print('Data prepped')

    def test_objective01_xgboost(self):
        automator=MLAutomator(self.x, self.y, iterations=self.iters, specific_algos=['01'], num_cv_folds=self.folds)
        automator.find_best_algorithm()
        self.assertEqual(automator.best_algo, 'xgboost_classifier')

    def test_objective02_sgd_classifier(self):
        automator=MLAutomator(self.x, self.y, iterations=self.iters, specific_algos=['02'], num_cv_folds=self.folds)
        automator.find_best_algorithm()
        self.assertEqual(automator.best_algo, 'SGDClassifier')

    def test_objective03_sgd_classifier(self):
        automator=MLAutomator(self.x, self.y, iterations=self.iters, specific_algos=['03'], num_cv_folds=self.folds)
        automator.find_best_algorithm()
        self.assertEqual(automator.best_algo, 'RandomForestClassifier')

    def test_objective04_bag_of_svc(self):
        automator=MLAutomator(self.x, self.y, iterations=self.iters, specific_algos=['04'], num_cv_folds=self.folds)
        automator.find_best_algorithm()
        self.assertEqual(automator.best_algo, 'SVC')                

    def test_objective05_naive_bayes(self):
        automator=MLAutomator(self.x, self.y, iterations=self.iters, specific_algos=['05'], num_cv_folds=self.folds)
        automator.find_best_algorithm()
        self.assertEqual(automator.best_algo, 'GaussianNB')          

    def test_objective06_logistic_regression(self):
        automator=MLAutomator(self.x, self.y, iterations=self.iters, specific_algos=['06'], num_cv_folds=self.folds)
        automator.find_best_algorithm()
        self.assertEqual(automator.best_algo, 'LogisticRegression')    

    def test_objective07_knn(self):
        automator=MLAutomator(self.x, self.y, iterations=self.iters, specific_algos=['07'], num_cv_folds=self.folds)
        automator.find_best_algorithm()
        self.assertEqual(automator.best_algo, 'KNeighborClassifier')                    


  