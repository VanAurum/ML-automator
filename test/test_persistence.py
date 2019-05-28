#Standard python library imports
import unittest
import os

#Local imports
from mlautomator.mlautomator import MLAutomator
from mlautomator.search_spaces import classifiers, regressors, get_space
from mlautomator.search_keys import get_keys, ALGORITHM_KEYS
from data.utilities import clf_prep
from tempfile import mkdtemp



class TestMLAutomator(unittest.TestCase):

    directory=mkdtemp()
    x, y = clf_prep('pima-indians-diabetes.csv')
    automator=MLAutomator(x,y, iterations = 2, specific_algos=['01'])
    automator.find_best_algorithm()

    def test_fit_best_pipeline(self):
        self.automator.fit_best_pipeline()
        self.assertIsNotNone(self.automator.best_pipeline)
        print(self.automator.best_pipeline)

    def test_model_dump(self):
        self.automator.save_best_pipeline(self.directory)    

    def test_model_load(self):
        self.automator.load_best_pipeline(filename=self.directory+'/pipeline.joblib')           