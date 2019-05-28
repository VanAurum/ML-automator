#Standard python library imports
import unittest

#Local imports
from mlautomator.mlautomator import MLAutomator
from mlautomator.search_spaces import classifiers, regressors, get_space
from mlautomator.search_keys import get_keys, ALGORITHM_KEYS
from data.utilities import clf_prep


class TestMLAutomator(unittest.TestCase):

    x, y = clf_prep('pima-indians-diabetes.csv')
    automator=MLAutomator(x,y)

    def test_automator_initialization(self):
        '''
        Test that all class properties are being initialized properly.
        '''
        
        self.assertEqual(self.automator.best,0)
        self.assertEqual(self.automator.count,0)
        self.assertEqual(self.automator.start_time,None)
        self.assertEqual(self.automator._objective,None)
        self.assertEqual(self.automator.keys,None)
        self.assertEqual(self.automator.master_results,[])
        self.assertEqual(self.automator.type,'classifier')
        self.assertEqual(self.automator.score_metric,'accuracy')
        self.assertEqual(self.automator.iterations,25)
        self.assertEqual(self.automator.num_cv_folds,10)
        self.assertEqual(self.automator.repeats,1)

    def test_get_obj_key_list(self):
        self.assertIsNotNone(classifiers().keys())
        self.assertIsNotNone(regressors().keys())

    def test_get_keys(self):
        for key in ALGORITHM_KEYS.keys():
            self.assertIsNotNone(get_keys(key))
            print(get_keys(key))    

    def test_get_space_regressors(self):
        for key in regressors().keys():
            self.assertIsNotNone(get_space(self.automator, key))        

    def test_get_space_classifiers(self):
        for key in classifiers().keys():
            self.assertIsNotNone(get_space(self.automator, key))

    def test_user_feedback_when_best_space_not_evaluated(self):
        self.assertIsNone(self.automator.print_best_space()) 
        self.assertIsNone(self.automator.save_best_pipeline(directory = None)) 
        self.assertIsNone(self.automator.fit_best_pipeline())                                      