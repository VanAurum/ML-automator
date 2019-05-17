#Standard python library imports
import unittest

#Local imports
from mlautomator.mlautomator import MLAutomator
from data.utilities import clf_prep

class TestMLAutomator(unittest.TestCase):

    def setUp(self):
        self.x, self.y = clf_prep('pima-indians-diabetes.csv')
  
    def test_automator_initialization(self):
        '''
        Test that all class properties are being initialized properly.
        '''
        automator=MLAutomator(self.x,self.y)
        self.assertEqual(automator.best,0)
        self.assertEqual(automator.count,0)
        self.assertEqual(automator.start_time,None)
        self.assertEqual(automator.objective,None)
        self.assertEqual(automator.keys,None)
        self.assertEqual(automator.master_results,[])
        self.assertEqual(automator.type,'classifier')
        self.assertEqual(automator.score_metric,'accuracy')
        self.assertEqual(automator.iterations,25)
        self.assertEqual(automator.num_cv_folds,10)
        self.assertEqual(automator.repeats,1)

    def test_clf_prep(self):
        pass
        


#if __name__=='__main__':
#    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMLAutomator)
#    unittest.TextTestRunner(verbosity=2).run(suite)

