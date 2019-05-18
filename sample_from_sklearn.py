from data.utilities import from_sklearn
from mlautomator.mlautomator import MLAutomator

if __name__=='__main__':
    x,y=from_sklearn('boston')
    automator=MLAutomator(x,y,iterations=30,algo_type='regressor',score_metric='neg_mean_squared_error')
    automator.find_best_algorithm()
    automator.print_best_space()
    