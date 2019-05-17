from data.utilities import clf_single_target
from mlautomator.mlautomator import MLAutomator

if __name__=='__main__':
    x,y=clf_single_target('pima-indians-diabetes.csv')
    automator=MLAutomator(x,y,iterations=2)
    automator.find_best_algorithm()
    automator.print_best_space()