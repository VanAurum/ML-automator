from data.utilities import clf_prep
from mlautomator.mlautomator import MLAutomator

if __name__=='__main__':

    x,y=clf_prep('pima-indians-diabetes.csv')
    automator=MLAutomator(x,y,iterations=30)
    #automator.find_best_algorithm()
    automator.print_best_space()
    print(automator)
    