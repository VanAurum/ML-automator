from data.utilities import clf_prep
from mlautomator.mlautomator import MLAutomator

if __name__=='__main__':
    
    x,y=clf_prep('boston_housing.csv')
    automator=MLAutomator(x,y,iterations=20)
    automator.find_best_algorithm()
    automator.print_best_space()