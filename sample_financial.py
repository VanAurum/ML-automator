from data.utilities import clf_prep
from mlautomator.mlautomator import MLAutomator

if __name__=='__main__':
    
    x,y=clf_prep('GOLD_D.csv')
    automator=MLAutomator(x, y, iterations=200, specific_algos=['01'], score_metric='neg_log_loss')
    automator.find_best_algorithm()
    automator.print_best_space()
    #automator.fit_best_model()
    