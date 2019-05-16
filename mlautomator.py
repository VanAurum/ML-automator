#Standard Python libary imports
import time 
import numpy as np

#Local imports
from objective_functions import Classifiers, Regressors
from search_spaces import get_space
from main import get_data

#3rd party imports
from hyperopt import hp, fmin, tpe, rand, STATUS_OK, Trials



class MLAutomator:
    '''
    '''

    def __init__(
        self,
        x_train, 
        y_train,
        algo_type='classifier', 
        score_metric='accuracy',
        iterations=25, 
        num_cv_folds=10, 
        repeats=1,
    ):

        self.start_time=None
        self.count=0
        self.objective=None
        self.keys=None
        self.master_results=[]
        self.x_train=x_train
        self.y_train=y_train
        self.type=algo_type
        self.score_metric=score_metric
        self.iterations=iterations
        self.num_cv_folds=10
        self.repeats=repeats
        self.objective=None
        self._initialize_best()
        self.best_space=None


    def _initialize_best(self):
        '''
        '''
        initializer_dict={
            'accuracy': 0, 
            'neg_log_loss': 5,
        }
        self.best=initializer_dict[self.score_metric]



    def get_objective(self,obj):
        '''
        '''
        if self.type=='classifier':
        
            objective_list= {        
                        '01': Classifiers.objective01,
                        '02': Classifiers.objective02,
                        '03': Classifiers.objective03,
                        '04': Classifiers.objective04,
                            }
        
        else:  

            objective_list= {        
                        '01': Regressors.objective01,
                        '02': Regressors.objective02,
                        '03': Regressors.objective03,
                        '04': Regressors.objective04,
                            }            

        return objective_list[obj]


    def f(self,space):
        '''
        '''
        iter_start=time.time()
        loss, algo = self.objective(self,space)
        self.count+=1
        iter_end=round((time.time()-iter_start)/60,3)
        total_time=round((time.time()-self.start_time)/60,3)
        avg_time=round((total_time/self.count),3)
        
        str1= 'Placeholder '
        str2=' No Improvement. Iter time: '+str(iter_end)+'.'
        str3=' Total Time Elapsed: '+str(total_time)+'.'
        str4=' AVG Time: '+str(avg_time)
        
        if loss < self.best:
            
            self.best = loss
            self.best_space=space
            self.best_algo=algo
            print('')
            print('new best score:', self.best)
            for key,values in space.items():
                print(str(key) +' : ' +str(values))
            print('')    
            
        else:  
            print(str1+str2+str3+str4) 

        self.master_results.append([loss,space])
        
        return {'loss': loss, 'status': STATUS_OK}


    def find_best_algorithm(self):   
        '''
        '''
        self.start_time=time.time()
        objectives=['04']

        for obj in objectives:
            keys=obj
            self.objective=self.get_objective(obj)
            seed=np.random.seed(1985)
            space=obj  
            trials=Trials()
            best=fmin(
                fn=self.f,
                space=get_space(space),
                algo=tpe.suggest,
                max_evals=self.iterations,
                trials=trials
                        )    


    def print_best_space(self):
        '''
        '''
        print('Best Algorithm Configuration:')
        print('    '+'Best algorithm: '+self.best_algo)
        print('    '+'Best '+self.score_metric+' : '+str(self.best))
        for key,val in self.best_space.items():
            print('    '+ str(key)+' : '+ str(val), end='\n')                                
        print('    '+'Found solution in '+str(self.iterations)+' iterations') 
        print('    '+'Validation used: '+str(self.num_cv_folds)+'-fold cross-validation')          


if __name__=='__main__':
    x,y=get_data('pima-indians-diabetes.csv')
    automator=MLAutomator(x,y,iterations=5)
    automator.find_best_algorithm()
    automator.print_best_space()
    
