#Standard Python libary imports
import time 
import numpy as np
import pandas as pd

#Local imports
from objective_functions import Classifiers
from search_spaces import get_space

#3rd party imports
from hyperopt import hp, fmin, tpe, rand,STATUS_OK, Trials


def get_data(filename=None):
    
    '''
    '''
    
    seed=np.random.seed(1985)
    
    #Load Numerai data from disk    
    data_directory='datasets/'
    training_data = pd.read_csv(data_directory+filename, header=0)
    
    # Transform the loaded CSV data into numpy arrays
    columns = list(training_data)
    features=columns[:-1]
    X = training_data[features]
    Y = training_data[columns[-1]]
         
    x_train=X.values
    y_train=Y.values

    return x_train, y_train


def get_objective(obj):
    
    objective_list= {        
                        '01': Classifiers.objective01,
                        '02': Classifiers.objective02,
                        '03': Classifiers.objective03,
                        '04': Classifiers.objective04,
                    }
    
    return objective_list[obj]

def f(space):

    global best
    global count

    iter_start=time.time()
    loss = objective(space, X_train, y_train)
    count+=1
    iter_end=round((time.time()-iter_start)/60,3)
    total_time=round((time.time()-start_time)/60,3)
    avg_time=round((total_time/count),3)
    
    str1= 'Placeholder '
    str2=' No Improvement. Iter time: '+str(iter_end)+'.'
    str3=' Total Time Elapsed: '+str(total_time)+'.'
    str4=' AVG Time: '+str(avg_time)
    
    if loss < best:
        
        best = loss
        
        print('')
        print('new best:', best)
        for key,values in space.items():
            print(key, values)
        print('')    
        
    else:  
        print(str1+str2+str3+str4) 

    master_results.append([loss,space])
    
    return {'loss': loss, 'status': STATUS_OK}


def main():   
    
    #Declare a set of global variables that are used throughout
    global start_time, count, objective, best, keys, master_results
    global X_train, y_train
    
    start_time=time.time()
    objectives=['01','02','03']
    max_evals=25  
    master_results=[]

    for obj in objectives:
        keys=obj
        objective=get_objective(obj)
        seed=np.random.seed(1985)
        best=0.75
        count=0

        X_train, y_train=get_data('pima-indians-diabetes.csv')

        space=obj  
        trials=Trials()
        best=fmin(fn=f,
                    space=get_space(space),
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials)

if __name__=='__main__':

    main()                    