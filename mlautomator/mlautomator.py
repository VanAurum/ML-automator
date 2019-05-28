#Standard Python libary imports
import time
import joblib
import os
from sklearn.pipeline import Pipeline 
from sklearn.feature_selection import SelectKBest


#3rd party imports
from hyperopt import fmin, tpe, STATUS_OK, Trials

#Local imports
from mlautomator.objectives.classifier_objectives import Classifiers
from mlautomator.objectives.regressor_objectives import Regressors
from mlautomator.search_spaces import get_space, classifiers, regressors
from mlautomator.get_model import get_model


class MLAutomator:
    '''A tool for automating the algorithm tuning process in data science applications.

    MLAutomator leverages the intelligent search properties of Hyperopt to reduce 
    hyperparameter tuning times across large hyperparameter search spaces.  This extra 
    time allows you to spot-check a larger cross section of algorithms on your dataset.  

    MLAutomator does not perform predictions, its sole function is to find the optimal pre-processors, features, and 
    hyperparameters.

    Args:
        x_train (numpy ndarray): The training data that the models will be trained on.
        y_train (numpy ndarray): The target variables for the model.
        algo_type (str, optional, default='classifier'): Accepts 'classifier' or 'regressor'.  
        score_metric (str, optional, default='accuracy'): The scoring metric that Hyperopt will minimize on.  
        iterations (int, optional, default=25): The number of trials that Hyperopt will run on each algorithm candidate.
        num_cv_folds (int, optional, default=10): The number of folds to use in cross validation.
        repeats (int, optional, default=1): The number of passes to perform on cross validation.
        specific_algos (list of strings, default=None): A list of objective keys to overide comprehensive search.

    Public Attributes:
        start_time (time object): Used to measure elapsed time during training.
        count (int): A count of the total number of search space permutations analyzes.
        master_results (list): A history of all search spaces and their results.
        x_train (numpy ndarray): The training data passed to MLAutomator.
        y_train (numpy ndarray): The target data passed to MLAutomator.
        type (string): The type of model being built - classifier or regressor.
        score_metric (string): The score metric models are being optimized on.
        iterations (int): The number of iterations to perform on each objective function.
        num_cv_folds (int): The number of folds to use in cross validation.
        repeats (int): The number of repeats to perform in cross-validations.
        best_space (dict): The best search space discovered in the optimization process.
        best_algo (string): The description of the top performing algorithm.
        num_features (int): The number of features in the training data.
        num_samples (int): The number of samples in the training (and target) data.
    '''

    def __init__(
        self,
        x_train, 
        y_train,
        algo_type = 'classifier', 
        score_metric = 'accuracy',
        iterations = 25, 
        num_cv_folds = 10, 
        repeats = 1,
        specific_algos = None,
    ):
        # Beginning of initialization
        self.count = 0
        self.master_results = []
        self.x_train = x_train.copy()
        self.y_train = y_train.copy()
        self.type = algo_type
        self.score_metric = score_metric
        self.iterations = iterations
        self.num_cv_folds = num_cv_folds
        self.repeats = repeats
        self._initialize_best()
        self.num_features = self.x_train.shape[1]
        self.num_samples = self.x_train.shape[0]
        self.specific_algos = specific_algos

        # Initialize empty attributes
        self._objective = None
        self.keys = None
        self.best_space = None
        self.best_algo = None
        self.found_best_on = None
        self.best_pipeline = None
        self.start_time = None


    def __repr__(self):
        return f'MLAutomator ({self.type}, {self.score_metric}, Iterations: {self.iterations})'   


    def _initialize_best(self):
        '''Utility method for properly initializing the 'best' parameter.  
        
        After each iteration, the score will be checked
        against 'best' to see if the space used in that iteration was superior.  Depending on the scoring 
        metric used, 'best' needs to be initialized to different values.
        '''
        initializer_dict = {
            'accuracy': 0, 
            'neg_log_loss': 5,
            'neg_mean_squared_error' : 10000000,
        }
        self.best = initializer_dict[self.score_metric]


    def _get_objective(self,obj):
        '''A dictionary look-up utility that offers a clean way of looping through the objective 
        functions by key and returning the function call.  

        Args:
            obj (string): Key value representing the ojective function to call.

        Returns:
            <objective_list[obj]> (function call): A call to the appropriate objective function.    
        '''

        if self.type == 'classifier':
            objective_list = {        
                        '01': Classifiers.objective01,
                        '02': Classifiers.objective02,
                        '03': Classifiers.objective03,
                        '04': Classifiers.objective04,
                        '05': Classifiers.objective05,
                        '06': Classifiers.objective06,
                        '07': Classifiers.objective07,
                    }   
        else:  
            objective_list= {        
                        '01': Regressors.objective01,
                        '02': Regressors.objective02,
                        '03': Regressors.objective03,
                        '04': Regressors.objective04,
                        '05': Regressors.objective05,
                    }     


        return objective_list[obj]


    def _minimize_this(self,space):
        '''This is the "function to be minimized" by hyperopt. 
        
        This gets passed to the fmin function within the method find_best_algorithm.

        Args:
            space (dict): Subset of total search space selected by Hyperopt.
        '''
        iter_start = time.time()
        loss, algo = self._objective(self, space)
        self.count += 1

        # time methods for providing analytics on how each iteration is taking.
        iter_end = round((time.time()-iter_start)/60, 3)
        total_time = round((time.time()-self.start_time) / 60, 3)
        avg_time = round((total_time / self.count), 3)
        
        # If current loss is better than the best score reset best space, 
        # best score, and best algorithm.
        if loss < self.best:  
            self.best = loss
            self.best_space = space
            self.best_algo = algo
            self.found_best_on = self.count
            print('')
            print('new best score:', self.best)
            for key,values in space.items():
                print(str(key) + ' : ' + str(values))
            print('')    
            
        else:  
            str1 = 'Scanning '+algo+'.'
            str2 =' No Improvement. Iter time: '+str(iter_end)+'.'
            str3 =' Total Time Elapsed: '+str(total_time)+'.'
            str4 =' AVG Time: '+str(avg_time)
            print(str1 + str2 + str3 + str4) 

        self.master_results.append([loss, space])

        
        return {'loss': loss, 'status': STATUS_OK}


    def find_best_algorithm(self):   
        '''Finds the top permorning algorithm on the data.  
        
        It loops through each objective function that is 
        provided in the array 'objectives' and passes it to the Hyperopt function fmin.  fmin 
        will intelligently search the search spaces for each algorithm and attempt to minimize (optimize) 
        the scoring function provided.
        '''
        self.start_time = time.time()

        #If use provides specific objective list use that. 
        if self.specific_algos:
            objectives = self.specific_algos
        #Otheriwse, retrieve all objective keys from corresponding space.    
        else:    
            objectives = self.get_obj_key_list()

        #Pass each objective function to Hyperopt.
        for obj in objectives:
            self._objective = self._get_objective(obj)
            space = obj  
            trials = Trials()
            fmin(
                fn = self._minimize_this,
                space = get_space(self, space),
                algo = tpe.suggest,
                max_evals = self.iterations,
                trials = trials,
            )    


    def get_obj_key_list(self):
        '''Retrieves list of keys from classifier/regressor search spaces.
        '''
        if self.type == 'classifier':
            return classifiers().keys()
        elif self.type == 'regressor':
            return regressors().keys()   


        return []


    def print_best_space(self):

        if not self.best_space:
            print('Best space has not been determined yet. No models have tried on this data.')
            return 

        '''Prints out a report with the best algorithm and its configuration.
        '''
        print('Best Algorithm Configuration:')
        print('    ' + 'Best algorithm: '+ self.best_algo)
        print('    ' + 'Best ' + self.score_metric + ' : ' + str(self.best))
        for key,val in self.best_space.items():
            print('    '+ str(key)+' : '+ str(val), end='\n')                                
        print('    ' + 'Found best solution on iteration '+ str(self.found_best_on) + ' of ' + str(self.count)) 
        print('    ' + 'Validation used: ' +str(self.num_cv_folds) + '-fold cross-validation')   


    def save_best_pipeline(self, directory, filename='pipeline.joblib'):

        # In case the find_best_algorithm method hasn't been run yet...
        if not self.best_pipeline:
            print('No pipeline has been fit. Try running fit_best_pipeline() first.')
            return

        if not directory: 
            print ('No directory has been specified and is a requirement.')
            return    

        save_as = os.path.join(directory, filename)
        # Export the pipeline to a file
        joblib.dump(self.best_pipeline, open(save_as, 'wb'))
        print('Best model has been saved to: '+save_as)


    def load_best_pipeline(self, filename):

        if not filename: 
            print ('No directory has been specified and is a requirement.')
            return    

        # Load the pipeline from a file
        self.best_pipeline = joblib.load(filename)     
 

    def fit_best_pipeline(self):

        # In case the find_best_algorithm method hasn't been run yet...
        if not self.best_space:
            print('Best model has not been determined. Run find_best_algorithm() first.')
            return

        print('\n')
        print('Fitting best pipeline...')    

        # Assemble pipeline variables
        k_best = self.best_space['k_best']
        scaler = self.best_space['scaler']
        model = get_model(self.best_algo, self.best_space)

        # Build optimal pipeline again
        pipeline = []
        pipeline = Pipeline([
            ('scaler', scaler),
            ('select_best', SelectKBest(k = k_best)),
            ('classifier', model),
        ])

        print(model.get_params)
        self.best_pipeline = pipeline.fit(self.x_train, self.y_train)
        print('Completed fitting the model on entire dataset with optimal parameters')
     
    
