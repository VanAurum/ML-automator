# ML Automator
Machine Learning Automator ('ML Automator' for short) is an automation project that integrates the Hyperopt asynchronous optimization engine with the main learning algorithms from Python's Sci-kit Learn to generate a really fast, automated tool for tuning machine learning algorithms.  

## Usage 

```Python
import MLAutomator

#Compile training and target data from .csv file
x,y=get_data('pima-indians-diabetes.csv')

automator=MLAutomator(x,y,iterations=25)
automator.find_best_algorithm()
automator.print_best_space()
```

MLAutomator can typically find a 98th percentile solution in a fraction of the time of Gridsearch of randomized search.  Here it did a 
comprehensive scan across all hyperparameters for 6 common machine learning algorithms and produced exceptional model performance for the classic Pima Indians Diabetes dataset.

```
Best Algorithm Configuration:
    Best algorithm: Logistic Regression
    Best accuracy : 77.73239917976761%
    C : 0.02341
    k_best : 6
    penalty : l2
    scaler : RobustScaler(
        copy=True, 
        quantile_range=(25.0, 75.0), 
        with_centering=True,
        with_scaling=True)
    solver : lbfgs
    Found best solution on iteration 132 of 150
    Validation used: 10-fold cross-validation
```