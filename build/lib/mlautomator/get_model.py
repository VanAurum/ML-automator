
#Classifier imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

#regressor imports
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Local imports
from mlautomator.search_keys import get_keys




def get_model(algo, space_dict):

    keys = get_keys(algo)
    space = {k:space_dict[k] for k in set(space_dict).intersection(keys)}

    model_lib = { 
        'xgboost_classifier': XGBClassifier,
        'xgboost_regressor':  XGBRegressor,
        'SGDClassifier': SGDClassifier,
        'SGDRegressor': SGDRegressor,
        'RandomForestClassifier': RandomForestClassifier,
        'RandomForestRegressor': RandomForestRegressor,
        'SVC': SVC,
        'SVR': SVR,          
        'LogisticRegression' : LogisticRegression,
        'KNeighborClassifier': KNeighborsClassifier,
        'KNeighborRegressor': KNeighborsRegressor,
        'GaussianNB': GaussianNB,
    }

    return model_lib[algo](**space)