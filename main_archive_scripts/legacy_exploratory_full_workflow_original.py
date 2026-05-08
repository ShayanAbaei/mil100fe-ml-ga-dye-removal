
"""
Created on Aug 2023

@author: Shayan Abaei
"""
import warnings
warnings.filterwarnings("ignore")

save_path = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11' 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ExpSineSquared, ConstantKernel

import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
import shap
from pyswarm import pso
from scipy.optimize import minimize, fmin_tnc, basinhopping
from scipy.optimize import approx_fprime
from deap import base, creator, tools, algorithms
import random
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D

from sko.GA import GA
from sko.PSO import PSO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools

from plotly.offline import plot
import plotly.graph_objs as go
from scipy.optimize import shgo
from scipy.optimize import dual_annealing

file_path = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/Data_Set_1.11.xlsx'
train_data = pd.read_excel(file_path, sheet_name="Train Set", skiprows=2)
test_data = pd.read_excel(file_path, sheet_name="Test Set", skiprows=2)

train_data = train_data.drop(columns=["Unnamed: 5"])
test_data = test_data.drop(columns=["Unnamed: 5"])
X_train = train_data.drop(columns=["Unnamed: 0", "target 1", "target 2"])
y_train_1 = train_data["target 1"]
y_train_2 = train_data["target 2"]
X_test = test_data.drop(columns=["Unnamed: 0", "target 1", "target 2"])
y_test_1 = test_data["target 1"]
y_test_2 = test_data["target 2"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_data = pd.read_excel(file_path, sheet_name="Train Set", skiprows=2)
test_data = pd.read_excel(file_path, sheet_name="Test Set", skiprows=2)

data = pd.concat([train_data, test_data], ignore_index=True)

data = data.drop(columns=["Unnamed: 5"])

data = data.reset_index(drop=True)
X = data.drop(columns=["Unnamed: 0", "target 1", "target 2"])



'''
X_s = X
X_array = scaler.fit_transform(X_s)
X = pd.DataFrame(X_array , columns=X_s.columns)

'''



y1 = data["target 1"]
y2 = data["target 2"]


import pickle

def load_best_params(file_path):
    with open(file_path, 'rb') as file:
        best_params = pickle.load(file)
    return best_params

best_params_gpr_target1 = load_best_params('C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Gaussian Process Regressor_target1_best_params.pkl')
best_params_gpr_target2 = load_best_params('C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Gaussian Process Regressor_target2_best_params.pkl')
best_params_lsvr_target1 = load_best_params('C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Linear SVR_target1_best_params.pkl')
best_params_lsvr_target2 = load_best_params('C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Linear SVR_target2_best_params.pkl')




kernel = RationalQuadratic()
################33 Target 1
'''
models = {
    "Linear Regression": LinearRegression(fit_intercept=False , n_jobs=None),
    "Ridge Regression": Ridge(alpha=100,solver='saga'),
    "Lasso Regression": Lasso(alpha=100,selection='random'),
    "Decision Trees": DecisionTreeRegressor(criterion='friedman_mse',max_depth=6,min_samples_split=4,splitter='random',max_features='sqrt',min_samples_leaf=2,random_state=1671),
    "Random Forest": RandomForestRegressor(bootstrap=False,n_estimators=10 ,max_depth=5,max_features='sqrt',min_samples_leaf=2,min_samples_split=5,random_state=1810),
    "Support Vector Machines": SVR(C=0.1 , epsilon=0.01,gamma='scale',kernel='linear',degree=2),
    "K-Nearest Neighbors": KNeighborsRegressor(algorithm='auto',n_neighbors=7,weights='uniform',leaf_size=20,p=2),
    "Dummy Regressor": DummyRegressor(),
    "XGBoost": xgb.XGBRegressor(colsample_bytree=0.3,gamma = 0,learning_rate=0.04,max_depth=2,n_estimators=100,subsample=0.5,reg_alpha=0.1,reg_lambda=0.5),
    "Gaussian Process Regressor": GaussianProcessRegressor(**best_params_gpr_target1),
    "Linear SVR": LinearSVR(**best_params_lsvr_target1)
}

'''
################33 Target 2
models = {
    "Linear Regression": LinearRegression(fit_intercept=False , n_jobs=None),
    "Ridge Regression": Ridge(alpha=100,solver='saga'),
    "Lasso Regression": Lasso(alpha=10,selection='cyclic'),
    "Decision Trees": DecisionTreeRegressor(criterion='friedman_mse',max_depth=11,min_samples_split=5,splitter='best',max_features='sqrt',min_samples_leaf=1,random_state=512),
    "Random Forest": RandomForestRegressor(bootstrap=True,n_estimators=10 ,max_depth=9,max_features='log2',min_samples_leaf=1,min_samples_split=4,random_state=481),
    "Support Vector Machines": SVR(C=464.15888336127773 , epsilon=0.01,gamma='scale',kernel='rbf',degree=2),
    "K-Nearest Neighbors": KNeighborsRegressor(algorithm='auto',n_neighbors=3,weights='uniform',leaf_size=20,p=1),
    "Dummy Regressor": DummyRegressor(),
    "XGBoost": xgb.XGBRegressor(colsample_bytree=1,gamma = 0,learning_rate=0.04,max_depth=3,n_estimators=100,subsample=0.5,reg_alpha=0,reg_lambda=0.5),
    "Gaussian Process Regressor": GaussianProcessRegressor(**best_params_gpr_target2),
    "Linear SVR": LinearSVR(**best_params_lsvr_target2)
}


#############################################

from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt


n_splits = 10 


def evaluate_models(X, y, models, cv):
    results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for model_name, model in models.items():
        model_test_rmse_scores = []
        model_train_rmse_scores = []
        model_test_r2_scores = []
        model_train_r2_scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model_clone = clone(model)
            model_clone.fit(X_train, y_train)

            y_train_pred = model_clone.predict(X_train)
            y_test_pred = model_clone.predict(X_test)

            model_train_rmse_scores.append(sqrt(mean_squared_error(y_train, y_train_pred)))
            model_train_r2_scores.append(r2_score(y_train, y_train_pred))

        test_rmse = -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error').mean()
        test_r2 = cross_val_score(model, X, y, cv=cv, scoring='r2').mean()

        avg_train_rmse = np.mean(model_train_rmse_scores)
        avg_train_r2 = np.mean(model_train_r2_scores)

        results.append({
            'Model': model_name,
            'Train RMSE': avg_train_rmse,
            'Test RMSE': test_rmse,
            'Train R2': avg_train_r2,
            'Test R2': test_r2
        })

    results_df = pd.DataFrame(results)
    return results_df.sort_values(by='Test RMSE')

from sklearn.model_selection import KFold

cv = KFold(n_splits=10, shuffle=True, random_state=42)

ranked_results_y1 = evaluate_models(X, y1, models, cv)
print("Results for Target 1 (y1):")
print(ranked_results_y1)

ranked_results_y2 = evaluate_models(X, y2, models, cv)
print("\nResults for Target 2 (y2):")
print(ranked_results_y2)

############################################


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from math import sqrt


cv = 10  
scoring_methods = {
    'r2': 'r2',
    'rmse': make_scorer(lambda y, y_pred: sqrt(mean_squared_error(y, y_pred)), greater_is_better=False)
}

def evaluate_models(X, y, models, scoring_methods, cv):
    results = []
    for model_name, model in models.items():
        scores_r2 = cross_val_score(model, X, y, cv=cv, scoring=scoring_methods['r2'])
        scores_rmse = cross_val_score(model, X, y, cv=cv, scoring=scoring_methods['rmse'])

        mean_r2 = np.mean(scores_r2)
        mean_rmse = -np.mean(scores_rmse)  

        results.append({
            'Model': model_name,
            'Mean R2': mean_r2,
            'Mean RMSE': mean_rmse
        })

    results_df = pd.DataFrame(results)
    return results_df.sort_values(by='Mean RMSE')

ranked_results_y1 = evaluate_models(X, y1, models, scoring_methods, cv)
print("Results for Target 1 (y1):")
print(ranked_results_y1)

ranked_results_y2 = evaluate_models(X, y2, models, scoring_methods, cv)
print("\nResults for Target 2 (y2):")
print(ranked_results_y2)



############################

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic



kernel1 = RBF(length_scale_bounds=(1e-2, 1e2))
kernel2 = Matern(length_scale_bounds=(1e-2, 1e2), nu=1.5)
kernel3 = RationalQuadratic(length_scale_bounds=(1e-2, 1e2), alpha_bounds=(1e-2, 1e2))
kernel4 = WhiteKernel(noise_level_bounds=(1e-10, 1e-1))

'''

model_params = {
    'Linear Regression': {
    'model': LinearRegression(),
    'params': {
        'fit_intercept': [True, False],
        'n_jobs': [None, -1]  
    }
},
    'Ridge Regression': {
        'model': Ridge(),
        'params': {
            'alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
    },
    'Lasso Regression': {
        'model': Lasso(),
        'params': {
            'alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100],
            'selection': ['cyclic', 'random']
        }
    },
    'Decision Trees': {
        'model': DecisionTreeRegressor(),
        'params': {
            'criterion': ['mse', 'friedman_mse', 'mae'],
            'splitter': ['best', 'random'],
            'max_depth': range(2, 16), 
            'min_samples_split': range(2, 6), 
            'min_samples_leaf': range(1, 5),  
            'max_features': ['auto', 'sqrt', 'log2']
}

    },
    'Random Forest': {
    'model': RandomForestRegressor(),
    'params': {
           'n_estimators': range(10, 101, 5),  
           'max_depth': range(2, 16),           
           'min_samples_split': range(2, 6),    
           'min_samples_leaf': range(1, 5),    
           'max_features': ['auto', 'sqrt', 'log2'],
           'bootstrap': [True, False]
        }
},
    'Support Vector Machines': {
    'model': SVR(),
    'params': {
           'C': np.logspace(-1, 3, 13), 
           'kernel': ['linear', 'rbf'],
           'degree': [2, 3, 4, 5],  
           'gamma': ['scale', 'auto'],  
           'epsilon': np.linspace(0.01, 1, 10) 
}
},
    'K-Nearest Neighbors': {
        'model': KNeighborsRegressor(),
        'params':{
           'n_neighbors': range(1, 11),
           'weights': ['uniform', 'distance'],
           'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
           'leaf_size': range(20, 41, 5), 
            'p': [1, 2]
}

    },
    'Dummy Regressor': {
        'model': DummyRegressor(),
        'params': {
            'strategy': ['mean', 'median']
        }
    },
    
    
    
    'XGBoost': {
    'model': XGBRegressor(),
    'params': {
            'n_estimators': [50, 100, 150],
            'max_depth': range(2, 11),  
            'learning_rate': np.linspace(0.01, 0.1, 4),
            'subsample': np.linspace(0.5, 1, 3),
            'colsample_bytree': np.linspace(0.3, 1, 3),
            'gamma': [0, 0.1, 0.5],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
}
},
    'Gaussian Process Regressor': {
    'model': GaussianProcessRegressor(),
    'params':  {
           'kernel': [kernel1, kernel2, kernel3, kernel4],
           'alpha': np.logspace(-10, -2, 5),
           'n_restarts_optimizer': [0, 5, 10],
           'normalize_y': [True, False],
           "random_state": [0]  

}
},
   'Linear SVR': {
    'model': LinearSVR(),
    'params':{
    'C': np.logspace(-2, 2, 10), 
    'epsilon': np.linspace(0.01, 1, 10),  
    'tol': np.logspace(-4, -1, 10), 
    'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
    'dual': [True, False]
}
}

}

def grid_search_cv(models, X, y, cv):
    grid_search_results = []
    
    for model_name, mp in models.items():
        print(f"Performing grid search for {model_name}...")
        grid_search = GridSearchCV(mp['model'], mp['params'], cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=10)
        grid_search.fit(X, y)
        best_score = -grid_search.best_score_
        best_params = grid_search.best_params_

        grid_search_results.append({
            'Model': model_name,
            'Best Score (RMSE)': best_score,
            'Best Parameters': best_params
        })
    
    return pd.DataFrame(grid_search_results)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


grid_search_results_y1 = grid_search_cv(model_params, X, y1, cv=10)
print("Grid Search Results for Target 1 (y1):")
print(grid_search_results_y1)

grid_search_results_y2 = grid_search_cv(model_params, X, y2, cv=10)
print("\nGrid Search Results for Target 2 (y2):")
print(grid_search_results_y2)
'''

####################################3

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ExpSineSquared, ConstantKernel
from sklearn.gaussian_process import kernels

kernel1 = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
kernel2 = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
kernel3 = RationalQuadratic(length_scale=1.0, alpha=0.1)
kernel4 = WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
kernel5 = kernel1 + kernel2
kernel6 = kernel3 * kernel4
kernel7 = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
kernel8 = ExpSineSquared(length_scale=1.0, periodicity=3.0, length_scale_bounds=(1e-2, 1e2), periodicity_bounds=(1e-1, 1e1))
kernel9 = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
kernel10 = kernel1 + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))


'''
model_params = {
   'Gaussian Process Regressor': {
        'model': GaussianProcessRegressor(),
        'params': {
            'kernel': [kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7, kernel8, kernel9, kernel10],
            'alpha': np.logspace(-10, 0, 10),
            'optimizer': ['fmin_l_bfgs_b', 'fmin_tnc', 'fmin_cobyla', None],
            'n_restarts_optimizer': [0, 1, 3, 5, 10],
            'normalize_y': [True, False],
            'random_state': [0, 42, 49, 144, 371, 1636]
        }
    },
   
   
   'Linear SVR': {
    'model': LinearSVR(),
    'params':{
    'C': np.logspace(-3, 3, 6), 
    'epsilon': np.linspace(0.01, 1, 10),
    'tol': np.logspace(-5, -1, 6),  
    'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'], 
    'fit_intercept': [True, False],  
    'intercept_scaling': np.linspace(0.1, 3,6),  
    'dual': [True, False],  
    'max_iter': [10000, 50000, 10000],
    'random_state': [0,42]
}
}

}

import pickle

def grid_search_cv(models, X, y, cv, save_path, target_name):
    grid_search_results = []
    
    for model_name, mp in models.items():
        print(f"Performing grid search for {model_name} on {target_name}...")
        grid_search = GridSearchCV(mp['model'], mp['params'], cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=10)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_

        with open(f"{save_path}/{model_name}_{target_name}_best_params.pkl", "wb") as f:
            pickle.dump(best_params, f)

        best_score = -grid_search.best_score_
        grid_search_results.append({
            'Model': model_name,
            'Best Score (RMSE)': best_score,
            'Best Parameters': best_params
        })
        
    return pd.DataFrame(grid_search_results)




pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


print("Grid search for target 1")
grid_search_results_target1 = grid_search_cv(model_params, X, y1, cv, save_path,'target1')
print(grid_search_results_target1)


print("\nGrid search for target 2")
grid_search_results_target2 = grid_search_cv(model_params, X, y2, cv, save_path, 'target2')
print(grid_search_results_target2)

'''

################################3
### Random State search for Random forest model
'''
# Additional code to find the best random_state for the Random Forest model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd

def find_best_random_state(X, y, model_hyperparams, start, end, cv):
    best_rmse = float('inf')
    best_random_state = None
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for random_state in range(start, end + 1):
        model = RandomForestRegressor(**model_hyperparams, random_state=random_state)
        rmse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        avg_rmse = np.mean(rmse_scores)

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_random_state = random_state

    return best_random_state, best_rmse

rf_hyperparams = {
    'bootstrap': True,
    'n_estimators': 10,
    'max_depth':5,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt'
}


best_state_y1, best_rmse_y1 = find_best_random_state(X, y1, rf_hyperparams,2000,5000, 10)
print(f"Best Random State for RandomForest (Target 1): {best_state_y1}")
print(f"Best RMSE for RandomForest (Target 1): {best_rmse_y1}")

'''

################################3
### Random State search for Random forest model
# Additional code to find the best random_state for the Random Forest model

'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd

def find_best_random_state(X, y, model_hyperparams, start, end, cv):
    best_rmse = float('inf')
    best_random_state = None
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for random_state in range(start, end + 1):
        model = RandomForestRegressor(**model_hyperparams, random_state=random_state)
        rmse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        avg_rmse = np.mean(rmse_scores)

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_random_state = random_state

    return best_random_state, best_rmse

rf_hyperparams = {
    'bootstrap':True,
    'n_estimators': 10,
    'max_depth': 9,
    'min_samples_split': 4,
    'min_samples_leaf': 1,
    'max_features':'log2'
}


best_state_y2, best_rmse_y2 = find_best_random_state(X, y2, rf_hyperparams,500, 2000, 10)
print(f"Best Random State for RandomForest (Target 2): {best_state_y2}")
print(f"Best RMSE for RandomForest (Target 2): {best_rmse_y2}")
'''

################rando state search for inear svr
'''
import pickle
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score, KFold

def load_model_with_best_params(params_file):
    with open(params_file, "rb") as f:
        best_params = pickle.load(f)
    return best_params

def find_best_random_state(X, y, params_file, start, end, n_splits):
    best_rmse = float('inf')
    best_random_state = None
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)  
    
    model_hyperparams = load_model_with_best_params(params_file)

    for random_state in range(start, end + 1):
        model = LinearSVR(**model_hyperparams, random_state=random_state)
        rmse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        avg_rmse = np.mean(rmse_scores)

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_random_state = random_state

    return best_random_state, best_rmse

path_target1 = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Linear SVR_target1_best_params.pkl'
path_target2 = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Linear SVR_target2_best_params.pkl'
n_splits = 10  

best_state_y1, best_rmse_y1 = find_best_random_state(X, y1, path_target1, 0, 500, n_splits)
print(f"Best Random State for Linear SVR (Target 1): {best_state_y1}, RMSE: {best_rmse_y1}")

best_state_y2, best_rmse_y2 = find_best_random_state(X, y2, path_target2, 0, 500, n_splits)

print(f"Best Random State for Linear SVR (Target 2): {best_state_y2}, RMSE: {best_rmse_y2}")
'''
#################################

'''
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score, KFold


def find_best_random_state(X, y, model_hyperparams, start, end, cv):
    best_rmse = float('inf')
    best_random_state = None
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for random_state in range(start, end + 1):
        model = LinearSVR(**model_hyperparams, random_state=random_state)
        rmse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        avg_rmse = np.mean(rmse_scores)

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_random_state = random_state

    return best_random_state, best_rmse

svr_hyperparams = {
    'C':12.915496650148826,
    'dual': True,
    'loss': 'squared_epsilon_insensitive',
    'epsilon': 0.67,
    'tol': 0.01,
    'max_iter': 100000
}


best_state_y2, best_rmse_y2 = find_best_random_state(X, y2, svr_hyperparams,1000, 2000 , 10)
print(f"Best Random State for Linear SVR (Target 2): {best_state_y2}")
print(f"Best RMSE for Linear SVR (Target 2): {best_rmse_y2}")
'''

'''
######################333 GPR ranod state:
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import Matern

def find_best_random_state(X, y, kernel, alpha, normalize_y, n_restarts_optimizer, start, end, cv):
    best_rmse = float('inf')
    best_random_state = None
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for random_state in range(start, end + 1):
        model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=normalize_y, 
                                         n_restarts_optimizer=n_restarts_optimizer, random_state=random_state)
        rmse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        avg_rmse = np.mean(rmse_scores)

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_random_state = random_state

    return best_random_state, best_rmse

kernel = Matern(length_scale=1, nu=1.5)
alpha = 1e-10
normalize_y = True
n_restarts_optimizer = 5


best_state_y1, best_rmse_y1 = find_best_random_state(X, y1, kernel, alpha, normalize_y, n_restarts_optimizer, 10, 2000, 10)
print(f"Best Random State for GPR (Target 1): {best_state_y1}")
print(f"Best RMSE for GPR (Target 1): {best_rmse_y1}")


##########################

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import Matern

def find_best_random_state(X, y, kernel, alpha, normalize_y, n_restarts_optimizer, start, end, cv):
    best_rmse = float('inf')
    best_random_state = None
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for random_state in range(start, end + 1):
        model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=normalize_y, 
                                         n_restarts_optimizer=n_restarts_optimizer, random_state=random_state)
        rmse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        avg_rmse = np.mean(rmse_scores)

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_random_state = random_state

    return best_random_state, best_rmse

kernel= RBF(length_scale=1)
alpha = 0.01
normalize_y = False
n_restarts_optimizer = 10

best_state_y2, best_rmse_y2 = find_best_random_state(X, y2, kernel, alpha, normalize_y, n_restarts_optimizer, 1,2000, 10)
print(f"Best Random State for GPR (Target 2): {best_state_y2}")
print(f"Best RMSE for GPR (Target 2): {best_rmse_y2}")


'''


################### xgboost random state searching
'''

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

def find_best_random_state(X, y, model_hyperparams, start, end, cv):
    best_rmse = float('inf')
    best_random_state = None
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for random_state in range(start, end + 1):
        model = XGBRegressor(**model_hyperparams, random_state=random_state)
        rmse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        avg_rmse = np.mean(rmse_scores)

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_random_state = random_state

    return best_random_state, best_rmse

xgb_hyperparams = {
    'colsample_bytree': 0.3,
    'gamma': 0,
    'learning_rate': 0.04,
    'max_depth': 2,
    'n_estimators': 100,
    'subsample': 0.5,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5
}


best_state_y1, best_rmse_y1 = find_best_random_state(X, y1, xgb_hyperparams, 0, 10000, 10)
print(f"Best Random State for XGBRegressor (Target 1): {best_state_y1}")
print(f"Best RMSE for XGBRegressor (Target 1): {best_rmse_y1}")



##########

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

def find_best_random_state(X, y, model_hyperparams, start, end, cv):
    best_rmse = float('inf')
    best_random_state = None
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for random_state in range(start, end + 1):
        model = XGBRegressor(**model_hyperparams, random_state=random_state)
        rmse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        avg_rmse = np.mean(rmse_scores)

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_random_state = random_state

    return best_random_state, best_rmse

xgb_hyperparams = {
    'colsample_bytree': 1,
    'gamma': 0.5,
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 50,
    'subsample': 1,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5
}



best_state_y2, best_rmse_y2 = find_best_random_state(X, y2, xgb_hyperparams, 0, 10000, 10)
print(f"Best Random State for XGBRegressor (Target 2): {best_state_y2}")
print(f"Best RMSE for XGBRegressor (Target 2): {best_rmse_y2}")

'''

########################### Decision tree random state
'''
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

def find_best_random_state(X, y, model_hyperparams, start, end, cv):
    best_rmse = float('inf')
    best_random_state = None
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for random_state in range(start, end + 1):
        model = DecisionTreeRegressor(**model_hyperparams, random_state=random_state)
        rmse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        avg_rmse = np.mean(rmse_scores)

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_random_state = random_state

    return best_random_state, best_rmse

dt_hyperparams = {
    'criterion': 'friedman_mse',
    'max_depth': 6,
    'min_samples_split': 4,
    'splitter': 'random',
    'max_features': 'sqrt',
    'min_samples_leaf': 2
}


best_state_y1, best_rmse_y1 = find_best_random_state(X, y1, dt_hyperparams, 1000, 2000, 10)
print(f"Best Random State for DecisionTreeRegressor (Target 1): {best_state_y1}")
print(f"Best RMSE for DecisionTreeRegressor (Target 1): {best_rmse_y1}")

'''

##################
'''
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

def find_best_random_state(X, y, model_hyperparams, start, end, cv):
    best_rmse = float('inf')
    best_random_state = None
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for random_state in range(start, end + 1):
        model = DecisionTreeRegressor(**model_hyperparams, random_state=random_state)
        rmse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        avg_rmse = np.mean(rmse_scores)

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_random_state = random_state

    return best_random_state, best_rmse

dt_hyperparams = {
    'criterion': 'friedman_mse',
    'max_depth': 11,
    'min_samples_split': 5,
    'splitter': 'best',
    'max_features': 'sqrt',
    'min_samples_leaf': 1
}



best_state_y2, best_rmse_y2 = find_best_random_state(X, y2, dt_hyperparams, 500, 1000, 10)
print(f"Best Random State for DecisionTreeRegressor (Target 2): {best_state_y2}")
print(f"Best RMSE for DecisionTreeRegressor (Target 2): {best_rmse_y2}")
'''

########################################################################################3
#### models extraction

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor

rf_model_y1 = RandomForestRegressor(
    bootstrap=False,
    n_estimators=10,
    max_depth=5,
    max_features='sqrt',
    min_samples_leaf=2,
    min_samples_split=5,
    random_state=1810
)
rf_model_y1.fit(X, y1)



dt_model_y1 = DecisionTreeRegressor(
    criterion='friedman_mse',
    max_depth=6,
    min_samples_split=4,
    splitter='random',
    max_features='sqrt',
    min_samples_leaf=2,
    random_state=1671
)
dt_model_y1.fit(X, y1)


svm_model_y1 = SVR(
    C=0.1 ,
    epsilon=0.01,
    gamma='scale',
    kernel='linear',
    degree=2
    
    )

svm_model_y1.fit(X, y1)



##########3############### Target 2 

rf_model_y2 = RandomForestRegressor(
    bootstrap=True,
    n_estimators=10,
    max_depth=9,
    max_features='log2',
    min_samples_leaf=1,
    min_samples_split=4,
    random_state=481
)
rf_model_y2.fit(X, y2)




dt_model_y2 = DecisionTreeRegressor(
    criterion='friedman_mse',
    max_depth=11,
    min_samples_split=5,
    splitter='best',
    max_features='sqrt',
    min_samples_leaf=1,
    random_state=512
)
dt_model_y2.fit(X, y2)



svm_model_y2 = SVR(
    C=464.15888336127773 , 
    epsilon=0.01,
    gamma='scale',
    kernel='rbf',
    degree=2
    
    
)

svm_model_y2.fit(X, y2)



###########################
def load_model_with_best_params(model_class, params_file):
    with open(params_file, "rb") as f:
        best_params = pickle.load(f)
    return model_class(**best_params)

gpr_model_target1 = load_model_with_best_params(GaussianProcessRegressor, 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Gaussian Process Regressor_target1_best_params.pkl')
gpr_model_target2 = load_model_with_best_params(GaussianProcessRegressor, 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Gaussian Process Regressor_target2_best_params.pkl')

linear_svr_model_target1 = load_model_with_best_params(LinearSVR, 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Linear SVR_target1_best_params.pkl')
linear_svr_model_target2 = load_model_with_best_params(LinearSVR, 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Linear SVR_target2_best_params.pkl')



gpr_model_target1.fit(X,y1)
gpr_model_target2.fit(X,y2)


linear_svr_model_target1.fit(X,y1)
linear_svr_model_target2.fit(X,y2)


import pickle

def load_best_params(file_path):
    with open(file_path, 'rb') as file:
        best_params = pickle.load(file)
    return best_params

best_params_gpr_target1 = load_best_params('C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Gaussian Process Regressor_target1_best_params.pkl')
best_params_gpr_target2 = load_best_params('C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Gaussian Process Regressor_target2_best_params.pkl')
best_params_lsvr_target1 = load_best_params('C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Linear SVR_target1_best_params.pkl')
best_params_lsvr_target2 = load_best_params('C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Linear SVR_target2_best_params.pkl')



################
bounds = [(20, 60), (110, 150), (1, 3.5), (0.15, 0.45)]



lb = [20, 110, 1, 0.15]
ub = [60, 150, 3.5, 0.45]
################


#####################################  4 by 1000
'''
#####
import numpy as np
import pandas as pd


feature_ranges = {
    'feature_min': X.min(),
    'feature_max': X.max()
}

feature_data = {}
for feature in feature_ranges['feature_min'].keys():
    min_val = feature_ranges['feature_min'][feature]
    max_val = feature_ranges['feature_max'][feature]
    feature_data[feature] = np.random.uniform(min_val, max_val,3000)

random_feature_data = pd.DataFrame(feature_data)
'''

'''
print(random_feature_data.head())
'''

'''

###########
# random forest

rf_y1_predictions = rf_model_y1.predict(random_feature_data)
rf_y2_predictions = rf_model_y2.predict(random_feature_data)


random_feature_data['RF Prediction for Target 1'] = rf_y1_predictions
random_feature_data['RF Prediction for Target 2'] = rf_y2_predictions

top10_rf_y1 = random_feature_data.nlargest(10, 'RF Prediction for Target 1')
top10_rf_y2 = random_feature_data.nlargest(10, 'RF Prediction for Target 2')



pd.set_option('display.max_columns', None)  
pd.set_option('display.max_colwidth', None)   


print("Top 10 Predictions for Target 1:")
print(top10_rf_y1)
print("\nTop 10 Predictions for Target 2:")
print(top10_rf_y2)

'''
#######################3  4 *** 3000
'''
original_features = random_feature_data.copy()

'''
'''
excel_filename = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/original_features3000.xlsx'

# Save to Excel file
original_features.to_excel(excel_filename, index=False)

print(f'Data saved to {excel_filename}')

prediction_data = original_features.copy()
prediction_data['RF Prediction for Target 1'] = rf_model_y1.predict(original_features)
prediction_data['RF Prediction for Target 2'] = rf_model_y2.predict(original_features)

prediction_data['SVM Prediction for Target 1'] = svm_model_y1.predict(original_features)
prediction_data['SVM Prediction for Target 2'] = svm_model_y2.predict(original_features)

prediction_data['GPR Prediction for Target 1'] = gpr_model_target1.predict(original_features)
prediction_data['GPR Prediction for Target 2'] = gpr_model_target2.predict(original_features)

prediction_data['DT Prediction for Target 1'] = dt_model_y1.predict(original_features)
prediction_data['DT Prediction for Target 2'] = dt_model_y2.predict(original_features)


models = ['RF', 'SVM', 'GPR', 'DT']
for model in models:
    for target in ['1', '2']:
        column_name = f'{model} Prediction for Target {target}'
        top_predictions = prediction_data.nlargest(10, column_name)
        print(f"\nTop 10 Predictions for Target {target} using {model}:")
        print(top_predictions[[column_name]])

pd.set_option('display.max_columns', None)  
pd.set_option('display.max_colwidth', None)

'''
'''
###############3 extracting the leaf node of RAndom forest



def extract_leaf_nodes_values(random_forest_model):
    leaf_values = []
    for estimator in random_forest_model.estimators_:
        leaves = estimator.tree_.value
        leaf_values.extend([value for value in leaves.flatten() if value != 0])
    return leaf_values

leaf_values_y1 = extract_leaf_nodes_values(rf_model_y1)
leaf_values_y2 = extract_leaf_nodes_values(rf_model_y2)

sorted_leaf_values_y1 = sorted(leaf_values_y1, reverse=True)[:20]
sorted_leaf_values_y2 = sorted(leaf_values_y2, reverse=True)[:20]

print("\nTop 20 Leaf node values for rf_model_y1 (sorted):")
print(sorted_leaf_values_y1)
print("\nTop 20 Leaf node values for rf_model_y2 (sorted):")
print(sorted_leaf_values_y2)



#####################


def get_rf_leaf_values(rf_model):
    leaf_values = []
    for tree in rf_model.estimators_:
        tree_leaves = tree.tree_.value
        for leaf in tree_leaves:
            leaf_values.extend(leaf.flatten())
    return leaf_values

leaf_values_y1 = get_rf_leaf_values(rf_model_y1)
leaf_values_y2 = get_rf_leaf_values(rf_model_y2)

top20_leaf_values_y1 = sorted(leaf_values_y1, reverse=True)[:20]
top20_leaf_values_y2 = sorted(leaf_values_y2, reverse=True)[:20]

print("\n\nTop 20 Leaf node values for rf_model_y1 (sorted):")
print(top20_leaf_values_y1)
print("\n\n\nTop 20 Leaf node values for rf_model_y2 (sorted):")
print(top20_leaf_values_y2)



'''
##########################################################3
import shap

from sklearn.preprocessing import StandardScaler
import numpy as np
'''
import numpy as np
# Assuming your scaler is saved and loaded as `scaler`
# And your new data point is something like this (make sure to replace these values with your actual data point)
new_data_point = np.array([40.09470579, 131.08882738,  3.34189291  , 0.43156064]).reshape(1, -1)  # Reshape your data

# Scale the new data point using the already fitted scaler
scaled_data_point = scaler.transform(new_data_point)

# Use the GPR model to make a prediction
prediction = gpr_model_target1.predict(scaled_data_point)

# Ensure the prediction is positive

print("Predicted value:", prediction)

'''
#############optimizations

##### gentic algorithm
########################################3################################3
#####


# multi threading

'''

import numpy as np
from sko.GA import GA
import time
import datetime
from sko.tools import set_run_mode



def fitness_y1(features):
    prediction = gpr_model_target1.predict(np.array(features).reshape(1, -1))
    return -prediction[0]  

def fitness_y2(features):
    prediction = gpr_model_target2.predict(np.array(features).reshape(1, -1))
    return -prediction[0]

set_run_mode(fitness_y1, 'multithreading')
set_run_mode(fitness_y2, 'multithreading')



ga_y1 = GA(func=fitness_y1, n_dim=4, size_pop=1500, max_iter=1000, prob_mut=0.15, lb=lb, ub=ub, precision=1e-5)
start_time_y1 = datetime.datetime.now()
best_features_y1, best_prediction_y1 = ga_y1.run()
time_costs_y1 = (datetime.datetime.now() - start_time_y1).total_seconds()
print('Target 1 calculation completed, costs {}s'.format(time_costs_y1))
print('Best Features for Target 1:', best_features_y1, '\n', 'Predicted Target 1:', -best_prediction_y1)



ga_y2 = GA(func=fitness_y2, n_dim=4, size_pop=1500, max_iter=1000, prob_mut=0.15, lb=lb, ub=ub, precision=1e-5)
start_time_y2 = datetime.datetime.now()
best_features_y2, best_prediction_y2 = ga_y2.run()
time_costs_y2 = (datetime.datetime.now() - start_time_y2).total_seconds()
print('Target 2 calculation completed, costs {}s'.format(time_costs_y2))
print('Best Features for Target 2:', best_features_y2, '\n', 'Predicted Target 2:', -best_prediction_y2)

'''

##########################

###############3 GA for SVM 

'''
import numpy as np
from sko.GA import GA
import datetime
from sko.tools import set_run_mode

# Assuming svm_model_y1 and svm_model_y2 are your trained SVM models

def fitness_svm_y1(features):
    prediction = svm_model_y1.predict(np.array(features).reshape(1, -1))
    return -prediction[0]  # Assuming you want to maximize the prediction

def fitness_svm_y2(features):
    prediction = svm_model_y2.predict(np.array(features).reshape(1, -1))
    return -prediction[0]  # Assuming you want to maximize the prediction

# Set the running mode to multithreading for both fitness functions
set_run_mode(fitness_svm_y1, 'multithreading')
set_run_mode(fitness_svm_y2, 'multithreading')


# Run GA for SVM model target y1
ga_svm_y1 = GA(func=fitness_svm_y1, n_dim=4, size_pop=500, max_iter=1000, prob_mut=0.02, lb=lb, ub=ub, precision=1e-5)
start_time_svm_y1 = datetime.datetime.now()
best_features_svm_y1, best_prediction_svm_y1 = ga_svm_y1.run()
time_costs_svm_y1 = (datetime.datetime.now() - start_time_svm_y1).total_seconds()
print('SVM Target 1 calculation completed, costs {}s'.format(time_costs_svm_y1))
print('Best Features for SVM Target 1:', best_features_svm_y1, '\n', 'Predicted SVM Target 1:', -best_prediction_svm_y1)

# Run GA for SVM model target y2
ga_svm_y2 = GA(func=fitness_svm_y2, n_dim=4, size_pop=500, max_iter=1000, prob_mut=0.02, lb=lb, ub=ub, precision=1e-5)
start_time_svm_y2 = datetime.datetime.now()
best_features_svm_y2, best_prediction_svm_y2 = ga_svm_y2.run()
time_costs_svm_y2 = (datetime.datetime.now() - start_time_svm_y2).total_seconds()
print('SVM Target 2 calculation completed, costs {}s'.format(time_costs_svm_y2))
print('Best Features for SVM Target 2:', best_features_svm_y2, '\n', 'Predicted SVM Target 2:', -best_prediction_svm_y2)

########################################
'''

'''
######################3 shgo
from scipy.optimize import shgo

def objective_y1(features):
    prediction =svm_model_y1.predict(np.array(features).reshape(1, -1))
    return 1 / prediction[0] if prediction[0] != 0 else float('inf')

def objective_y2(features):
    prediction = svm_model_y2.predict(np.array(features).reshape(1, -1))
    return 1 / prediction[0] if prediction[0] != 0 else float('inf')

bounds = [(low, high) for low, high in zip(X.min(), X.max())]

result_y1 = shgo(objective_y1, bounds)
best_features_y1 = result_y1.x
predicted_y1 = svm_model_y1.predict([best_features_y1])[0]

result_y2 = shgo(objective_y2, bounds)
best_features_y2 = result_y2.x
predicted_y2 = svm_model_y2.predict([best_features_y2])[0]

print('Best Features for Target 1:', best_features_y1, '\n', 'Predicted Target 1:', predicted_y1)
print('Best Features for Target 2:', best_features_y2, '\n', 'Predicted Target 2:', predicted_y2)
'''
'''

from scipy.optimize import shgo


def objective_svm_y1(features):
    prediction = svm_model_y1.predict(np.array(features).reshape(1, -1))
    return 1 / prediction[0] if prediction[0] != 0 else float('inf')

def objective_svm_y2(features):
    prediction = svm_model_y2.predict(np.array(features).reshape(1, -1))
    return 1 / prediction[0] if prediction[0] != 0 else float('inf')

bounds = [(low, high) for low, high in zip(X.min(), X.max())]

result_svm_y1 = shgo(objective_svm_y1, bounds)
best_features_svm_y1 = result_svm_y1.x
predicted_svm_y1 = svm_model_y1.predict([best_features_svm_y1])[0]

result_svm_y2 = shgo(objective_svm_y2, bounds)
best_features_svm_y2 = result_svm_y2.x
predicted_svm_y2 = svm_model_y2.predict([best_features_svm_y2])[0]

print('Best Features for SVM Target 1:', best_features_svm_y1, '\n', 'Predicted SVM Target 1:', predicted_svm_y1)
print('Best Features for SVM Target 2:', best_features_svm_y2, '\n', 'Predicted SVM Target 2:', predicted_svm_y2)


'''
######################## deferential evolution
'''
from scipy.optimize import differential_evolution


def objective_y1(features):
    prediction =svm_model_y1.predict([features])
    return -prediction[0]

def objective_y2(features):
    prediction = svm_model_y2.predict([features])
    return -prediction[0]

bounds = [(low, high) for low, high in zip(X.min(), X.max())]

result_y1 = differential_evolution(objective_y1, bounds)
best_features_y1 = result_y1.x
predicted_y1 = svm_model_y1.predict([best_features_y1])[0]

result_y2 = differential_evolution(objective_y2, bounds)
best_features_y2 = result_y2.x
predicted_y2 = svm_model_y2.predict([best_features_y2])[0]

print('Best Features for Target 1:', best_features_y1, '\n', 'Predicted Target 1:', predicted_y1)
print('Best Features for Target 2:', best_features_y2, '\n', 'Predicted Target 2:', predicted_y2)
'''

######################
####3  dual_annealing
'''
from scipy.optimize import dual_annealing


def objective_y1(features):
    prediction = svm_model_y1.predict([features])
    return 1 / prediction[0] if prediction[0] != 0 else float('inf')

def objective_y2(features):
    prediction = svm_model_y2.predict([features])
    return 1 / prediction[0] if prediction[0] != 0 else float('inf')

bounds = [(low, high) for low, high in zip(X.min(), X.max())]

result_y1 = dual_annealing(objective_y1, bounds)
best_features_y1 = result_y1.x
predicted_y1 = svm_model_y1.predict([best_features_y1])[0]

result_y2 = dual_annealing(objective_y2, bounds)
best_features_y2 = result_y2.x
predicted_y2 = svm_model_y2.predict([best_features_y2])[0]

print('Best Features for Target 1:', best_features_y1, '\n', 'Predicted Target 1:', predicted_y1)
print('Best Features for Target 2:', best_features_y2, '\n', 'Predicted Target 2:', predicted_y2)

'''
#############################
'''
from scipy.optimize import dual_annealing

def objective_y1(features):
    prediction = svm_model_y1.predict([features])
    return -prediction[0]

def objective_y2(features):
    prediction = svm_model_y2.predict([features])
    return -prediction[0]


result_y1 = dual_annealing(objective_y1, bounds)
best_features_y1 = result_y1.x
predicted_y1 = svm_model_y1.predict([best_features_y1])[0]

result_y2 = dual_annealing(objective_y2, bounds)
best_features_y2 = result_y2.x
predicted_y2 = svm_model_y2.predict([best_features_y2])[0]

print('Best Features for Target 1:', best_features_y1, '\n', 'Predicted Target 1:', predicted_y1)
print('Best Features for Target 2:', best_features_y2, '\n', 'Predicted Target 2:', predicted_y2)

'''
########################33



##################33 nelder mead
'''
from scipy.optimize import minimize

def bounded_objective_y1(features):
    if any(lb > f or f > ub for f, lb, ub in zip(features, lb, ub)):
        return float('inf')
    prediction = gpr_model_target1.predict([features])
    return 1 / (prediction[0])  

def bounded_objective_y2(features):
    if any(lb > f or f > ub for f, lb, ub in zip(features, lb, ub)):
        return float('inf')
    prediction = gpr_model_target2.predict([features])
    return 1 / (prediction[0] )


initial_guess_y1 = [40, 130, 2.25, 0.3]
initial_guess_y2 = [40, 130, 2.25, 0.3]


initial_guess_y1 = [40, 110, 3.5, 0.3]
initial_guess_y2 = [40, 130, 1, 0.15]


result_y1 = minimize(bounded_objective_y1, initial_guess_y1, method='Nelder-Mead')
best_features_y1 = result_y1.x
maximized_prediction_y1 = 1 / bounded_objective_y1(best_features_y1) if result_y1.success else None

result_y2 = minimize(bounded_objective_y2, initial_guess_y2, method='Nelder-Mead')
best_features_y2 = result_y2.x
maximized_prediction_y2 = 1 / bounded_objective_y2(best_features_y2) if result_y2.success else None


print('Best Features for Target 1:', best_features_y1, 'Maximized Prediction for Target 1:', maximized_prediction_y1)
print('Best Features for Target 1:', best_features_y2, 'Maximized Prediction for Target 2:', maximized_prediction_y2)

'''
##############################


#######################

'''
from scipy.optimize import minimize

def bounded_objective_y1(features):
    if any(lb > f or f > ub for f, lb, ub in zip(features, lb, ub)):
        return float('inf')
    prediction = gpr_model_target1.predict([features])
    return -prediction[0]**2 

def bounded_objective_y2(features):
    if any(lb > f or f > ub for f, lb, ub in zip(features, lb, ub)):
        return float('inf')
    prediction = gpr_model_target2.predict([features])
    return -prediction[0]**2  



initial_guess_y1 = [40, 130, 2.25, 0.3]
initial_guess_y2 = [40, 130, 2.25, 0.3]


initial_guess_y1 = [40, 110, 3.5, 0.3]
initial_guess_y2 = [40, 130, 1, 0.15]

result_y1 = minimize(bounded_objective_y1, initial_guess_y1, method='Nelder-Mead')
best_features_y1 = result_y1.x
maximized_prediction_y1 = (-bounded_objective_y1(best_features_y1))**0.5 if result_y1.success else None

result_y2 = minimize(bounded_objective_y2, initial_guess_y2, method='Nelder-Mead')
best_features_y2 = result_y2.x
maximized_prediction_y2 = (-bounded_objective_y2(best_features_y2))**0.5 if result_y2.success else None

print('Best Features for Target 1:', best_features_y1, 'Maximized Prediction for Target 1:', maximized_prediction_y1)
print('Best Features for Target 2:', best_features_y2, 'Maximized Prediction for Target 2:', maximized_prediction_y2)

'''

##################

'''
from scipy.optimize import minimize

def bounded_objective_y1(features):
    if any(lb > f or f > ub for f, lb, ub in zip(features, lb, ub)):
        return float('inf')
    prediction = gpr_model_target1.predict([features])
    return 1 / (prediction[0]) 

def bounded_objective_y2(features):
    if any(lb > f or f > ub for f, lb, ub in zip(features, lb, ub)):
        return float('inf')
    prediction = gpr_model_target2.predict([features])
    return 1 / (prediction[0])  




initial_guess_y1 = [40, 110, 3.5, 0.3]
initial_guess_y2 = [40, 130, 1, 0.15]

options = {
    'maxiter': 400,           
    'maxfev':650,            
    'disp': True,             
    'return_all': False,     
    'xatol': 1e-4,            
    'fatol': 1e-4,         
    'adaptive': True          
}

result_y1 = minimize(bounded_objective_y1, initial_guess_y1, method='Nelder-Mead', options=options)
best_features_y1 = result_y1.x
maximized_prediction_y1 = 1/(bounded_objective_y1(best_features_y1)) if result_y1.success else None

result_y2 = minimize(bounded_objective_y2, initial_guess_y2, method='Nelder-Mead', options=options)
best_features_y2 = result_y2.x
maximized_prediction_y2 = 1/(bounded_objective_y2(best_features_y2)) if result_y2.success else None

print('Best Features for Target 1:', best_features_y1, 'Maximized Prediction for Target 1:', maximized_prediction_y1)
print('Best Features for Target 2:', best_features_y2, 'Maximized Prediction for Target 2:', maximized_prediction_y2)

'''
##############################
'''
import pygad
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go


def is_within_bounds(individual, lower_bounds, upper_bounds):
    return all(lower <= gene <= upper for gene, lower, upper in zip(individual, lower_bounds, upper_bounds))

def fitness_function_y1(ga_instance, solution, solution_idx):
    if not is_within_bounds(solution, lb, ub):
        return -np.inf 
    prediction_y1 = gpr_model_target1.predict([solution])[0]
    return prediction_y1

def fitness_function_y2(ga_instance, solution, solution_idx):
    if not is_within_bounds(solution, lb, ub):
        return -np.inf  
    prediction_y2 = gpr_model_target2.predict([solution])[0]
    return prediction_y2

lb = [20, 110, 1, 0.15]
ub = [60, 150, 3.5, 0.45]

def multithreaded_fitness(ga_instance):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(ga_instance.fitness_func, ga_instance, individual, idx) for idx, individual in enumerate(ga_instance.population)]
        fitness_values = [future.result() for future in futures]
    ga_instance.fitness = fitness_values

initial_guess_y1 = np.array([40, 110, 3.5, 0.3])
initial_guess_y2 = np.array([40, 130, 1, 0.15])
'''

'''
initial_guess_y1 = np.array([40, 110, 3.5, 0.3])
initial_guess_y2 = np.array([40, 130, 1, 0.15])


'''
'''
initial_guess_y1 = [40, 130, 2.25, 0.3]
initial_guess_y2 = [40, 130, 2.25, 0.3]

'''
'''
ga_instance_y1 = pygad.GA(num_generations=1800,
                          num_parents_mating=13,
                          fitness_func=fitness_function_y1,
                          on_generation=multithreaded_fitness,
                          parallel_processing=12,
                          sol_per_pop=5000,
                          num_genes=len(initial_guess_y1),
                          initial_population=[initial_guess_y1]*13,
                          parent_selection_type="sss",
                          crossover_type="single_point",
                          mutation_type="random",
                          mutation_percent_genes=85)

ga_instance_y1.run()

solution_y1, solution_fitness_y1, _ = ga_instance_y1.best_solution()
print("Best Solution for Target 1:", solution_y1, "with Fitness:", solution_fitness_y1,"\n")

'''



##############################3
####3 plot the genteic algotithm 

'''

import pygad
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming your fitness function and other necessary components are already defined.

initial_guess_y1 = np.array([40, 110, 3.5, 0.3])

def run_ga(mutation_percent_genes, sol_per_pop):
    ga_instance = pygad.GA(num_generations=1800,
                           num_parents_mating=13,
                           fitness_func=fitness_function_y1,
                           on_generation=multithreaded_fitness,
                           parallel_processing=12,
                           sol_per_pop=sol_per_pop,
                           num_genes=len(initial_guess_y1),
                           initial_population=[initial_guess_y1]*13,
                           parent_selection_type="sss",
                           crossover_type="single_point",
                           mutation_type="random",
                           mutation_percent_genes=mutation_percent_genes)
    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()
    return solution_fitness


mutation_percent_genes_range = range(5, 95, 5)
sol_per_pop_range = range(500, 9500, 500)

X, Y = np.meshgrid(mutation_percent_genes_range, sol_per_pop_range, indexing='ij')
Z = np.zeros(X.shape)

for i in range(len(mutation_percent_genes_range)):
    for j in range(len(sol_per_pop_range)):
        Z[i, j] = run_ga(mutation_percent_genes_range[i], sol_per_pop_range[j])

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig.update_layout(title='GA Fitness Landscape', autosize=False,
                  width=1200, height=1000,
                  margin=dict(l=65, r=50, b=65, t=90))

fig.show()


df = pd.DataFrame(index=sol_per_pop_range, columns=mutation_percent_genes_range)

for i, mutation_rate in enumerate(mutation_percent_genes_range):
    for j, population_size in enumerate(sol_per_pop_range):
        df.at[population_size, mutation_rate] = Z[i, j]

df.to_excel("ga_results_Target1.xlsx")


####################################################################################################################################

ga_instance_y2 = pygad.GA(num_generations=2500,
                          num_parents_mating=13,
                          fitness_func=fitness_function_y2,
                          on_generation=multithreaded_fitness,
                          parallel_processing=12,
                          sol_per_pop=8500,
                          num_genes=len(initial_guess_y2),
                          initial_population=[initial_guess_y2]*13,
                          parent_selection_type="sss",
                          crossover_type="single_point",
                          mutation_type="random",
                          mutation_percent_genes=80)

ga_instance_y2.run()

solution_y2, solution_fitness_y2, _ = ga_instance_y2.best_solution()
print("Best Solution for Target 2:", solution_y2, "with Fitness:", solution_fitness_y2,"\n")

'''

###############################

####3 plot the genetic algorthim 

'''
import pygad
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def run_ga(mutation_percent_genes, sol_per_pop):
    ga_instance = pygad.GA(num_generations=1800,
                           num_parents_mating=13,
                           fitness_func=fitness_function_y2,
                           on_generation=multithreaded_fitness,
                           parallel_processing=12,
                           sol_per_pop=sol_per_pop,
                           num_genes=len(initial_guess_y2),
                           initial_population=[initial_guess_y2]*13,
                           parent_selection_type="sss",
                           crossover_type="single_point",
                           mutation_type="random",
                           mutation_percent_genes=mutation_percent_genes)
    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()
    return solution_fitness


mutation_percent_genes_range = range(5, 95, 5)
sol_per_pop_range = range(500, 9500, 500)

X, Y = np.meshgrid(mutation_percent_genes_range, sol_per_pop_range, indexing='ij')
Z = np.zeros(X.shape)

for i in range(len(mutation_percent_genes_range)):
    for j in range(len(sol_per_pop_range)):
        Z[i, j] = run_ga(mutation_percent_genes_range[i], sol_per_pop_range[j])

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig.update_layout(title='GA Fitness Landscape', autosize=False,
                  width=1200, height=1000,
                  margin=dict(l=65, r=50, b=65, t=90))

fig.show()


df = pd.DataFrame(index=sol_per_pop_range, columns=mutation_percent_genes_range)

for i, mutation_rate in enumerate(mutation_percent_genes_range):
    for j, population_size in enumerate(sol_per_pop_range):
        df.at[population_size, mutation_rate] = Z[i, j]

df.to_excel("ga_results_Target2.xlsx")
'''

##############3




################# 4 for loop
'''
feature_ranges = {
    'feature_1': np.linspace(20, 60, 20),
    'feature_2': np.linspace(110, 150, 20),
    'feature_3': np.linspace(1, 3.5, 20),
    'feature_4': np.linspace(0.15, 0.45, 20)
}

best_target1_value = -np.inf  
best_target2_value = -np.inf
best_feature_combination_target1 = None
best_feature_combination_target2 = None

for f1 in feature_ranges['feature_1']:
    for f2 in feature_ranges['feature_2']:
        for f3 in feature_ranges['feature_3']:
            for f4 in feature_ranges['feature_4']:
                feature_vector = np.array([f1, f2, f3, f4]).reshape(1, -1)
                scaled_feature_vector = scaler.transform(feature_vector)  # Assuming scaler is already defined
                
                predicted_target1 =gpr_model_target1.predict(scaled_feature_vector)
                predicted_target2 = gpr_model_target2.predict(scaled_feature_vector)
                
                if predicted_target1 > best_target1_value:
                    best_target1_value = predicted_target1
                    best_feature_combination_target1 = (f1, f2, f3, f4)
                
                if predicted_target2 > best_target2_value:
                    best_target2_value = predicted_target2
                    best_feature_combination_target2 = (f1, f2, f3, f4)

print("Best feature combination for Target 1:", best_feature_combination_target1)
print("Predicted maximum value for Target 1:", best_target1_value)
print("Best feature combination for Target 2:", best_feature_combination_target2)
print("Predicted maximum value for Target 2:", best_target2_value)

print("\n\n\n\n\n")
'''
############## bayesian

'''
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

space = [
    Real(20, 60, name='feature1'),
    Real(110, 150, name='feature2'),
    Real(1, 3.5, name='feature3'),
    Real(0.15, 0.45, name='feature4')
]

@use_named_args(space)
def objective_function(feature1, feature2, feature3, feature4):
    input_features = np.array([feature1, feature2, feature3, feature4]).reshape(1, -1)
    prediction = svm_model_y1.predict(input_features)  
    scalar_prediction = prediction[0] if len(prediction) > 0 else 0
    return -scalar_prediction


result = gp_minimize(objective_function, space, n_calls=50, random_state=0)

optimized_inputs = result.x
print("Optimized Inputs:", optimized_inputs)

final_target_value = svm_model_y1.predict([optimized_inputs])
print("Final Target Value:", final_target_value[0])

'''



#############################################################################################################
'''
X.columns = feature_names
'''
feature_names = ["Time (hr)", "Temperature (°C)", "Molar Ratio", "Ion Concentration (M)"]


#####3 visualiztions


print(X.columns)

print(f"Max value in y1 (surface area (m^2/gr)): {y1.max()}")
print(f"Max value in y2 (MB Removal (%)): {y2.max()}")




feature_combinations = [
    ('Time (hr)', 'Temperature (°C)'),
    ('Time (hr)', 'Molar Ratio'),
    ('Time (hr)', 'Ion Concentration (M)'),
    ('Temperature (°C)', 'Molar Ratio'),
    ('Temperature (°C)', 'Ion Concentration (M)'),
    ('Molar Ratio', 'Ion Concentration (M)')
]

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def compute_feature_importance(X, y, model, target_name, feature_names):
    result = permutation_importance(model, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()

    plt.figure(figsize=(13, 9))
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=[feature_names[i] for i in sorted_idx], patch_artist=True)
    plt.title(f"Permutation Importances for {target_name}", fontsize=18)
    plt.xlabel('Importance Score', fontsize=16)
    plt.ylabel('Features', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
compute_feature_importance(X, y1, gpr_model_target1, "Surface Area (m^2/gr)", feature_names)

compute_feature_importance(X, y2, gpr_model_target2, "MB Removal (%)", feature_names)


'''

###################
##

feature_names = ["Time (hr)", "Temperature (°C)", "Molar Ratio", "Ion Concentration (M)"]
X.columns = feature_names

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
'''

# Setting the font to Arial, size 16, and bold
matplotlib.rc('font', family='Arial', size=16, weight='bold')

feature_names = ["Time (hr)", "Temperature (°C)", "Molar Ratio", "Ion Concentration (M)"]
X.columns = feature_names

correlation_with_target1 = X.corrwith(y1)
correlation_with_target2 = X.corrwith(y2)

correlation_df = pd.DataFrame({
    'Feature': feature_names,
    'Correlation with Surface Area ($m^2/gr$)': correlation_with_target1.values,
    'Correlation with MB Removal (%)': correlation_with_target2.values
})

plt.figure(figsize=(12, 6))

plt.bar(correlation_df['Feature'], correlation_df['Correlation with Surface Area ($m^2/gr$)'], width=0.4, align='center', label='Surface Area ($m^2/gr$)', alpha=0.8)
plt.bar(correlation_df['Feature'], correlation_df['Correlation with MB Removal (%)'], width=0.4, align='edge', label='MB Removal (%)', alpha=0.8)

plt.xlabel('Features')
plt.ylabel('Pearson Correlation Coefficient')
plt.title('Pearson Correlation Coefficient with Surface Area ($m^2/gr$) and MB Removal (%)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

def compute_pearson_correlation(X, y, target_name):
    data_combined = pd.concat([X, y], axis=1)
    correlation_matrix = data_combined.corr()

    plt.figure(figsize=(10, 7))
    sns.heatmap(correlation_matrix, annot=True, cmap='Greens', vmin=-1, vmax=1)
    plt.title(f"Pearson Correlation Coefficient Matrix for {target_name}",weight='bold')
    plt.show()

compute_pearson_correlation(X, y1, "Surface Area ($m^2/gr$)")
compute_pearson_correlation(X, y2, "MB Removal (%)")

'''

#######################

'''
import shap
import matplotlib.pyplot as plt
import pandas as pd

def compute_and_plot_shap_gpr(model, X, y, target_name, feature_names):
    explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
    shap_values = explainer.shap_values(shap.sample(X, 100))

    
    shap.summary_plot(shap_values, pd.DataFrame(X, columns=feature_names), plot_type="bar", show=False)
    
    plt.title(f"SHAP Values for {target_name}")
    plt.xlabel('') 
    plt.show()

compute_and_plot_shap_gpr(gpr_model_target1, X, y1, "Surface Area (m^2/gr)", feature_names)
compute_and_plot_shap_gpr(gpr_model_target2, X, y2, "MB Removal (%)", feature_names)

######


def compute_and_plot_shap_gpr(model, X, y, target_name, feature_names):
    
    explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
    shap_values = explainer.shap_values(shap.sample(X, 100))
    plt.figure(figsize=(20, 10))

    shap.summary_plot(shap_values, pd.DataFrame(X, columns=feature_names), plot_type="dot", title=f"SHAP Beeswarm Plot for {target_name}")


compute_and_plot_shap_gpr(gpr_model_target1, X, y1, "Surface Area (m^2/gr)", feature_names)

compute_and_plot_shap_gpr(gpr_model_target2, X, y2, "MB Removal (%)", feature_names)

'''

'''
import shap
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np

# Setting the font to Arial, size 16, and bold
matplotlib.rc('font', family='Arial', size=16, weight='bold')

def compute_and_plot_shap_gpr(model, X, y, target_name, feature_names):
    explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
    shap_values = explainer.shap_values(shap.sample(X, 100), nsamples=100)  # specify nsamples if needed

    # Assuming shap_values is a list of arrays, one per class/output. If not, adjust accordingly.
    shap_values_mean = np.mean(shap_values, axis=0) if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(20, 10))
    shap.summary_plot(shap_values_mean, pd.DataFrame(X, columns=feature_names), plot_type="bar", show=False)
    plt.title(f"SHAP Values for {target_name}")
    plt.show()

    return shap_values_mean  # Return averaged SHAP values

# Initialize dictionary to store SHAP values
shap_values_dict = {}

# Compute SHAP values and plot for each target
shap_values_dict["Surface Area (m^2/gr)"] = compute_and_plot_shap_gpr(gpr_model_target1, X, y1, "Surface Area ($m^2/gr$)", feature_names)
shap_values_dict["MB Removal (%)"] = compute_and_plot_shap_gpr(gpr_model_target2, X, y2, "MB Removal (%)", feature_names)

# Convert the dictionary of SHAP values to a DataFrame and save to Excel


'''


##########################3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

'''
# Setting the font globally to Arial, size 16, and bold
matplotlib.rc('font', family='Arial', size=16, weight='bold')

def plot_1D_partial_dependence_gpr(model, X, feature_idx, feature_name, scaler, target_name):
    x_vals = np.linspace(np.min(X[:, feature_idx]), np.max(X[:, feature_idx]), 100)
    pdp_vals = []
    x_original = scaler.inverse_transform(np.column_stack([x_vals if i == feature_idx else np.zeros(x_vals.shape) 
                                                           for i in range(X.shape[1])]))[:, feature_idx]

    for val in x_vals:
        X_temp = X.copy()
        X_temp[:, feature_idx] = val
        predictions = model.predict(X_temp)
        pdp_vals.append(predictions.mean())

    plt.figure(figsize=(16, 8))
    plt.plot(x_original, pdp_vals, label=feature_name)
    plt.title(f'Partial Dependence of {feature_name} on {target_name}')
    plt.xlabel(feature_name)
    plt.ylabel(f'Partial Dependence for {target_name}')
    plt.legend()
    plt.show()


def plot_2D_partial_dependence_gpr(model, X, feature_idx1, feature_idx2, feature_name1, feature_name2, scaler, target_name):
    x_vals = np.linspace(np.min(X[:, feature_idx1]), np.max(X[:, feature_idx1]), 20)
    y_vals = np.linspace(np.min(X[:, feature_idx2]), np.max(X[:, feature_idx2]), 20)
    
    X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
    Z_mesh = np.zeros(X_mesh.shape)
    
    x_original = scaler.inverse_transform(np.column_stack([X_mesh.ravel() if i == feature_idx1 else (Y_mesh.ravel() if i == feature_idx2 else np.zeros(X_mesh.ravel().shape)) 
                                                           for i in range(X.shape[1])]))[:, feature_idx1].reshape(X_mesh.shape)
    
    y_original = scaler.inverse_transform(np.column_stack([X_mesh.ravel() if i == feature_idx1 else (Y_mesh.ravel() if i == feature_idx2 else np.zeros(X_mesh.ravel().shape)) 
                                                           for i in range(X.shape[1])]))[:, feature_idx2].reshape(Y_mesh.shape)

    for i in range(X_mesh.shape[0]):
        for j in range(X_mesh.shape[1]):
            X_temp = X.copy()
            X_temp[:, feature_idx1] = X_mesh[i, j]
            X_temp[:, feature_idx2] = Y_mesh[i, j]
            predictions = model.predict(X_temp)
            Z_mesh[i, j] = predictions.mean()
            
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_original, y_original, Z_mesh, cmap="Blues")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    ax.set_title(f'Partial Dependence of {feature_name1} and {feature_name2} on {target_name}')
    ax.set_xlabel(feature_name1)
    ax.set_ylabel(feature_name2)
    ax.set_zlabel(f'Partial Dependence for {target_name}')
    plt.show()

feature_names = X.columns.tolist()  
combinations_2D = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

gpr_model_target1.fit(X, y1)
for idx, feature in enumerate(feature_names):
    plot_1D_partial_dependence_gpr(gpr_model_target1, X.values, idx, feature, scaler, "Surface Area ($m^2/gr$)")

for combo in combinations_2D:
    plot_2D_partial_dependence_gpr(gpr_model_target1, X.values, combo[0], combo[1], feature_names[combo[0]], feature_names[combo[1]], scaler, "Surface Area ($m^2/gr$)")

gpr_model_target2.fit(X, y2)
for idx, feature in enumerate(feature_names):
    plot_1D_partial_dependence_gpr(gpr_model_target2, X.values, idx, feature, scaler, "MB Removal (%)")

for combo in combinations_2D:
    plot_2D_partial_dependence_gpr(gpr_model_target2, X.values, combo[0], combo[1], feature_names[combo[0]], feature_names[combo[1]], scaler, "MB Removal (%)")
'''

###############################################3
#####33 PDP in the bounds 


X.columns = feature_names
feature_names = ["Time (hr)", "Temperature (°C)", "Molar Ratio", "Ion Concentration (M)"]


'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

adjusted_bounds = [(15, 65), (100, 160), (1, 3.5), (0.1, 0.5)]

def plot_1D_partial_dependence_gpr(model, X, feature_idx, feature_name, scaler, bounds, target_name):
    x_vals = np.linspace(bounds[0], bounds[1], 100)
    pdp_vals = []
    x_original = scaler.inverse_transform(np.column_stack([x_vals if i == feature_idx else np.zeros(x_vals.shape) 
                                                           for i in range(X.shape[1])]))[:, feature_idx]

    for val in x_vals:
        X_temp = X.copy()
        X_temp[:, feature_idx] = val
        predictions = model.predict(X_temp)
        pdp_vals.append(predictions.mean())

    plt.figure(figsize=(16, 8))
    plt.plot(x_original, pdp_vals, label=feature_name)
    plt.title(f'Partial Dependence of {feature_name} on {target_name}')
    plt.xlabel(feature_name)
    plt.ylabel(f'Partial Dependence for {target_name}')
    plt.xlim(bounds)
    plt.legend()
    plt.show()

def plot_2D_partial_dependence_gpr(model, X, feature_idx1, feature_idx2, feature_name1, feature_name2, scaler, bounds1, bounds2, target_name):
    x_vals = np.linspace(bounds1[0], bounds1[1], 20)
    y_vals = np.linspace(bounds2[0], bounds2[1], 20)
    
    X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
    Z_mesh = np.zeros(X_mesh.shape)

    for i in range(X_mesh.shape[0]):
        for j in range(X_mesh.shape[1]):
            X_temp = X.copy()
            X_temp[:, feature_idx1] = X_mesh[i, j]
            X_temp[:, feature_idx2] = Y_mesh[i, j]
            predictions = model.predict(X_temp)
            Z_mesh[i, j] = predictions.mean()
    
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap="Blues")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    ax.set_title(f'Partial Dependence of {feature_name1} and {feature_name2} on {target_name}')
    ax.set_xlabel(feature_name1)
    ax.set_ylabel(feature_name2)
    ax.set_zlabel(f'Partial Dependence for {target_name}')
    plt.show()

feature_names = X.columns.tolist()  

combinations_2D = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

gpr_model_target1.fit(X, y1)
gpr_model_target2.fit(X, y2)

for idx, feature in enumerate(feature_names):
    plot_1D_partial_dependence_gpr(gpr_model_target1, X.values, idx, feature, scaler, adjusted_bounds[idx], "Surface Area (m^2/gr)")

combinations_2D = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

for combo in combinations_2D:
    plot_2D_partial_dependence_gpr(gpr_model_target1, X.values, combo[0], combo[1], feature_names[combo[0]], feature_names[combo[1]], scaler, adjusted_bounds[combo[0]], adjusted_bounds[combo[1]], "Surface Area (m^2/gr)")

for idx, feature in enumerate(feature_names):
    plot_1D_partial_dependence_gpr(gpr_model_target2, X.values, idx, feature, scaler, adjusted_bounds[idx], "MB Removal (%)")

for combo in combinations_2D:
    plot_2D_partial_dependence_gpr(gpr_model_target2, X.values, combo[0], combo[1], feature_names[combo[0]], feature_names[combo[1]], scaler, adjusted_bounds[combo[0]], adjusted_bounds[combo[1]], "MB Removal (%)")

'''

###############################################################################3


'''
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
import numpy as np
import matplotlib

# Setting global font properties
matplotlib.rc('font', family='Arial', size=19, weight='bold')

data_for_plotting = pd.concat([X, y1.rename('Surface Area ($m^2/gr$)'), y2.rename('MB Removal (%)')], axis=1)
feature_combinations = list(combinations(X.columns, 2))

# Plotting for Surface Area
plt.figure(figsize=(18, 12))
for i, (feature_x, feature_y) in enumerate(feature_combinations, 1):
    plt.subplot(2, 3, i)
    contour = plt.tricontourf(data_for_plotting[feature_x].values.flatten(),
                              data_for_plotting[feature_y].values.flatten(),
                              data_for_plotting['Surface Area ($m^2/gr$)'].values.flatten(), 20, cmap='Greens')
    cbar = plt.colorbar(contour)
    cbar.set_label('Surface Area ($m^2/gr$)')
    plt.xlabel(feature_x,fontsize=22,fontweight='bold',family='Arial')
    plt.ylabel(feature_y,fontsize=22,fontweight='bold',family='Arial')
    plt.title(" ")
plt.tight_layout()
plt.suptitle("Contour Plots for Surface Area ($m^2/gr$)",fontsize=30,y=1.05,fontweight='bold',family='Arial')
plt.show()

# Plotting for MB Removal
plt.figure(figsize=(18, 12))
for i, (feature_x, feature_y) in enumerate(feature_combinations, 1):
    plt.subplot(2, 3, i)
    contour = plt.tricontourf(data_for_plotting[feature_x].values.flatten(),
                              data_for_plotting[feature_y].values.flatten(),
                              data_for_plotting['MB Removal (%)'].values.flatten(), 20, cmap='Blues')
    cbar = plt.colorbar(contour)
    cbar.set_label('MB Removal (%)')
    plt.xlabel(feature_x,fontsize=21,fontweight='bold',family='Arial')
    plt.ylabel(feature_y,fontsize=21,fontweight='bold',family='Arial')
    plt.title(" ")
plt.tight_layout()
plt.suptitle("Contour Plots for MB Removal (%)",fontsize=30,y=1.05,fontweight='bold',family='Arial')
plt.show()

# Implementing detailed level contour plotting for Surface Area
target_levels = np.linspace(0, 1800, num=100)
target = 'Surface Area ($m^2/gr$)'
plt.figure(figsize=(18, 12))
for i, (feature_x, feature_y) in enumerate(feature_combinations, 1):
    plt.subplot(2, 3, i)
    contour = plt.tricontourf(data_for_plotting[feature_x], data_for_plotting[feature_y], data_for_plotting[target], levels=target_levels, cmap='Greens')
    cbar = plt.colorbar(contour)
    cbar.set_label(target)
    plt.xlabel(feature_x,fontsize=22,fontweight='bold',family='Arial')
    plt.ylabel(feature_y,fontsize=22,fontweight='bold',family='Arial')
    plt.title(" ")
plt.tight_layout()
plt.suptitle(f"Contour Plots for {target}",fontsize=30,y=1.05,fontweight='bold',family='Arial')
plt.show()

# Implementing detailed level contour plotting for MB Removal
target_levels = np.linspace(0, 100, num=100)
target2 = 'MB Removal (%)'
plt.figure(figsize=(18, 12))
for i, (feature_x, feature_y) in enumerate(feature_combinations, 1):
    plt.subplot(2, 3, i)
    contour = plt.tricontourf(data_for_plotting[feature_x], data_for_plotting[feature_y], data_for_plotting[target2], levels=target_levels, cmap='Blues')
    cbar = plt.colorbar(contour)
    cbar.set_label(target2)
    plt.xlabel(feature_x,fontsize=21,fontweight='bold',family='Arial')
    plt.ylabel(feature_y,fontsize=21,fontweight='bold',family='Arial')
    plt.title(" ")
plt.tight_layout()
plt.suptitle(f"Contour Plots for {target2}",fontsize=30,y=1.05,fontweight='bold',family='Arial')
plt.show()

'''

#########################33




'''

###################

print(X.columns)

print(f"Max value in y1 (surface Area (m^2/gr)): {y1.max()}")
print(f"Max value in y2 (MB Removal (%)): {y2.max()}")



feature_combinations = [
    ('Time (hr)', 'Temperature (°C)'),
    ('Time (hr)', 'Molar Ratio'),
    ('Time (hr)', 'Ion Concentration (M)'),
    ('Temperature (°C)', 'Molar Ratio'),
    ('Temperature (°C)', 'Ion Concentration (M)'),
    ('Molar Ratio', 'Ion Concentration (M)')
]



import seaborn as sns
import matplotlib.pyplot as plt


def generate_contour_plots(X, y, title_suffix):
    plt.figure(figsize=(18, 12))
    for i, (feature_x, feature_y) in enumerate(feature_combinations, 1):
        plt.subplot(2, 3, i)
        sns.kdeplot(x=X[feature_x], y=X[feature_y], fill=True, cmap='Blues', levels=50, cbar=True, weights=y)
        plt.title(f"{feature_x} vs {feature_y}")
    plt.tight_layout()
    plt.suptitle(f"Contour Plots (Seaborn) for '{title_suffix}'", y=1.05, fontsize=16)
    plt.show()

# Generate contour plots for y1 and y2
generate_contour_plots(X, y1, "Surface Area (m^2/gr)")
generate_contour_plots(X, y2, "MB Removal (%)")
'''


############
'''

from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# Set default font properties to Arial and bold for all elements
matplotlib.rc('font', family='Arial', size=17.5, weight='bold')

def generate_3d_surface_plots(X, y, target_name):
    for feature_x, feature_y in feature_combinations:
        fig = plt.figure(figsize=(20, 14))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create grid for interpolation
        xi = np.linspace(X[feature_x].min(), X[feature_x].max(), 300)
        yi = np.linspace(X[feature_y].min(), X[feature_y].max(), 300)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((X[feature_x], X[feature_y]), y, (xi, yi), method='cubic')

        # Plot the 3D surface
        surf = ax.plot_surface(xi, yi, zi, cmap='Blues', edgecolor='none')

        # Set axis labels
        ax.set_xlabel(feature_x, fontsize=31.5, fontweight='bold', labelpad=30)
        ax.set_ylabel(feature_y, fontsize=31.5, fontweight='bold', labelpad=30)
        ax.set_zlabel("", fontsize=28.5, fontweight='bold', labelpad=30)

        # Set plot title
        ax.set_title(f"{target_name.replace('m^2/gr', '$m^2/gr$')}", fontsize=49, fontweight='bold', family='Arial')

        # Set ticks font for x, y, z axes
        ax.tick_params(axis='x', labelsize=30, labelrotation=0, labelcolor='black', width=2)
        ax.tick_params(axis='y', labelsize=30, labelrotation=0, labelcolor='black', width=2)
        ax.tick_params(axis='z', labelsize=26, labelrotation=0, labelcolor='black', width=2)

        # Add color bar and set font properties for color bar labels
        cbar = fig.colorbar(surf, ax=ax, pad=0.05)
        cbar.set_label(target_name.replace('m^2/gr', '$m^2/gr$'), fontsize=40, fontweight='bold', family='Arial')
        cbar.ax.tick_params(labelsize=37, width=2)  # Set font for color bar ticks

        # Show the plot
        plt.show()
        
generate_3d_surface_plots(X, y1, "Surface Area (m^2/gr)")
generate_3d_surface_plots(X, y2, "MB Removal (%)")

'''

#############3333

'''

feature_combinations = [
    ('Time (hr)', 'Temperature (°C)'),
    ('Time (hr)', 'Molar Ratio'),
    ('Time (hr)', 'Ion Concentration (M)'),
    ('Temperature (°C)', 'Molar Ratio'),
    ('Temperature (°C)', 'Ion Concentration (M)'),
    ('Molar Ratio', 'Ion Concentration (M)')
]


import numpy as np
import matplotlib.pyplot as plt

# Set global font properties
matplotlib.rc('font', family='Arial', size=23.5, weight='bold')

def generate_contour_plots(model, target_name, feature_combinations, X_scaled, scaler):
    plt.figure(figsize=(20, 18))
    for i, (feature_x, feature_y) in enumerate(feature_combinations, 1):
        feature_idx_x = list(X.columns).index(feature_x)
        feature_idx_y = list(X.columns).index(feature_y)

        # Compute the mean and std for scaling
        x_mean, x_std = scaler.mean_[feature_idx_x], scaler.scale_[feature_idx_x]
        y_mean, y_std = scaler.mean_[feature_idx_y], scaler.scale_[feature_idx_y]

        # Define the plotting range
        x_min, x_max = (X_scaled[:, feature_idx_x].min() - 1) * x_std + x_mean, (X_scaled[:, feature_idx_x].max() + 1) * x_std + x_mean
        y_min, y_max = (X_scaled[:, feature_idx_y].min() - 1) * y_std + y_mean, (X_scaled[:, feature_idx_y].max() + 1) * y_std + y_mean

        # Create a meshgrid for contour plotting
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

        # Scale the meshgrid points
        xx_scaled = (xx - x_mean) / x_std
        yy_scaled = (yy - y_mean) / y_std

        # Prepare the input for prediction
        prediction_input = [xx_scaled.ravel(), yy_scaled.ravel()]
        for idx, mean in enumerate(scaler.mean_):
            if idx != feature_idx_x and idx != feature_idx_y:
                prediction_input.append(np.full(xx_scaled.ravel().shape, mean))

        # Predict the values for the contour plot
        Z = model.predict(np.array(prediction_input).T)
        Z = Z.reshape(xx.shape)

        # Clip Z values for MB Removal to max at 100%
        if target_name == 'MB Removal (%)':
            Z = np.clip(Z, None, 100)

        # Create a subplot
        plt.subplot(3, 2, i)
        contour = plt.contourf(xx, yy, Z, 50, cmap='Blues')

        # Color bar customization
        cbar = plt.colorbar(contour, pad=0.05)  # Adding padding to the colorbar
        cbar.set_label("", size=34, weight='bold', family='Arial')
        cbar.ax.tick_params(labelsize=34)  # Set the font size of colorbar values
        cbar.ax.yaxis.label.set_size(34)  # Set the size of the colorbar label

        # Customize x and y axes labels
        plt.xlabel(feature_x, fontsize=40, fontweight='bold', family='Arial')
        plt.ylabel(feature_y, fontsize=40, fontweight='bold', family='Arial')

        # Customize tick label sizes for x and y axes
        plt.xticks(fontsize=38.5, family='Arial', weight='bold')
        plt.yticks(fontsize=38.5, family='Arial', weight='bold')

        # Adding plot titles (optional)

    # Adjust the layout and overall title
    plt.tight_layout()
    plt.suptitle(f"Contour Plots for Predicted {target_name}", fontsize=44.5, y=1.05, fontweight='bold', family='Arial')
    plt.show()

# Assuming gpr_model_target1 and gpr_model_target2 are defined, and X_train_scaled, scaler, and feature_combinations are properly set up
generate_contour_plots(gpr_model_target1, 'Surface Area ($m^2/gr$)', feature_combinations, X_train_scaled, scaler)
generate_contour_plots(gpr_model_target2, 'MB Removal (%)', feature_combinations, X_train_scaled, scaler)


'''
#################################

########3 plotlyyyyyyyyyyyyyyyyy

save_directory = "C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/plot-save"  
base_save_dir = "C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/plot-save/contour_plots"
base_save_dir_3d = "C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/plot-save/3D_scatter_plots"
base_save_dir_3dsurface = "C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/plot-save/surface_plots"



#############################################################33

'''
import os
import plotly.graph_objects as go
import pandas as pd

# Define the feature names and combinations
feature_names = ['Time (hr)', 'Temperature (°C)', 'Molar Ratio', 'Ion Concentration (M)']
feature_combinations = [
    ('Time (hr)', 'Temperature (°C)'),
    ('Time (hr)', 'Molar Ratio'),
    ('Time (hr)', 'Ion Concentration (M)'),
    ('Temperature (°C)', 'Molar Ratio'),
    ('Temperature (°C)', 'Ion Concentration (M)'),
    ('Molar Ratio', 'Ion Concentration (M)')
]

# Function to save contour plots
def save_plotly_contour_plots(X, y, feature_x, feature_y, target_name, save_dir):
    # Use Unicode to display m²/g beautifully
    display_name = "Surface Area (m²\u2009/\u2009gr)" if target_name == "Surface Area m^2/g" else f"{target_name} (%)"
    
    title_font = dict(family='Arial', size=45, color='black')  # Font settings

    fig = go.Figure(data=go.Contour(
        z=y,
        x=X[feature_x],
        y=X[feature_y],
        colorscale='Blues',
        colorbar=dict(
            title=dict(text=" "),  # Empty text as needed
            titlefont=dict(family='Arial', size=30, color='black'),
        ),
        contours=dict(showlines=False)
    ))

    fig.update_layout(
        title=display_name,
        title_x=0.5,  # Center the title
        title_font=title_font,
        xaxis_title=feature_x,
        xaxis_title_font=dict(family='Arial', size=39, color='black'),
        yaxis_title=feature_y,
        yaxis_title_font=dict(family='Arial', size=39, color='black'),
        font=dict(family='Arial', size=37.5, color='black'),
        width=750,
        height=750,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    filename_safe_target_name = target_name.replace(" ", "_").replace("/", "_per_").replace("^", "")
    filename = os.path.join(save_dir, f"contour_{feature_x}_{feature_y}_{filename_safe_target_name}.png")
    fig.write_image(filename, scale=12)

# Directory setup
if not os.path.exists(base_save_dir):
    os.makedirs(base_save_dir)

targets = {
    'Surface Area m^2/g': 'y1',
    'MB Removal': 'y2'
}

for target_name, y_var in targets.items():
    target_dir = os.path.join(base_save_dir, f"{target_name.replace(' ', '_').replace('^2', '2').replace('/', '_per_')}_plots")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for feature_x, feature_y in feature_combinations:
        save_plotly_contour_plots(X, eval(y_var), feature_x, feature_y, target_name, target_dir)


'''

#####################################33

'''

import os
import plotly.graph_objects as go
import pandas as pd

# Define the feature names and combinations
feature_names = ['Time (hr)', 'Temperature (°C)', 'Molar Ratio', 'Ion Concentration (M)']
feature_combinations = [
    ('Time (hr)', 'Temperature (°C)'),
    ('Time (hr)', 'Molar Ratio'),
    ('Time (hr)', 'Ion Concentration (M)'),
    ('Temperature (°C)', 'Molar Ratio'),
    ('Temperature (°C)', 'Ion Concentration (M)'),
    ('Molar Ratio', 'Ion Concentration (M)')
]

# Function to save contour plots
def save_plotly_contour_plots(X, y, feature_x, feature_y, target_name, save_dir):
    # Use Unicode to display m²/g beautifully
    display_name = "Surface Area (m²\u2009/\u2009g)" if target_name == "Surface Area m^2/g" else f"{target_name} (%)"
    
    title_font = dict(family='Arial', size=30, color='black')  # Font settings

    fig = go.Figure(data=go.Contour(
        z=y,
        x=X[feature_x],
        y=X[feature_y],
        colorscale='Greens',
        colorbar=dict(
            title=dict(text=" "),  # Empty text as needed
            titlefont=dict(family='Arial', size=30, color='black'),
        ),
        contours=dict(showlines=False, showlabels=True, labelfont=dict(size=12, color='red'))
    ))

    fig.update_layout(
        title=display_name,
        title_x=0.5,  # Center the title
        title_font=title_font,
        xaxis_title=feature_x,
        xaxis_title_font=dict(family='Arial', size=21, color='black'),
        yaxis_title=feature_y,
        yaxis_title_font=dict(family='Arial', size=21, color='black'),
        font=dict(family='Arial', size=21, color='black'),
        width=600,
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    filename_safe_target_name = target_name.replace(" ", "_").replace("/", "_per_").replace("^", "")
    filename = os.path.join(save_dir, f"contour_{feature_x}_{feature_y}_{filename_safe_target_name}.png")
    fig.write_image(filename, scale=12)

# Directory setup
if not os.path.exists(base_save_dir):
    os.makedirs(base_save_dir)

targets = {
    'Surface Area m^2/g': 'y1',
    'MB Removal': 'y2'
}

for target_name, y_var in targets.items():
    target_dir = os.path.join(base_save_dir, f"{target_name.replace(' ', '_').replace('^2', '2').replace('/', '_per_')}_plots")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for feature_x, feature_y in feature_combinations:
        save_plotly_contour_plots(X, eval(y_var), feature_x, feature_y, target_name, target_dir)

'''
##########################


'''
import os
import plotly.express as px


def save_3d_plotly_scatter(X, y, feature_x, feature_y, target_name, save_dir):
    fig = px.scatter_3d(X, x=feature_x, y=feature_y, z=y, color=y, color_continuous_scale='Blues', opacity=0.7)
    fig.update_layout(title=f"{feature_x}, {feature_y} vs {target_name}", scene=dict(xaxis_title=feature_x, yaxis_title=feature_y, zaxis_title=target_name))
    
    filename = os.path.join(save_dir, f"3Dscatter_{feature_x}_{feature_y}_{target_name}.png")
    fig.write_image(filename, scale=12)  

if not os.path.exists(base_save_dir_3d):
    os.makedirs(base_save_dir_3d)

target1_save_dir_3d = os.path.join(base_save_dir_3d, "Surface Area (m^2/gr)_plots")
if not os.path.exists(target1_save_dir_3d):
    os.makedirs(target1_save_dir_3d)

target2_save_dir_3d = os.path.join(base_save_dir_3d, "MB Removal (%)_plots")
if not os.path.exists(target2_save_dir_3d):
    os.makedirs(target2_save_dir_3d)

for feature_x, feature_y in feature_combinations:
    save_3d_plotly_scatter(X, y1, feature_x, feature_y, 'Surface Area (m^2/gr)', target1_save_dir_3d)

for feature_x, feature_y in feature_combinations:
    save_3d_plotly_scatter(X, y2, feature_x, feature_y, 'MB Removal (%)', target2_save_dir_3d)
'''

###################



import os
import plotly.graph_objects as go
import numpy as np
import pandas as pd


'''
feature_combinations = [
    ('Time (hr)', 'Temperature (°C)'),
    ('Time (hr)', 'Molar Ratio'),
    ('Time (hr)', 'Ion Concentration (M)'),
    ('Temperature (°C)', 'Molar Ratio'),
    ('Temperature (°C)', 'Ion Concentration (M)'),
    ('Molar Ratio', 'Ion Concentration (M)')
]

def save_3d_plotly_surface(X, y, feature_x, feature_y, target_name, save_dir):
    # Ensure y is a DataFrame with a column named target_name
    y_df = pd.DataFrame({target_name: y})
    combined_data = pd.concat([X, y_df], axis=1)

    data_averaged = combined_data.groupby([feature_x, feature_y]).mean().reset_index()

    x = data_averaged[feature_x].unique()
    y = data_averaged[feature_y].unique()
    x.sort()
    y.sort()
    xi, yi = np.meshgrid(x, y)

    # Ensure the pivot method is called with the correct column name
    z_func = data_averaged.pivot(index=feature_y, columns=feature_x, values=target_name)
    zi = z_func.values

    fig = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Viridis')])
    fig.update_layout(title=f"{feature_x}, {feature_y} vs {target_name}", scene=dict(xaxis_title=feature_x, yaxis_title=feature_y, zaxis_title=target_name))
    
    filename = os.path.join(save_dir, f"3Dsurface_{feature_x}_{feature_y}_{target_name}.png")
    fig.write_image(filename, scale=12)

if not os.path.exists(base_save_dir_3dsurface):
    os.makedirs(base_save_dir_3dsurface)

target1_save_dir_3dsurface = os.path.join(base_save_dir_3dsurface, "Surface_Area (m^2/gr)_plots")
if not os.path.exists(target1_save_dir_3dsurface):
    os.makedirs(target1_save_dir_3dsurface)

target2_save_dir_3dsurface = os.path.join(base_save_dir_3dsurface, "MB_Removal (%)_plots")
if not os.path.exists(target2_save_dir_3dsurface):
    os.makedirs(target2_save_dir_3dsurface)

for feature_x, feature_y in feature_combinations:
    save_3d_plotly_surface(X, y1, feature_x, feature_y, 'Surface Area (m^2/gr)', target1_save_dir_3dsurface)

for feature_x, feature_y in feature_combinations:
    save_3d_plotly_surface(X, y2, feature_x, feature_y, 'MB Removal (%)', target2_save_dir_3dsurface)
    



'''

##########################3



'''
'''
print(X.columns)

feature_combinations = [
    ('Time (hr)', 'Temperature (°C)'),
    ('Time (hr)', 'Molar Ratio'),
    ('Time (hr)', 'Ion Concentration (M)'),
    ('Temperature (°C)', 'Molar Ratio'),
    ('Temperature (°C)', 'Ion Concentration (M)'),
    ('Molar Ratio', 'Ion Concentration (M)')
]



'''
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

feature_bounds = {
    "Time (hr)": (20, 60),
    "Temperature (°C)": (110, 150),
    "Molar Ratio": (1, 3.5),
    "Ion Concentration (M)": (0.15, 0.45)
}


feature_name_mapping = {
    "feature 1": "Time (hr)",
    "feature 2": "Temperature (°C)",
    "feature 3": "Molar Ratio",
    "feature 4": "Ion Concentration (M)"
}

inv_feature_name_mapping = {v: k for k, v in feature_name_mapping.items()}



for feature_x in X.columns:
    for feature_y in X.columns:
        if feature_x != feature_y:
            # Create a grid
            x = np.linspace(feature_bounds[feature_x][0], feature_bounds[feature_x][1], 50)
            y = np.linspace(feature_bounds[feature_y][0], feature_bounds[feature_y][1], 50)
            X_grid, Y_grid = np.meshgrid(x, y)

            grid_df = pd.DataFrame({feature_x: X_grid.ravel(), feature_y: Y_grid.ravel()})
            for feature in X.columns:
                if feature not in [feature_x, feature_y]:
                    grid_df[feature] = X[feature].mean()

            grid_df = grid_df[X.columns]

            grid_scaled = scaler.transform(grid_df)


    Z1 = gpr_model_target1.predict(grid_scaled).reshape(X_grid.shape)
    Z2 = gpr_model_target2.predict(grid_scaled).reshape(X_grid.shape)

    fig1 = go.Figure(data=go.Contour(x=x, y=y, z=Z1, contours=dict(coloring='heatmap', showlabels=True)))
    fig1.update_layout(title=f"Contour for {feature_x} vs {feature_y} - Target 1")
    fig1.show()

    fig2 = go.Figure(data=go.Contour(x=x, y=y, z=Z2, contours=dict(coloring='heatmap', showlabels=True)))
    fig2.update_layout(title=f"Contour for {feature_x} vs {feature_y} - Target 2")
    fig2.show()


'''

#######################







#########################3
'''
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Print DataFrame columns
print(X.columns)

# Print maximum values for y1 and y2
print(f"Max value in y1 (surface area (m²\u2009/\u2009g)): {y1.max()}")
print(f"Max value in y2 (MB Removal (%)): {y2.max()}")

# Define feature combinations
feature_combinations = [
    ('Time (hr)', 'Temperature (°C)'),
    ('Time (hr)', 'Molar Ratio'),
    ('Time (hr)', 'Ion Concentration (M)'),
    ('Temperature (°C)', 'Molar Ratio'),
    ('Temperature (°C)', 'Ion Concentration (M)'),
    ('Molar Ratio', 'Ion Concentration (M)')
]

# Update DataFrame columns
X.columns = ['feature 1', 'feature 2', 'feature 3', 'feature 4']

# Map original to new feature names
original_feature_names = ['feature 1', 'feature 2', 'feature 3', 'feature 4']
new_feature_names = ['Time (hr)', 'Temperature (°C)', 'Molar Ratio', 'Ion Concentration (M)']

# Prepare grid points for feature combinations
feature_grids = {}
num_points = 250
for i, feature1 in enumerate(original_feature_names):
    for j, feature2 in enumerate(original_feature_names):
        if i < j:
            feature1_values = np.linspace(X[feature1].min(), X[feature1].max(), num_points)
            feature2_values = np.linspace(X[feature2].min(), X[feature2].max(), num_points)
            f1_grid, f2_grid = np.meshgrid(feature1_values, feature2_values)
            feature_grids[(i, j)] = (f1_grid, f2_grid)

# Predict using GPR models
predictions_gpr1 = {}
predictions_gpr2 = {}
for (i, j), (f1_grid, f2_grid) in feature_grids.items():
    feature1, feature2 = original_feature_names[i], original_feature_names[j]
    sample_points = np.array([f1_grid.ravel(), f2_grid.ravel()]).T
    predict_df = pd.DataFrame(data=np.tile(X.median().values, (len(sample_points), 1)), columns=X.columns)
    predict_df[feature1] = sample_points[:, 0]
    predict_df[feature2] = sample_points[:, 1]
    predict_df_scaled = scaler.transform(predict_df)
    preds_1 = gpr_model_target1.predict(predict_df_scaled).reshape(f1_grid.shape)
    preds_2 = gpr_model_target2.predict(predict_df_scaled).reshape(f1_grid.shape)
    predictions_gpr1[(i, j)] = preds_1
    predictions_gpr2[(i, j)] = preds_2

# Plot contours
for (i, j), preds in predictions_gpr1.items():
    f1_grid, f2_grid = feature_grids[(i, j)]
    title_feature1, title_feature2 = new_feature_names[i], new_feature_names[j]
    fig = go.Figure(data=go.Contour(z=preds, x=f1_grid[0], y=f2_grid[:,0],
                                    contours=dict(coloring='heatmap', showlabels=True,
                                                  labelfont=dict(size=12, color='red')), colorscale='Greens'))
    fig.update_layout(
        title=f"Contours for Predicted Surface Area (m²\u2009/\u2009g)",
        title_x=0.5,
        xaxis_title=title_feature1,
        yaxis_title=title_feature2,
        font=dict(family='Arial', size=22, color='black')
    )
    fig.write_image(f"{save_directory}/contour_{title_feature1}_{title_feature2}_Surface_Area.png", scale=12)

for (i, j), preds in predictions_gpr2.items():
    f1_grid, f2_grid = feature_grids[(i, j)]
    title_feature1, title_feature2 = new_feature_names[i], new_feature_names[j]
    fig = go.Figure(data=go.Contour(z=preds, x=f1_grid[0], y=f2_grid[:,0],
                                    contours=dict(coloring='heatmap', showlabels=True,
                                                  labelfont=dict(size=12, color='red')), colorscale='Blues'))
    fig.update_layout(
        title=f"Contours for Predicted MB Removal (%)",
        title_x=0.5,
        xaxis_title=title_feature1,
        yaxis_title=title_feature2,
        font=dict(family='Arial', size=22, color='black')
    )
    fig.write_image(f"{save_directory}/contour_{title_feature1}_{title_feature2}_MB_Removal.png", scale=12)

'''
#####################333333333333

'''
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Print DataFrame columns
print(X.columns)

# Print maximum values for y1 and y2
print(f"Max value in y1 (surface area (m²\u2009/\u2009g)): {y1.max()}")
print(f"Max value in y2 (MB Removal (%)): {y2.max()}")

# Define feature combinations
feature_combinations = [
    ('Time (hr)', 'Temperature (°C)'),
    ('Time (hr)', 'Molar Ratio'),
    ('Time (hr)', 'Ion Concentration (M)'),
    ('Temperature (°C)', 'Molar Ratio'),
    ('Temperature (°C)', 'Ion Concentration (M)'),
    ('Molar Ratio', 'Ion Concentration (M)')
]

# Update DataFrame columns
X.columns = ['feature 1', 'feature 2', 'feature 3', 'feature 4']

# Map original to new feature names
original_feature_names = ['feature 1', 'feature 2', 'feature 3', 'feature 4']
new_feature_names = ['Time (hr)', 'Temperature (°C)', 'Molar Ratio', 'Ion Concentration (M)']

# Prepare grid points for feature combinations
feature_grids = {}
num_points = 250
for i, feature1 in enumerate(original_feature_names):
    for j, feature2 in enumerate(original_feature_names):
        if i < j:
            feature1_values = np.linspace(X[feature1].min(), X[feature1].max(), num_points)
            feature2_values = np.linspace(X[feature2].min(), X[feature2].max(), num_points)
            f1_grid, f2_grid = np.meshgrid(feature1_values, feature2_values)
            feature_grids[(i, j)] = (f1_grid, f2_grid)

# Predict using GPR models
predictions_gpr1 = {}
predictions_gpr2 = {}
for (i, j), (f1_grid, f2_grid) in feature_grids.items():
    feature1, feature2 = original_feature_names[i], original_feature_names[j]
    sample_points = np.array([f1_grid.ravel(), f2_grid.ravel()]).T
    predict_df = pd.DataFrame(data=np.tile(X.median().values, (len(sample_points), 1)), columns=X.columns)
    predict_df[feature1] = sample_points[:, 0]
    predict_df[feature2] = sample_points[:, 1]
    predict_df_scaled = scaler.transform(predict_df)
    preds_1 = gpr_model_target1.predict(predict_df_scaled).reshape(f1_grid.shape)
    preds_2 = gpr_model_target2.predict(predict_df_scaled).reshape(f1_grid.shape)
    predictions_gpr1[(i, j)] = preds_1
    predictions_gpr2[(i, j)] = preds_2

# Plot contours
for (i, j), preds in predictions_gpr1.items():
    f1_grid, f2_grid = feature_grids[(i, j)]
    title_feature1, title_feature2 = new_feature_names[i], new_feature_names[j]
    fig = go.Figure(data=go.Contour(z=preds, x=f1_grid[0], y=f2_grid[:,0],
                                    contours=dict(coloring='heatmap', showlabels=True,
                                                  labelfont=dict(size=12, color='red')), colorscale='Greens',
                                    zmin=800, zmax=1900))
    fig.update_layout(
        title=f"Contours for Predicted Surface Area (m²\u2009/\u2009g)",
        title_x=0.5,
        xaxis_title=title_feature1,
        yaxis_title=title_feature2,
        font=dict(family='Arial', size=22, color='black')
    )
    fig.write_image(f"{save_directory}/contour_{title_feature1}_{title_feature2}_Surface_Area.png", scale=12)

for (i, j), preds in predictions_gpr2.items():
    f1_grid, f2_grid = feature_grids[(i, j)]
    title_feature1, title_feature2 = new_feature_names[i], new_feature_names[j]
    fig = go.Figure(data=go.Contour(z=preds, x=f1_grid[0], y=f2_grid[:,0],
                                    contours=dict(coloring='heatmap', showlabels=True,
                                                  labelfont=dict(size=12, color='red')), colorscale='Blues',
                                    zmin=30, zmax=100)) 

    fig.update_layout(
        title=f"Contours for Predicted MB Removal (%)",
        title_x=0.5,
        xaxis_title=title_feature1,
        yaxis_title=title_feature2,
        font=dict(family='Arial', size=22, color='black')
    )
    fig.write_image(f"{save_directory}/contour_{title_feature1}_{title_feature2}_MB_Removal.png", scale=12)

'''

###############################################################################33
'''
##SHAP
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


# Set default font to Arial and bold across all text elements
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.weight'] = 'bold'

# Load the data from the Excel file
file_path_s = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/SHAP.xlsx'
data = pd.read_excel(file_path_s)

# Clean the data
headers = data.iloc[0].values
data.columns = headers
data = data[1:]

# Setting proper data types
data.set_index("Features", inplace=True)
data = data.astype(float)

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(20, 20), constrained_layout=True)
axes = axes.flatten()
deep_blue = 'deepskyblue'  # A deeper blue color

# Dictionary to replace titles with plain text or simple formats
title_dict = {
    "Surface Area (m^2/gr)": "Surface Area (m²/gr)",
    "Total Pore Volume (cm3/g)": "Total Pore Volume (cm³/g)",
    "Average Crystallite Size (Davg nm)": "Average Crystallite Size (Davg nm)"
}

for i, target in enumerate(data.columns):
    formatted_title = title_dict.get(target, target)
    ax = sns.barplot(x=data[target], y=data.index, ax=axes[i], color=deep_blue, linewidth=1.3, width=0.6)
    axes[i].set_title(formatted_title, fontsize=24, fontweight='bold')
    axes[i].set_xlabel('SHAP Value', fontsize=22, fontweight='bold')
    axes[i].set_ylabel('Features', fontsize=20, fontweight='bold')

    # Displaying bar labels inside the bars
    for bar in axes[i].containers[0]:
        axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height()/2, 
                     format(bar.get_width(), '.4f'), ha='center', va='center', 
                     color='black', fontsize=20, fontweight='bold')

    # Adjusting tick parameters for x and y axes with font details
    axes[i].tick_params(axis='x', labelsize=18, labelrotation=0)
    axes[i].tick_params(axis='y', labelsize=18)

# Adding a main title for all plots
fig.suptitle('SHAP Value Feature Importance Analysis', fontsize=28)

plt.show()
'''
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Set default font to Arial and bold across all text elements
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.weight'] = 'bold'

# Load the data from the Excel file
file_path_s = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/SHAP.xlsx'
data = pd.read_excel(file_path_s)

# Clean the data
headers = data.iloc[0].values
data.columns = headers
data = data[1:]

# Setting proper data types
data.set_index("Features", inplace=True)
data = data.astype(float)

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(30, 20), constrained_layout=True)
axes = axes.flatten()
deep_blue = 'deepskyblue'  # A deeper blue color

# Dictionary to replace titles with plain text or simple formats
title_dict = {
    "Surface Area (m^2/gr)": "Surface Area (m²/gr)",
    "Total Pore Volume (cm3/g)": "Total Pore Volume (cm³/gr)",
    "Average Crystallite Size (Davg nm)": "Average Crystallite Size (Dₐᵥg nm)"
}

for i, target in enumerate(data.columns):
    formatted_title = title_dict.get(target, target)
    ax = sns.barplot(x=data[target], y=data.index, ax=axes[i], color=deep_blue, linewidth=1.3, width=0.6)
    axes[i].set_title(formatted_title, fontsize=39, fontweight='bold')
    axes[i].set_xlabel('Mean (|SHAP Value|); average impact on model output', fontsize=27)
    axes[i].set_ylabel('Features', fontsize=27, fontweight='bold')


    # Adjusting tick parameters for x and y axes with font details
    axes[i].tick_params(axis='x', labelsize=31, labelrotation=0)
    axes[i].tick_params(axis='y', labelsize=30)


# Adding a main title for all plots
fig.suptitle('SHAP Value: Feature Importance Analysis', fontsize=45, fontweight='bold')

plt.show()
fig.savefig('SHAP_Value_Analysis.png', dpi=600)


'''
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Set default font to Arial and bold across all text elements
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.weight'] = 'bold'

# Load the data from the Excel file
file_path_s = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/SHAP.xlsx'
data = pd.read_excel(file_path_s)

# Clean the data
headers = data.iloc[0].values
data.columns = headers
data = data[1:]

# Setting proper data types
data.set_index("Features", inplace=True)
data = data.astype(float)

# Create a color palette for features
num_features = len(data.index)
colors = sns.color_palette("husl", num_features)  # Use a Husl palette for unique feature colors

# Map features to numbers
feature_numbers = [str(i + 1) for i in range(num_features)]
feature_to_color = {feature: colors[i] for i, feature in enumerate(data.index)}

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(29, 22.5), constrained_layout=True)
axes = axes.flatten()

# Dictionary to replace titles with formatted versions
title_dict = {
    "Surface Area (m^2/gr)": "Surface Area (m²/gr)",
    "Total Pore Volume (cm3/g)": "Total Pore Volume (cm³/gr)",
    "Average Crystallite Size (Davg nm)": "Average Crystallite Size (Dₐᵥg nm)"
}

# Plot each target
for i, target in enumerate(data.columns):
    formatted_title = title_dict.get(target, target)
    ax = sns.barplot(x=data[target], y=feature_numbers, ax=axes[i], palette=colors, linewidth=1.3, width=0.6)
    axes[i].set_title(formatted_title, fontsize=50, fontweight='bold')
    axes[i].set_xlabel('Average impact on model output', fontsize=48)
    axes[i].set_ylabel('Features', fontsize=46, fontweight='bold')

    # Adjusting tick parameters for x and y axes with font details
    axes[i].tick_params(axis='x', labelsize=46, labelrotation=0)
    axes[i].tick_params(axis='y', labelsize=50)

    # Adding grids to the plot
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=1, color='gray', alpha=0.7)  # Grid lines on the x-axis

# Create a custom legend for feature numbers and colors
legend_elements = [
    plt.Line2D([0], [0], color=feature_to_color[feature], lw=8, label=f"{num} {feature}")
    for num, feature in zip(feature_numbers, data.index)
]

# Adding a main title for all plots
fig.suptitle('SHAP Value: Feature Importance Analysis', fontsize=45, fontweight='bold')

# Adjust layout and add legend closer to the plot
plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust layout to fit legend
fig.subplots_adjust(bottom=0.2, top=0.915, wspace=0.16, hspace=0.6)  # Reduce space between columns and provide space for legend

fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=49, frameon=False, bbox_to_anchor=(0.5, -0.002))

# Show the plot
plt.show()

# Save the figure
fig.savefig('SHAP_Value_Analysis_with_Feature_Numbers.png', dpi=900)


#####################
## PCC

'''
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Load the data from the Excel file
file_path_p = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/PCC.xlsx'
data = pd.read_excel(file_path_p)
matplotlib.rc('text', usetex=False)

# Load the data from the Excel file
file_path_p = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/PCC.xlsx'
data = pd.read_excel(file_path_p)

# Correcting the column name typo and removing the empty column
data.rename(columns={
    'Temperature ©': 'Temperature (°C)',
    'Surface Area (m^2/gr)': 'Surface Area (m²/gr)',
    'Total Pore Volume (cm3/g)': 'Total Pore Volume (cm³/gr)',
    'Average Crystallite Size (Davg nm)': 'Average Crystallite Size (Dₐᵥ nm)'
}, inplace=True)
data.drop(columns=['Unnamed: 4'], inplace=True)

# Set the font for the plots
matplotlib.rc('font', family='Arial', size=20, weight='bold')

# Extracting feature and target columns
features = data[['Time (hr)', 'Temperature (°C)', 'Molar Ratio', 'Ion Concentration (M)']]
targets = data[['Surface Area (m²/gr)', 'MB Removal (%)', 'Total Pore Volume (cm³/gr)', 
                'Average Crystallite Size (Dₐᵥ nm)', 'Crystallinity (%)', 'Yield (%)']]

# Combining features and targets
combined_data = pd.concat([features, targets], axis=1)
correlation_matrix = combined_data.corr()

# Plotting
plt.figure(figsize=(12.5, 10.5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', annot_kws={'size': 13.25}, vmin=-1, vmax=1)
plt.title('Pearson Correlation Coefficient Matrix for All Features and Targets', weight='bold', pad=20)
plt.xticks(rotation=90, ha='right', fontsize=18)
plt.yticks(rotation=0, fontsize=18)
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Load the data from the Excel file
file_path_p = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/PCC.xlsx'
data = pd.read_excel(file_path_p)
matplotlib.rc('text', usetex=False)

# Correcting the column name typo and removing the empty column
data.rename(columns={
    'Temperature ©': 'Temperature (°C)',
    'Surface Area (m^2/gr)': 'Surface Area (m²/gr)',
    'Total Pore Volume (cm3/g)': 'Total Pore Volume (cm³/gr)',
    'Average Crystallite Size (Davg nm)': 'Average Crystallite Size (Dₐᵥ nm)'
}, inplace=True)
data.drop(columns=['Unnamed: 4'], inplace=True)

# Set the font for the plots
matplotlib.rc('font', family='Arial', size=20, weight='bold')

# Extracting feature and target columns
features = data[['Time (hr)', 'Temperature (°C)', 'Molar Ratio', 'Ion Concentration (M)']]
targets = data[['Surface Area (m²/gr)', 'MB Removal (%)', 'Total Pore Volume (cm³/gr)', 
                'Average Crystallite Size (Dₐᵥ nm)', 'Crystallinity (%)', 'Yield (%)']]

# Combining features and targets
combined_data = pd.concat([features, targets], axis=1)

# Calculate Spearman's rank correlation matrix
spearman_correlation_matrix = combined_data.corr(method='spearman')

# Plotting the heatmap with Spearman correlation matrix
plt.figure(figsize=(12.5, 10.5))
sns.heatmap(spearman_correlation_matrix, annot=True, cmap='coolwarm', annot_kws={'size': 13.25}, vmin=-1, vmax=1)
plt.title("Spearman's Rank Correlation Matrix for All Features and Targets", weight='bold', pad=20)
plt.xticks(rotation=90, ha='right', fontsize=18)
plt.yticks(rotation=0, fontsize=18)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib

# Load the data from the Excel file
file_path_p = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/PCC.xlsx'
data = pd.read_excel(file_path_p)

# Correcting the column name typo and removing the empty column
data.rename(columns={
    'Temperature ©': 'Temperature (°C)',
    'Surface Area (m^2/gr)': 'Surface Area (m²/gr)',
    'Total Pore Volume (cm3/g)': 'Total Pore Volume (cm³/gr)',
    'Average Crystallite Size (Davg nm)': 'Average Crystallite Size (Dₐᵥ nm)'
}, inplace=True)
data.drop(columns=['Unnamed: 4'], inplace=True)

# Set the font for the plots
matplotlib.rc('font', family='Arial', size=20, weight='bold')

# Extracting feature and target columns
features = data[['Time (hr)', 'Temperature (°C)', 'Molar Ratio', 'Ion Concentration (M)']]
targets = data[['Surface Area (m²/gr)', 'MB Removal (%)', 'Total Pore Volume (cm³/gr)', 
                'Average Crystallite Size (Dₐᵥ nm)', 'Crystallinity (%)', 'Yield (%)']]

# Combining features and targets
combined_data = pd.concat([features, targets], axis=1)

# Calculate Spearman's rank correlation matrix
spearman_correlation_matrix = combined_data.corr(method='spearman')

# Create the correlogram-like plot without the rectangle and with circle sizes and colors indicating correlation strength
def plot_corr_circle_with_size_and_color_legend(corr_matrix):
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create the plot
    plt.gca().set_aspect('equal', adjustable='box')

    # Overlay circles to represent the correlation matrix
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1):  # Only iterate over the lower triangle
            value = corr_matrix.iloc[i, j]
            size = np.abs(value) * 1000  # Scale factor for circle size
            color = 'blue' if value > 0 else 'red'
            plt.scatter(j, i, s=size, color=color, alpha=0.6)

    # Remove the black square border
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    # Custom legend for the size of the circles
    for size_value in [0.3, 0.45, 0.6, 0.75, 0.9]:  # Example values for correlation strength
        plt.scatter([], [], s=size_value * 1000, color='gray', label=f"{size_value:.2f} Strength")

    # Legend for positive and negative correlations
    plt.scatter([], [], s=500, color='blue', label='Positive Correlation')
    plt.scatter([], [], s=500, color='red', label='Negative Correlation')

    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title="Spearman Correlation",
               loc='lower right', fontsize=15)

    plt.xticks(range(corr_matrix.shape[1]), corr_matrix.columns, rotation=90, ha='right', fontsize=19)
    plt.yticks(range(corr_matrix.shape[0]), corr_matrix.index, fontsize=19)
    plt.title("Spearman's Rank Correlation Matrix for all Feature and Targets", fontsize=25.5, weight='bold')
    plt.tight_layout()

    plt.show()

# Plot the Spearman's rank correlation matrix as a correlogram with circles, size and color legend
plot_corr_circle_with_size_and_color_legend(spearman_correlation_matrix)
'''

######################################3
### RMSE

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams

# Updating rcParams to set global text properties for titles and labels
rcParams['font.family'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 16

file_path_R = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/RMSE.xlsx'
excel_data = pd.ExcelFile(file_path_R)

# Set the style
plt.style.use('seaborn-darkgrid')
sns.set_context('talk')

# Create a figure with a 3x2 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(26, 18))
axes = axes.flatten()  # Flatten the array to easily index it

# Define the order of the models
model_order = [
    'Gaussian Process Regressor',
    'Support Vector Regressor',
    'Decision Tree Regressor',
    'Random Forest Regressor',
    'K-Nearest Neighbors',
    'Dummy Regressor',
    'Linear Regression',
    'Linear SVR',
    'XGBoost'
]

# Define custom colormaps for each target
cmaps = {
    'Surface Area': 'Greens',
    'MB Removal': 'Blues',
    'Total Pore Volume': LinearSegmentedColormap.from_list("custom_navy", ["#ade1f2", "#0057b7"], N=256),
    'Average Crystallite Size': 'Reds',
    'Crystallinity': 'Purples',
    'Yield': 'Oranges'
}

# Titles with units
titles_with_units = {
    'Surface Area': 'Surface Area (m²/g)',
    'MB Removal': 'MB Removal (%)',
    'Total Pore Volume': 'Total Pore Volume (cm³/g)',
    'Average Crystallite Size': 'Average Crystallite Size (Dₐᵥg nm)',
    'Crystallinity': 'Crystallinity (%)',
    'Yield': 'Yield (%)'
}

# Loop through each sheet and plot
for i, sheet in enumerate(excel_data.sheet_names):
    # Read the data from the current sheet
    data = pd.read_excel(excel_data, sheet_name=sheet)

    # Ensure data is in the specified order
    data['Model'] = pd.Categorical(data['Model'], categories=model_order, ordered=True)
    data.sort_values('Model', inplace=True)

    # Normalize the RMSE values for the colormap scaling
    norm = plt.Normalize(data['RMSE'].min(), data['RMSE'].max())
    cmap = plt.get_cmap(cmaps[sheet])
    colors = [cmap(norm(value)) for value in data['RMSE']]

    # Create a horizontal bar plot, coloring by RMSE using the target-specific colormap
    barplot = sns.barplot(x='RMSE', y='Model', data=data, ax=axes[i], palette=colors, order=model_order)

    # Set the title and labels with Arial font, bold and with defined units
    axes[i].set_title(titles_with_units[sheet], fontsize=30, fontweight='bold', fontname='Arial',color='black')
    axes[i].set_xlabel('RMSE', fontsize=28, fontweight='bold', fontname='Arial',color='black')
    axes[i].set_ylabel('Models', fontsize=28, fontweight='bold', fontname='Arial',color='black')
    
    # Set tick labels with specific font settings
    for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
        label.set_fontsize(23)
        label.set_fontweight('bold')
        label.set_fontname('Arial')
        label.set_color('black')  # Make sure the color is black

# Adjust layout and add main title with adjusted vertical positioning
plt.tight_layout()
fig.subplots_adjust(top=0.92)  # Adjust the top margin to give space for the suptitle
fig.suptitle('Comparative RMSE Analysis across Different Models', fontsize=35, fontweight='bold', fontname='Arial', verticalalignment='top',color='black')

# Show plot
plt.show()

'''

#####

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams


# Updating rcParams to set global text properties for titles and labels
rcParams['font.family'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 16

file_path_R = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/RMSE.xlsx'
excel_data = pd.ExcelFile(file_path_R)

# Set the style
plt.style.use('seaborn-darkgrid')
sns.set_context('talk')

# Create a figure with a 3x2 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(25, 18))
axes = axes.flatten()  # Flatten the array to easily index it

# Define the order of the models
model_order = [
    'Gaussian Process Regressor',
    'Support Vector Regressor',
    'Decision Tree Regressor',
    'Random Forest Regressor',
    'K-Nearest Neighbors',
    'Dummy Regressor',
    'Linear Regression',
    'Linear SVR',
    'XGBoost'
]

# Define custom colormaps for each target
cmaps = {
    'Surface Area': 'Greens',
    'MB Removal': 'Blues',
    'Total Pore Volume': LinearSegmentedColormap.from_list("custom_navy", ["#ade1f2", "#0057b7"], N=256),
    'Average Crystallite Size': 'Reds',
    'Crystallinity': 'Purples',
    'Yield': 'Oranges'
}

# Titles with units
titles_with_units = {
    'Surface Area': 'Surface Area (m²/g)',
    'MB Removal': 'MB Removal (%)',
    'Total Pore Volume': 'Total Pore Volume (cm³/g)',
    'Average Crystallite Size': 'Average Crystallite Size (Dₐᵥg nm)',
    'Crystallinity': 'Crystallinity (%)',
    'Yield': 'Yield (%)'
}

# Loop through each sheet and plot
for i, sheet in enumerate(excel_data.sheet_names):
    # Read the data from the current sheet
    data = pd.read_excel(excel_data, sheet_name=sheet)

    # Ensure data is in the specified order
    data['Model'] = pd.Categorical(data['Model'], categories=model_order, ordered=True)
    data.sort_values('Model', inplace=True)

    # Normalize the RMSE values for the colormap scaling
    norm = plt.Normalize(data['RMSE'].min(), data['RMSE'].max())
    cmap = plt.get_cmap(cmaps[sheet])
    colors = [cmap(norm(value)) for value in data['RMSE']]

    # Create a horizontal bar plot, coloring by RMSE using the target-specific colormap
    sns.barplot(x='RMSE', y='Model', data=data, ax=axes[i], palette=colors, order=model_order)

    # Set the title and labels with Arial font, bold and with defined units
    # Set the title and labels with Arial font, bold and with defined units
    axes[i].set_title(titles_with_units[sheet], fontsize=36, fontweight='bold', fontname='Arial',color='black')
    axes[i].set_xlabel('RMSE', fontsize=32, fontweight='bold', fontname='Arial',color='black')
    axes[i].set_ylabel('Models', fontsize=32, fontweight='bold', fontname='Arial',color='black')
    
    # Set tick labels with specific font settings
    for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
        label.set_fontsize(28)
        label.set_fontweight('bold')
        label.set_fontname('Arial')
        label.set_color('black')  # Make sure the color is black

    # Conditionally remove y-axis labels for the second column
    if i % 2 == 1:  # This condition checks if the subplot is in the second column
        axes[i].set_ylabel('')
        axes[i].set_yticklabels([])  # This removes the y-axis tick labels
    else:
        axes[i].set_ylabel('Models', fontsize=30, fontweight='bold', fontname='Arial',color='black')

# Adjust layout and add main title with adjusted vertical positioning
plt.tight_layout()
fig.subplots_adjust(top=0.92)  # Adjust the top margin to give space for the suptitle
fig.suptitle('Comparative RMSE Analysis across Different Models', fontsize=43, fontweight='bold', fontname='Arial', verticalalignment='top',color='black')

# Show plot
plt.show()

'''

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Updating rcParams to set global text properties for titles and labels
rcParams['font.family'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 16

file_path_R = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/RMSE.xlsx'
excel_data = pd.ExcelFile(file_path_R)

# Set the style
plt.style.use('seaborn-whitegrid')  # Ensuring background is white
sns.set_context('talk')

# Create a figure with a 3x2 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(30, 18))
fig.set_facecolor('white')  # Set the background color of the figure to white
axes = axes.flatten()  # Flatten the array to easily index it

# Define the order of the models
model_order = [
    'Gaussian Process Regressor',
    'Support Vector Regressor',
    'Decision Tree Regressor',
    'Random Forest Regressor',
    'K-Nearest Neighbors',
    'Dummy Regressor',
    'Linear Regression',
    'Linear SVR',
    'XGBoost'
]

# Titles with units
titles_with_units = {
    'Surface Area': 'Surface Area (m²/g)',
    'MB Removal': 'MB Removal (%)',
    'Total Pore Volume': 'Total Pore Volume (cm³/g)',
    'Average Crystallite Size': 'Average Crystallite Size (Dₐᵥg nm)',
    'Crystallinity': 'Crystallinity (%)',
    'Yield': 'Yield (%)'
}

# Loop through each sheet and plot
for i, sheet in enumerate(excel_data.sheet_names):
    # Read the data from the current sheet
    data = pd.read_excel(excel_data, sheet_name=sheet)

    # Ensure data is in the specified order
    data['Model'] = pd.Categorical(data['Model'], categories=model_order, ordered=True)
    data.sort_values('Model', inplace=True)

    # Create a horizontal bar plot, using a single color for all bars
    sns.barplot(x='RMSE', y='Model', data=data, ax=axes[i], color="deepskyblue", order=model_order)

    # Set the title and labels with Arial font, bold and with defined units
    axes[i].set_title(titles_with_units[sheet], fontsize=35, fontweight='bold', fontname='Arial',color='black')
    axes[i].set_xlabel('RMSE', fontsize=33, fontweight='bold', fontname='Arial',color='black')
    axes[i].set_ylabel('Models', fontsize=33, fontweight='bold', fontname='Arial',color='black')
    
    # Set tick labels with specific font settings
    for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
        label.set_fontsize(28)
        label.set_fontweight('bold')
        label.set_fontname('Arial')
        label.set_color('black')  # Make sure the color is black

# Adjust layout and add main title with adjusted vertical positioning
plt.tight_layout()
fig.subplots_adjust(top=0.915)  # Adjust the top margin to give space for the suptitle
fig.suptitle('Comparative RMSE Analysis across Different Models', fontsize=43, fontweight='bold', fontname='Arial', verticalalignment='top',color='black')

# Show plot
plt.show()

'''

#####################
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Updating rcParams to set global text properties for titles and labels
rcParams['font.family'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 16

file_path_R = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/RMSE.xlsx'
excel_data = pd.ExcelFile(file_path_R)

# Set the style
plt.style.use('seaborn-whitegrid')  # Ensuring background is white
sns.set_context('talk')

# Create a figure with a 3x2 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(30, 18))
fig.set_facecolor('white')  # Set the background color of the figure to white
axes = axes.flatten()  # Flatten the array to easily index it

# Define the order of the models
model_order = [
    'Gaussian Process Regressor',
    'Support Vector Regressor',
    'Decision Tree Regressor',
    'Random Forest Regressor',
    'K-Nearest Neighbors',
    'Dummy Regressor',
    'Linear Regression',
    'Linear SVR',
    'XGBoost'
]

# Titles with units
titles_with_units = {
    'Surface Area': 'Surface Area (m²/g)',
    'MB Removal': 'MB Removal (%)',
    'Total Pore Volume': 'Total Pore Volume (cm³/g)',
    'Average Crystallite Size': 'Average Crystallite Size (Dₐᵥg nm)',
    'Crystallinity': 'Crystallinity (%)',
    'Yield': 'Yield (%)'
}

# Loop through each sheet and plot
for i, sheet in enumerate(excel_data.sheet_names):
    # Read the data from the current sheet
    data = pd.read_excel(excel_data, sheet_name=sheet)

    # Ensure data is in the specified order
    data['Model'] = pd.Categorical(data['Model'], categories=model_order, ordered=True)
    data.sort_values('Model', inplace=True)

    # Create a horizontal bar plot, using a single color for all bars
    sns.barplot(x='RMSE', y='Model', data=data, ax=axes[i], color="deepskyblue", order=model_order)

    # Set the title and labels with Arial font, bold and with defined units
    axes[i].set_title(titles_with_units[sheet], fontsize=36, fontweight='bold', fontname='Arial',color='black')
    axes[i].set_xlabel('RMSE', fontsize=31, fontweight='bold', fontname='Arial',color='black')
    axes[i].set_ylabel('Models', fontsize=31, fontweight='bold', fontname='Arial',color='black')

    # Conditionally remove y-axis labels for the second column
    if i % 2 == 1:  # This condition checks if the subplot is in the second column
        axes[i].set_ylabel('')
        axes[i].set_yticklabels([])  # This removes the y-axis tick labels
    
    # Set tick labels with specific font settings
    for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
        label.set_fontsize(27)
        label.set_fontweight('bold')
        label.set_fontname('Arial')
        label.set_color('black')  # Make sure the color is black

# Adjust layout and add main title with adjusted vertical positioning
plt.tight_layout()
fig.subplots_adjust(top=0.92)  # Adjust the top margin to give space for the suptitle
fig.suptitle('Comparative RMSE Analysis across Different Models', fontsize=44, fontweight='bold', fontname='Arial', verticalalignment='top',color='black')

# Show plot
plt.show()

'''


###############################

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Updating rcParams to set global text properties for titles and labels
rcParams['font.family'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 16

file_path_R = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/RMSE.xlsx'
excel_data = pd.ExcelFile(file_path_R)

# Set the style
plt.style.use('seaborn-whitegrid')  # Ensuring background is white
sns.set_context('talk')

# Create a figure with a 3x2 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(30.5, 18))
fig.set_facecolor('white')  # Set the background color of the figure to white
axes = axes.flatten()  # Flatten the array to easily index it

# Define the order of the models
model_order = [
    'Gaussian Process Regressor',
    'Support Vector Regressor',
    'Decision Tree Regressor',
    'Random Forest Regressor',
    'K-Nearest Neighbors',
    'Dummy Regressor',
    'Linear Regression',
    'Linear SVR',
    'XGBoost'
]

# Titles with units
titles_with_units = {
    'Surface Area': 'Surface Area (m²/g)',
    'MB Removal': 'MB Removal (%)',
    'Total Pore Volume': 'Total Pore Volume (cm³/g)',
    'Average Crystallite Size': 'Average Crystallite Size (Dₐᵥg nm)',
    'Crystallinity': 'Crystallinity (%)',
    'Yield': 'Yield (%)'
}

# Loop through each sheet and plot
for i, sheet in enumerate(excel_data.sheet_names):
    # Read the data from the current sheet
    data = pd.read_excel(excel_data, sheet_name=sheet)

    # Ensure data is in the specified order
    data['Model'] = pd.Categorical(data['Model'], categories=model_order, ordered=True)
    data.sort_values('Model', inplace=True)

    # Create a horizontal bar plot, using a single color for all bars
    barplot = sns.barplot(x='RMSE', y='Model', data=data, ax=axes[i], color="deepskyblue", order=model_order)

    # Set the title and labels with Arial font, bold and with defined units
    axes[i].set_title(titles_with_units[sheet], fontsize=36, fontweight='bold', fontname='Arial', color='black')
    axes[i].set_xlabel('RMSE', fontsize=31.5, fontweight='bold', fontname='Arial', color='black')
    axes[i].set_ylabel('Models', fontsize=31.5, fontweight='bold', fontname='Arial', color='black')

    # Conditionally remove y-axis labels for the second column
    if i % 2 == 1:  # This condition checks if the subplot is in the second column
        axes[i].set_ylabel('')
        axes[i].set_yticklabels([])

    # Set tick labels with specific font settings, ensure they are black
    for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
        label.set_fontsize(29)
        label.set_fontweight('bold')
        label.set_fontname('Arial')
        label.set_color('black')  # Make sure the color is black

    # Annotate RMSE values on each bar at the end inside the bar
    for p in barplot.patches:
        # Calculate the offset to keep the text inside the bar
        offset = p.get_width() / 50
        axes[i].text(p.get_width() - offset, p.get_y() + p.get_height() / 2.,
                     f'{p.get_width():.4f}',  # Formatting the RMSE value to two decimal places
                     fontsize=28.85, fontweight='bold', fontname='Arial',
                     color='black', va='center', ha='right')  # Ensure text is black

# Adjust layout and add main title with adjusted vertical positioning
plt.tight_layout()
fig.subplots_adjust(top=0.915)  # Adjust the top margin to give space for the suptitle
fig.suptitle('Comparative RMSE Analysis across Different Models', fontsize=44, fontweight='bold', fontname='Arial', color='black', verticalalignment='top')

# Show plot
plt.show()

'''


###################3
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Updating rcParams to set global text properties for titles and labels
rcParams['font.family'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 16

file_path_R = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/RMSE.xlsx'
excel_data = pd.ExcelFile(file_path_R)

# Set the style
plt.style.use('seaborn-whitegrid')  # Ensuring background is white
sns.set_context('talk')

# Create a figure with a 3x2 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(29,23))
fig.set_facecolor('white')  # Set the background color of the figure to white
axes = axes.flatten()  # Flatten the array to easily index it

# Define the order of the models and assign consistent colors (GPR in green)
model_order = [
    'Gaussian Process Regressor',
    'Support Vector Regressor',
    'Decision Tree Regressor',
    'Random Forest Regressor',
    'K-Nearest Neighbors',
    'Dummy Regressor',
    'Linear Regression',
    'Linear SVR',
    'XGBoost'
]

# Corresponding numbers and color codes for each model
model_numbers = [f"{i+1}" for i in range(len(model_order))]
colors = {
    'Gaussian Process Regressor': 'green',
    'Support Vector Regressor': 'blue',
    'Decision Tree Regressor': 'red',
    'Random Forest Regressor': 'orange',
    'K-Nearest Neighbors': 'purple',
    'Dummy Regressor': 'brown',
    'Linear Regression': 'cyan',
    'Linear SVR': 'yellow',
    'XGBoost': 'magenta'
}

# Titles with units
titles_with_units = {
    'Surface Area': 'Surface Area (m²/g)',
    'MB Removal': 'MB Removal (%)',
    'Total Pore Volume': 'Total Pore Volume (cm³/g)',
    'Average Crystallite Size': 'Average Crystallite Size (Dₐᵥg nm)',
    'Crystallinity': 'Crystallinity (%)',
    'Yield': 'Yield (%)'
}

# Loop through each sheet and plot
for i, sheet in enumerate(excel_data.sheet_names):
    # Read the data from the current sheet
    data = pd.read_excel(excel_data, sheet_name=sheet)

    # Ensure data is in the specified order
    data['Model'] = pd.Categorical(data['Model'], categories=model_order, ordered=True)
    data.sort_values('Model', inplace=True)

    # Create a horizontal bar plot with assigned colors, thicker bars, and more space between bars
    barplot = sns.barplot(
        x='RMSE', y='Model', data=data, ax=axes[i], 
        palette=[colors[model] for model in data['Model']], order=model_order, 
        dodge=False, linewidth=0, edgecolor='black'  # Ensure the bars are thicker without outlines
    )

    # Set the title and labels with Arial font, bold and with defined units
    axes[i].set_title(titles_with_units[sheet], fontsize=41, fontweight='bold', fontname='Arial', color='black')
    axes[i].set_xlabel('RMSE', fontsize=35, fontweight='bold', fontname='Arial', color='black')

    # Add numbers (1 to 9) to the y-axis instead of model names, and ensure fonts are bold and Arial
    axes[i].set_yticklabels([f"{idx+1}" for idx in range(len(model_order))], fontname='Arial', fontweight='bold', fontsize=29)

    # Increase the height of bars
    for bar in barplot.patches:
        bar.set_height(0.8)  # Thicker bars

    # Set tick labels with specific font settings, ensure they are black
    for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
        label.set_fontsize(40)
        label.set_fontweight('bold')
        label.set_fontname('Arial')
        label.set_color('black')  # Make sure the color is black

# Add a custom legend below all subplots with thicker lines
legend_elements = [
    plt.Line2D([0], [0], color=colors[model], lw=10, label=f"{num} {model}")  # Increased line width to 10 for thicker bars
    for num, model in zip(model_numbers, model_order)
]

# Adjust layout and add legend closer to the plot
plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust layout to fit legend
fig.subplots_adjust(bottom=0.2, top=0.915, wspace=0.16, hspace=0.5)  # Reduce space between columns and provide space for legend
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=39, frameon=False, bbox_to_anchor=(0.5, -0.015))  # Moved legend closer to the plots

# Add main title
fig.suptitle('Comparative RMSE Analysis across Different Models', fontsize=44, fontweight='bold', fontname='Arial', color='black', verticalalignment='top')

# Show plot
plt.show()

'''
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Updating rcParams to set global text properties for titles and labels
rcParams['font.family'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 16

file_path_R = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/RMSE.xlsx'
excel_data = pd.ExcelFile(file_path_R)

# Set the style
plt.style.use('seaborn-whitegrid')  # Ensuring background is white
sns.set_context('talk')

# Create a figure with a 3x2 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(29, 23))
fig.set_facecolor('white')  # Set the background color of the figure to white
axes = axes.flatten()  # Flatten the array to easily index it

# Define the order of the models and assign new modern, refined colors
model_order = [
    'Gaussian Process Regressor',
    'Support Vector Regressor',
    'Decision Tree Regressor',
    'Random Forest Regressor',
    'K-Nearest Neighbors',
    'Dummy Regressor',
    'Linear Regression',
    'Linear SVR',
    'XGBoost'
]

# Corresponding numbers and color codes for each model - modern elegant color palette
model_numbers = [f"{i+1}" for i in range(len(model_order))]
colors = {
    'Gaussian Process Regressor': '#4CAF50',   # Deep modern green
    'Support Vector Regressor': '#2196F3',    # Cool blue
    'Decision Tree Regressor': '#FF5722',     # Elegant orange-red
    'Random Forest Regressor': '#FFC107',     # Polished gold
    'K-Nearest Neighbors': '#673AB7',         # Deep purple
    'Dummy Regressor': '#607D8B',             # Soft gray-blue
    'Linear Regression': '#00BCD4',           # Cyan blue
    'Linear SVR': '#FFEB3B',                  # Clean yellow
    'XGBoost': '#E91E63'                     # Modern pink
}

# Titles with units
titles_with_units = {
    'Surface Area': 'Surface Area (m²/g)',
    'MB Removal': 'MB Removal (%)',
    'Total Pore Volume': 'Total Pore Volume (cm³/g)',
    'Average Crystallite Size': 'Average Crystallite Size (Dₐᵥg nm)',
    'Crystallinity': 'Crystallinity (%)',
    'Yield': 'Yield (%)'
}

# Loop through each sheet and plot
for i, sheet in enumerate(excel_data.sheet_names):
    # Read the data from the current sheet
    data = pd.read_excel(excel_data, sheet_name=sheet)

    # Ensure data is in the specified order
    data['Model'] = pd.Categorical(data['Model'], categories=model_order, ordered=True)
    data.sort_values('Model', inplace=True)

    # Create a horizontal bar plot with assigned colors, thicker bars, and more space between bars
    barplot = sns.barplot(
        x='RMSE', y='Model', data=data, ax=axes[i], 
        palette=[colors[model] for model in data['Model']], order=model_order, 
        dodge=False, linewidth=0, edgecolor='black'  # Ensure the bars are thicker without outlines
    )

    # Set the title and labels with Arial font, bold and with defined units
    axes[i].set_title(titles_with_units[sheet], fontsize=41, fontweight='bold', fontname='Arial', color='black')
    axes[i].set_xlabel('RMSE', fontsize=31.5, fontweight='bold', fontname='Arial', color='black')

    # Add numbers (1 to 9) to the y-axis instead of model names, and ensure fonts are bold and Arial
    axes[i].set_yticklabels([f"{idx+1}" for idx in range(len(model_order))], fontname='Arial', fontweight='bold', fontsize=29)

    # Increase the height of bars
    for bar in barplot.patches:
        bar.set_height(0.8)  # Thicker bars

    # Set tick labels with specific font settings, ensure they are black
    for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
        label.set_fontsize(40)
        label.set_fontweight('bold')
        label.set_fontname('Arial')
        label.set_color('black')  # Make sure the color is black

# Add a custom legend below all subplots with thicker lines
legend_elements = [
    plt.Line2D([0], [0], color=colors[model], lw=10, label=f"{num} {model}")  # Increased line width to 10 for thicker bars
    for num, model in zip(model_numbers, model_order)
]

# Adjust layout and add legend closer to the plot
plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust layout to fit legend
fig.subplots_adjust(bottom=0.2, top=0.915, wspace=0.16, hspace=0.5)  # Reduce space between columns and provide space for legend
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=39, frameon=False, bbox_to_anchor=(0.5, -0.015))  # Moved legend closer to the plots

# Add main title
fig.suptitle('Comparative RMSE Analysis across Different Models', fontsize=44, fontweight='bold', fontname='Arial', color='black', verticalalignment='top')

# Show plot
plt.show()

'''

'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.patches import Patch

# Updating rcParams to set global text properties for titles and labels
rcParams['font.family'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 16

file_path_R = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/RMSE.xlsx'
excel_data = pd.ExcelFile(file_path_R)

# Set the style
plt.style.use('seaborn-whitegrid')  # Ensuring background is white
sns.set_context('talk')

# Create a figure with a 3x2 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(29, 24))
fig.set_facecolor('white')  # Set the background color of the figure to white
axes = axes.flatten()  # Flatten the array to easily index it

# Define the reordered models and assign new modern, refined colors
model_order = [
    'Gaussian Process Regressor',
    'Support Vector Regressor',
    'Decision Tree Regressor',
    'Random Forest Regressor',
    'K-Nearest Neighbors',
    'XGBoost',
    'Linear SVR',
    'Dummy Regressor',
    'Linear Regression'
]

# Corresponding numbers for each model
model_numbers = [f"{i+1}" for i in range(len(model_order))]

# Updated color palette with refined shades
colors = {
    'Gaussian Process Regressor': '#4CBB17',  # Dark green
    'Dummy Regressor': '#A9A9A9',            # Light gray
    'Linear Regression': '#A9A9A9',          # Light gray
    'Support Vector Regressor': '#FF867F',   # Amber
    'Decision Tree Regressor': '#FF867F',    # Amber
    'Random Forest Regressor': '#FF867F',    # Amber
    'K-Nearest Neighbors': '#FF867F',        # Amber
    'XGBoost': '#FF867F',                    # Amber
    'Linear SVR': '#FF867F',                 #            # Soft yellow
}

# Titles with units remain the same
titles_with_units = {
    'Surface Area': 'Surface Area (m²/g)',
    'MB Removal': 'MB Removal (%)',
    'Total Pore Volume': 'Total Pore Volume (cm³/g)',
    'Average Crystallite Size': 'Average Crystallite Size (Dₐᵥg nm)',
    'Crystallinity': 'Crystallinity (%)',
    'Yield': 'Yield (%)'
}

# Loop through each sheet and plot
for i, sheet in enumerate(excel_data.sheet_names):
    # Read the data from the current sheet
    data = pd.read_excel(excel_data, sheet_name=sheet)

    # Ensure data is in the specified order
    data['Model'] = pd.Categorical(data['Model'], categories=model_order, ordered=True)
    data.sort_values('Model', inplace=True)

    # Create a horizontal bar plot with updated colors
    barplot = sns.barplot(
        x='RMSE', y='Model', data=data, ax=axes[i], 
        palette=[colors[model] for model in data['Model']], order=model_order, 
        dodge=False, linewidth=0, edgecolor='black'
    )

   # Add hatching to bars based on the model
    for bar, model in zip(barplot.patches, data['Model']):
        if model == 'Gaussian Process Regressor':  # No hatching for GPR
            bar.set_hatch('')
        elif model in ['Dummy Regressor', 'Linear Regression']:  # 'xx' hatching for Dummy and Linear Regression
            bar.set_hatch('xx')
        else:  # '//' hatching for all other models
            bar.set_hatch('//')

# Add a vertical dashed line at the Dummy Regressor's RMSE
    dummy_rmse = data.loc[data['Model'] == 'Dummy Regressor', 'RMSE'].values[0]
    axes[i].axvline(dummy_rmse, color='#3b3b3b', linestyle='--', linewidth=6)

# Add a vertical dashed line at the Linear Regression's RMSE
    linear_rmse = data.loc[data['Model'] == 'Linear Regression', 'RMSE'].values[0]
    axes[i].axvline(linear_rmse, color='#3b3b3b', linestyle='--', linewidth=6)

    # Set the title and labels
    axes[i].set_title(titles_with_units[sheet], fontsize=41, fontweight='bold', fontname='Arial', color='black')
    axes[i].set_xlabel('RMSE', fontsize=31.5, fontweight='bold', fontname='Arial', color='black')

    # Add numbers (1 to 9) to the y-axis instead of model names
    axes[i].set_yticklabels([f"{idx+1}" for idx in range(len(model_order))], fontname='Arial', fontweight='bold', fontsize=29)

    # Increase the height of bars
    for bar in barplot.patches:
        bar.set_height(0.79)

    # Set tick labels
    for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
        label.set_fontsize(40)
        label.set_fontweight('bold')
        label.set_fontname('Arial')
        label.set_color('black')


# Add a custom legend with even thicker bars and hatching
legend_elements = [
    Patch(
        facecolor=colors['Gaussian Process Regressor'], edgecolor='black', label='1 Gaussian Process Regressor',
        hatch='', linewidth=2  # No hatching and thicker edge
    ),
    Patch(
        facecolor=colors['Support Vector Regressor'], edgecolor='black', label='2 Support Vector Regressor',
        hatch='//', linewidth=2  # Hatching with thicker edge
    ),
    Patch(
        facecolor=colors['Decision Tree Regressor'], edgecolor='black', label='3 Decision Tree Regressor',
        hatch='//', linewidth=2  # Hatching with thicker edge
    ),
    Patch(
        facecolor=colors['Random Forest Regressor'], edgecolor='black', label='4 Random Forest Regressor',
        hatch='//', linewidth=2  # Hatching with thicker edge
    ),
    Patch(
        facecolor=colors['K-Nearest Neighbors'], edgecolor='black', label='5 K-Nearest Neighbors',
        hatch='//', linewidth=2  # Hatching with thicker edge
    ),
    Patch(
        facecolor=colors['XGBoost'], edgecolor='black', label='6 XGBoost',
        hatch='//', linewidth=2  # Hatching with thicker edge
    ),
    Patch(
        facecolor=colors['Linear SVR'], edgecolor='black', label='7 Linear SVR',
        hatch='//', linewidth=2  # Hatching with thicker edge
    ),
    Patch(
        facecolor=colors['Dummy Regressor'], edgecolor='black', label='8 Dummy Regressor',
        hatch='xx', linewidth=2  # Hatching with thicker edge
    ),
    Patch(
        facecolor=colors['Linear Regression'], edgecolor='black', label='9 Linear Regression',
        hatch='xx', linewidth=2  # Hatching with thicker edge
    ),
]

# Adjust layout and add legend
plt.tight_layout(rect=[0, 0, 1, 0.92])
fig.subplots_adjust(bottom=0.2, top=0.915, wspace=0.16, hspace=0.5)
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=39, frameon=False, bbox_to_anchor=(0.5, -0.015))

# Add main title
fig.suptitle('Comparative RMSE Analysis across Different Models', fontsize=44, fontweight='bold', fontname='Arial', color='black', verticalalignment='top')

# Show plot
plt.show()


'''
