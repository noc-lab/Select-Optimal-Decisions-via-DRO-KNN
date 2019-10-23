import sys
sys.path.append('/home/rchen15/prescription/run_pres')
import pickle
import os
import argparse
import numpy as np
from load_table import load_diabetes_final_table_for_prescription, \
    load_hypertension_final_table_for_prescription
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from tqdm import tqdm
from util import build_validation_set_prescription
from util import str2bool
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge

parser = argparse.ArgumentParser()

parser.add_argument("--trial", type=int, default=0)
parser.add_argument("--test_ratio", type=float, default=0.2)
parser.add_argument("--save_dir", type=str, default='../ckpt/hypertension/')
parser.add_argument("--diabetes", type=str2bool, default=False)

args = parser.parse_args()


def find_best_models_each_group(data, estimator_func, parameter_grid):
    """
    find the best model for each group
    :param data: data dict
    :param estimator_func: estimate function
    :param parameter_grid: parameter grid for hyperparameter tuning
    :return: model object
    """
    num_prescription = len(data['train_x'])
    all_models = {'core_model': [], 'submodels': []}
    for i in range(num_prescription):  # group using prescription
        x = data['train_x'][i]
        y = data['train_y'][i]

        # find best model  
        model = estimator_func()
        estimator_search = GridSearchCV(model, parameter_grid,
                                        cv=5, scoring='neg_mean_squared_error',
                                        error_score=0)
                
        estimator_search.fit(x, y)
        all_models['core_model'].append(estimator_search.best_estimator_)
        best_parameter = estimator_search.best_params_

        # fit 100 models 
        submodels = []
        for random_seed in tqdm(range(100)):
            rs = ShuffleSplit(n_splits=1, test_size=.10, random_state=random_seed + 1)
            train_index, _ = rs.split(x).__next__()
            sub_x, sub_y = x[train_index], y[train_index]
    
            model = estimator_func(**best_parameter)
            model.fit(sub_x, sub_y)
            submodels.append(model)
    
        all_models['submodels'].append(submodels)
    return all_models

def find_rf_best_models_each_group(data, estimator_func, parameter_grid):
    """
    find the best model for each group
    :param data: data dict
    :param estimator_func: estimate function
    :param parameter_grid: parameter grid for hyperparameter tuning
    :return: model object
    """
    num_prescription = len(data['train_x'])
    all_models = {'core_model': []}
    for i in range(num_prescription):  # group using prescription
        x = data['train_x'][i]
        y = data['train_y'][i]

        # find best model
        model = estimator_func(kernel='polynomial')
        estimator_search = GridSearchCV(model, parameter_grid,
                                        cv=5, scoring='neg_median_absolute_error',
                                        error_score=0)
                
        estimator_search.fit(x, y)
        all_models['core_model'].append(estimator_search.best_estimator_)

    return all_models


def main():
    model_save_dir = args.save_dir
    trial_number = args.trial
    test_ratio = args.test_ratio
    use_diabetes = args.diabetes

    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)

    # load data
    if use_diabetes:
        train_all_x, train_all_y, train_all_z, train_all_u, test_x, test_y, test_z, test_u = \
            load_diabetes_final_table_for_prescription(trial_number, test_ratio=test_ratio)
    else:
        train_all_x, train_all_y, train_all_z, train_all_u, test_x, test_y, test_z, test_u = \
            load_hypertension_final_table_for_prescription(trial_number, test_ratio=test_ratio)

    # train drlr kNN models for different prescription
    data = build_validation_set_prescription(train_all_x, train_all_y, train_all_u, test_size=0)

    # first fit LASSO
    lasso_param = {'alpha': np.logspace(-3, 2, 8).tolist()}
    lasso_model = find_best_models_each_group(data, estimator_func=Lasso, parameter_grid=lasso_param)
    pickle.dump(lasso_model, open(model_save_dir + 'lasso_trial_' + str(trial_number) + '.pkl', 'wb'))

    # next fit CART
    cart_param = {'max_depth': list(range(10, 50, 5)), 'min_samples_split': list(np.arange(150, 600, 100))}
    cart_model = find_best_models_each_group(data, estimator_func=DecisionTreeRegressor, parameter_grid=cart_param)
    pickle.dump(cart_model, open(model_save_dir + 'cart_trial_' + str(trial_number) + '.pkl', 'wb'))
    
    # Kernel Ridge (for imputation only)
    #rf_param = {'alpha': [1e0, 0.1, 1e-2, 1e-3], 'gamma': np.logspace(-2, 2, 5)}
    #rf_model = find_rf_best_models_each_group(data, estimator_func=KernelRidge, parameter_grid=rf_param)
    #pickle.dump(rf_model, open(model_save_dir + 'rf_trial_' + str(trial_number) + '.pkl', 'wb'))


if __name__ == '__main__':
    main()
