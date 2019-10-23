import sys
sys.path.append('/home/rchen15/prescription/run_pres')
import pickle
import os
import argparse
import numpy as np
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Memory
from load_table import load_diabetes_final_table_for_prescription, \
    load_hypertension_final_table_for_prescription
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from joblib import Parallel, delayed
from transform import OLSTransformer
from util import build_validation_set_prescription, str2bool

parser = argparse.ArgumentParser()

parser.add_argument("--trial", type=int, default=0)
parser.add_argument("--test_ratio", type=float, default=0.2)
parser.add_argument("--save_dir", type=str, default='../ckpt/hypertension/')
parser.add_argument("--diabetes", type=str2bool, default=False)

args = parser.parse_args()


def fit_ols_knn_submodels(x, y, num_neighbor, random_seed=0):
    """
    fit ols kNN sub models
    :param x: data
    :param y: label
    :param num_neighbor: K
    :param random_seed: random seed
    :return: model
    """
    rs = ShuffleSplit(n_splits=1, test_size=.10, random_state=random_seed + 1)
    train_index, _ = rs.split(x).__next__()
    sub_x, sub_y = x[train_index], y[train_index]
    transformer = OLSTransformer()
    informed_knn = Pipeline(
        [('transformer', transformer), ('knn', KNeighborsRegressor(n_neighbors=int(num_neighbor*np.sqrt(0.9))))])
    informed_knn.fit(sub_x, sub_y)
    return informed_knn


def find_best_ols_knn_parameter_each_group(data):
    '''
    find models of each group with the best beta and K
    :param data: data dict with training-validation split
    :return: best parameter set
    '''
    num_prescription = len(data['train_x'])
    all_parameters = []
    for i in range(num_prescription):
        x = data['train_x'][i]
        y = data['train_y'][i]

        cachedir = mkdtemp()
        memory = Memory(cachedir=cachedir, verbose=0)

        transformer = OLSTransformer()

        informed_knn = Pipeline([
            ('transformer', transformer),
            ('knn', KNeighborsRegressor())
        ], memory=memory)

        num_train = len(x)
        num_sample_point = min((num_train / 2) - 3, 30)
        sr_point = np.linspace(1, np.sqrt(num_train / 2), num_sample_point)
        knn_space = list(set([int(np.square(xx)) for xx in sr_point]))

        parameter_grid = {'knn__n_neighbors': knn_space}

        estimator_search = GridSearchCV(informed_knn, parameter_grid,
                                        cv=5, scoring='neg_mean_squared_error',
                                        error_score=0, refit=False)
        estimator_search.fit(x, y)
        best_parameters = estimator_search.best_params_
        all_parameters.append(best_parameters['knn__n_neighbors'])

        rmtree(cachedir)

    return all_parameters


def obtain_best_ols_knn_model(data, best_parameters):
    '''
    find the best ols kNN model
    :param data: data dict
    :param best_parameters: a list of best k obtained for the model for different prescription
    :return: model dict and transformation list
    '''

    # regress k
    # IMPORTANT: cv use 5-fold, all_n in the training will be int(len(item) / 5 * 4)
    all_n = [np.sqrt(int(len(item) / 5 * 4)) for item in data['train_y']]
    all_k = best_parameters

    all_n = np.array(all_n).reshape([-1, 1])

    # regression
    lm = LinearRegression(fit_intercept=False)
    lm.fit(all_n, all_k)
    rho = lm.coef_[0]

    # build regression models
    model_collections = {'core_model': [], 'submodels': []}
    imputation_parts = {'rho': rho, 'transformer': []}

    num_prescription = len(all_n)
    for i in range(num_prescription):
        x = data['train_x'][i]
        y = data['train_y'][i]

        num_neighbor = int(np.ceil(rho * int(np.sqrt(len(y)))))
        transformer = OLSTransformer()
        informed_knn = Pipeline([('transformer', transformer), ('knn', KNeighborsRegressor(n_neighbors=num_neighbor))])
        informed_knn.fit(x, y)
        model_collections['core_model'].append(informed_knn)
        imputation_parts['transformer'].append(informed_knn.get_params()['steps'][0][1])

        submodels = Parallel(n_jobs=1)(
            delayed(fit_ols_knn_submodels)(x, y, num_neighbor, random_seed) for random_seed in range(100))

        model_collections['submodels'].append(submodels)

    return model_collections, imputation_parts


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
    print('Build Validation set')
    data = build_validation_set_prescription(train_all_x, train_all_y, train_all_u, test_size=0)

    print('Finding best parameters for different groups')
    best_prediction_param = find_best_ols_knn_parameter_each_group(data)

    print('Finding best models and submodels')
    model_collection, model_imputation_conf = obtain_best_ols_knn_model(data, best_prediction_param)
    pickle.dump(model_collection, open(model_save_dir + 'ols_knn_trial_' + str(trial_number) + '.pkl', 'wb'))
    pickle.dump(model_imputation_conf,
                open(model_save_dir + 'ols_knn_impute_trial_' + str(trial_number) + '.pkl', 'wb'))

    print('Finishing trial' + str(trial_number))


if __name__ == '__main__':
    main()
