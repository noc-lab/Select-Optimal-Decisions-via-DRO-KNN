import sys
sys.path.append('/home/rchen15/prescription/run_pres')
import os
import pickle
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
from transform import DRLRTransformer
from joblib import Parallel, delayed
from util import build_validation_set_prescription, str2bool, foo

parser = argparse.ArgumentParser()

parser.add_argument("--trial", type=int, default=0)
parser.add_argument("--test_ratio", type=float, default=0.2)
parser.add_argument("--save_dir", type=str, default='../ckpt/hypertension/')
parser.add_argument("--diabetes", type=str2bool, default=False)

args = parser.parse_args()


def fit_drlr_knn_submodels(x, y, num_neighbor, reg_l2, random_seed=0):
    """
    fit drlr kNN sub-models
    :param x: data
    :param y: label
    :param num_neighbor: K
    :param random_seed: random seed
    :return: model
    """
    rs = ShuffleSplit(n_splits=1, test_size=.10, random_state=random_seed + 1)
    train_index, _ = rs.split(x).__next__()
    sub_x, sub_y = x[train_index], y[train_index]
    transformer = DRLRTransformer(solver='scipy', reg_l1=0, reg_l2=reg_l2)
    informed_knn = Pipeline(
        [('transformer', transformer), ('knn', KNeighborsRegressor(n_neighbors=int(num_neighbor*np.sqrt(0.9)), weights = foo))])
    informed_knn.fit(sub_x, sub_y)
    return informed_knn


def find_best_parameter_each_group(data):
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

        transformer = DRLRTransformer(solver='scipy')

        informed_knn = Pipeline([
            ('transformer', transformer),
            ('knn', KNeighborsRegressor(weights = foo))
        ], memory=memory)

        num_train = len(x)
        num_sample_point = min((num_train / 2) - 3, 30)
        sr_point = np.linspace(1, np.sqrt(num_train / 2), num_sample_point)
        knn_space = list(set([int(np.square(xx)) for xx in sr_point]))
        
        pp = np.sqrt(np.linalg.norm(np.dot(np.transpose(x),y), ord=1)/num_train)
        reg_l2_space = np.logspace(np.log(pp/200), np.log(pp), 10, base = np.exp(1))
       

        parameter_grid = {'transformer__reg_l1': [0],
                          'transformer__reg_l2': reg_l2_space,
                          'knn__n_neighbors': knn_space}

        estimator_search = GridSearchCV(informed_knn, parameter_grid,
                                        cv=5, scoring='neg_median_absolute_error',
                                        error_score=0, refit=False)
        estimator_search.fit(x, y)
        best_parameters = estimator_search.best_params_
        all_parameters.append([best_parameters['transformer__reg_l2'], best_parameters['knn__n_neighbors']])

        rmtree(cachedir)

    return all_parameters


def obtain_best_model(data, best_parameters):
    """
    get the best drlr knn model
    :param data: data dict
    :param best_parameters: a list of best k obtained for the model for different prescription
    :return: model dict and transformation list
    """
    # regress k
    # IMPORTANT: cv use 5-fold, all_n in the training will be int(len(item) / 5 * 4)
    all_n = [np.sqrt(int(len(item) / 5 * 4)) for item in data['train_y']]
    all_k = [item[1] for item in best_parameters]

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
        transformer = DRLRTransformer(solver='scipy', reg_l1=0, reg_l2=best_parameters[i][0])
        informed_knn = Pipeline([('transformer', transformer), ('knn', KNeighborsRegressor(n_neighbors=num_neighbor, weights = foo))])
        informed_knn.fit(x, y)
        model_collections['core_model'].append(informed_knn)
        imputation_parts['transformer'].append(informed_knn.get_params()['steps'][0][1])

        submodels = Parallel(n_jobs=-1)(
            delayed(fit_drlr_knn_submodels)(x, y, num_neighbor, best_parameters[i][0], random_seed) for random_seed in
            range(100))

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
    best_prediction_param = find_best_parameter_each_group(data)

    print('Finding best models and submodels')
    model_collection, model_imputation_conf = obtain_best_model(data, best_prediction_param)

    pickle.dump(model_collection, open(model_save_dir + 'drlr_knn_foo_trial_' + str(trial_number) + '.pkl', 'wb'))
    pickle.dump(model_imputation_conf,
                open(model_save_dir + 'drlr_knn_foo_impute_trial_' + str(trial_number) + '.pkl', 'wb'))

    print('Finishing trial' + str(trial_number))


if __name__ == '__main__':
    main()
