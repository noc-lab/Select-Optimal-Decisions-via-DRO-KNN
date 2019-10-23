import sys
sys.path.append('/home/rchen15/prescription/run_pres')
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from collections import Counter
from util import get_base_path


def load_diabetes_final_table_for_prescription(trial_id, test_ratio=0.2):
    """
    load preprocess diabetes data
    :param trial_id: trial id
    :param test_ratio: ratio of test data
    :return: train_all_x, train_all_y, train_all_z, train_all_u, test_x, test_y, test_z, test_u
    """
    df = pd.read_pickle('/home/rchen15/prescription/presription_shared/diabetes.p')

    prescription_columns = ['prescription_oral', 'prescription_injectable']
    hist_pres_columns = ['hist_prescription_oral', 'hist_prescription_injectable']
    useful_feature = [item for item in df.columns.tolist() if
                      item not in prescription_columns and item != 'future_a1c']

    X = np.array(df[useful_feature].values, dtype=np.float32)
    y = np.array(df['future_a1c'].values, dtype=np.float32)
    z = np.array(df[prescription_columns].values, dtype=int)
    u = np.array(df[hist_pres_columns].values, dtype=int)

    z = np.array(z[:, 0] + 2 * z[:, 1], dtype=int)
    u = np.array(u[:, 0] + 2 * u[:, 1], dtype=int)

    train_all_x = []
    train_all_y = []
    train_all_z = []
    train_all_u = []

    test_x = []
    test_y = []
    test_z = []
    test_u = []

    for pres_id in range(4):
        valid_id = z == pres_id

        this_X = X[valid_id]
        this_y = y[valid_id]
        this_z = z[valid_id]
        this_u = u[valid_id]

        rs = ShuffleSplit(n_splits=1, test_size=test_ratio, random_state=trial_id)
        train_index, test_index = rs.split(this_X).__next__()
        X_train_all, X_test = this_X[train_index], this_X[test_index]
        y_train_all, y_test = this_y[train_index], this_y[test_index]
        z_train_all, z_test = this_z[train_index], this_z[test_index]
        u_train_all, u_test = this_u[train_index], this_u[test_index]

        train_all_x.append(X_train_all)
        train_all_y.append(y_train_all)
        train_all_z.append(z_train_all)
        train_all_u.append(u_train_all)

        test_x.append(X_test)
        test_y.append(y_test)
        test_z.append(z_test)
        test_u.append(u_test)

    return train_all_x, train_all_y, train_all_z, train_all_u, test_x, test_y, test_z, test_u


def load_hypertension_final_table_for_prescription(trial_id, test_ratio=0.2):
    """
    load preprocess hypertension data
    :param trial_id: trial id
    :param test_ratio: ratio of test data
    :return: train_all_x, train_all_y, train_all_z, train_all_u, test_x, test_y, test_z, test_u
    """
    df = pd.read_pickle('/home/rchen15/prescription/presription_shared/hypertension.p')
    not_use_columns = ['measure_systolic_future', 'visits_in_regimen', 'measure_height',
                       'measure_o2_saturation', 'measure_respiratory_rate', 'measure_temperature',
                       'measure_weight', 'diag_042', 'diag_070', 'diag_110', 'diag_174',
                       'diag_185', 'hist_prescription_ACEI', 'hist_prescription_ARB',
                       'hist_prescription_AlphaBlocker', 'hist_prescription_BetaBlocker',
                       'hist_prescription_CCB', 'hist_prescription_Diuretics']
    prescription_columns = ['prescription_ACEI', 'prescription_ARB', 'prescription_AlphaBlocker',
                            'prescription_BetaBlocker', 'prescription_CCB', 'prescription_Diuretics']
    hist_pres_columns = ['hist_prescription_ACEI', 'hist_prescription_ARB', 'hist_prescription_AlphaBlocker',
                         'hist_prescription_BetaBlocker', 'hist_prescription_CCB', 'hist_prescription_Diuretics']
    useful_feature = [item for item in df.columns.tolist()
                      if item not in not_use_columns and item not in prescription_columns]

    X = np.array(df[useful_feature].values, dtype=np.float32)
    y = np.array(df['measure_systolic_future'].values, dtype=np.float32)
    z = np.array(df[prescription_columns].values, dtype=int)
    u = np.array(df[hist_pres_columns].values, dtype=int)

    z_c = z[:, 0] + 2 * z[:, 1] + 4 * z[:, 2] + 8 * z[:, 3] + 16 * z[:, 4] + 32 * z[:, 5]
    z_c = np.asanyarray(z_c, dtype=int)

    u_c = u[:, 0] + 2 * u[:, 1] + 4 * u[:, 2] + 8 * u[:, 3] + 16 * u[:, 4] + 32 * u[:, 5]
    u_c = np.asanyarray(u_c, dtype=int)

    commom_19 = [item[0] for item in Counter(z_c).most_common(19)]
    new_id = {item: item_id for item_id, item in enumerate(commom_19)}
    for i in range(64):
        if i not in new_id.keys():
            new_id[i] = 19

    z = np.array([new_id[item] for item in z_c], dtype=int)
    u = np.array([new_id[item] for item in u_c], dtype=int)

    train_all_x = []
    train_all_y = []
    train_all_z = []
    train_all_u = []

    test_x = []
    test_y = []
    test_z = []
    test_u = []

    for pres_id in range(20):
        valid_id = z == pres_id
        this_X = X[valid_id]
        this_y = y[valid_id]
        this_z = z[valid_id]
        this_u = u[valid_id]

        rs = ShuffleSplit(n_splits=1, test_size=test_ratio, random_state=trial_id)
        train_index, test_index = rs.split(this_X).__next__()
        X_train_all, X_test = this_X[train_index], this_X[test_index]
        y_train_all, y_test = this_y[train_index], this_y[test_index]
        z_train_all, z_test = this_z[train_index], this_z[test_index]
        u_train_all, u_test = this_u[train_index], this_u[test_index]

        train_all_x.append(X_train_all)
        train_all_y.append(y_train_all)
        train_all_z.append(z_train_all)
        train_all_u.append(u_train_all)

        test_x.append(X_test)
        test_y.append(y_test)
        test_z.append(z_test)
        test_u.append(u_test)

    return train_all_x, train_all_y, train_all_z, train_all_u, test_x, test_y, test_z, test_u
