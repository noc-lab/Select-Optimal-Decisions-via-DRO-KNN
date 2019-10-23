import sys
sys.path.append('/home/rchen15/prescription/run_pres')
import pickle
import json
import argparse
import numpy as np
from load_table import load_diabetes_final_table_for_prescription, \
    load_hypertension_final_table_for_prescription

from util import get_impute_outcome, get_rf_impute_outcome, return_prediction_and_std, get_boltzman_policy, \
    eval_prescription_probability, find_prescription_threshold, str2bool

parser = argparse.ArgumentParser()

parser.add_argument("--trial", type=int, default=0)
parser.add_argument("--test_ratio", type=float, default=0.2)
parser.add_argument("--save_dir", type=str, default='../ckpt/hypertension/')
parser.add_argument("--diabetes", type=str2bool, default=False)

args = parser.parse_args()


def eval_random_policy(use_previous, T, y_pred, imputed_outcome, epsilon, previous_value):
    """
    evaluate the random prescription policy
    :param use_previous: outcome that using previous prescription
    :param T: if use random policy or previous one
    :param y_pred: predicted outcome for a patient using different prescription
    :param imputed_outcome: imputed outcome for a patient using different prescription
    :param epsilon: soften factor
    :param previous_value: previous bp or a1c
    :return: outcome
    """
    probability = get_boltzman_policy(y_pred, epsilon)
    random_outcome = eval_prescription_probability(probability, imputed_outcome)

    final_outcome = random_outcome * T + use_previous * (1 - T)

    return np.mean(final_outcome - previous_value)


def eval_deterministic_policy(use_previous, T, y_pred, imputed_outcome, previous_value):
    """
    evaluate the deterministic prescription policy
    :param use_previous: outcome that using previous prescription
    :param T: if use deterministic policy or previous one
    :param y_pred: predicted outcome for a patient using different prescription
    :param imputed_outcome: imputed outcome for a patient using different prescription
    :param previous_value: previous bp or a1c
    :return: outcome
    """
    prescription_rule = np.argmin(y_pred, axis=1)
    policy_outcome = np.array([imputed_outcome[i, prescription_rule[i]] for i in range(len(imputed_outcome))])

    final_outcome = policy_outcome * T + use_previous * (1 - T)

    return np.mean(final_outcome - previous_value)


def main():
    model_save_dir = args.save_dir
    trial_number = args.trial
    test_ratio = args.test_ratio
    use_diabetes = args.diabetes

    all_result_dict = {}

    # load data
    if use_diabetes:
        train_all_x, train_all_y, train_all_z, train_all_u, test_x, test_y, test_z, test_u = \
            load_diabetes_final_table_for_prescription(trial_number, test_ratio=test_ratio)
    else:
        train_all_x, train_all_y, train_all_z, train_all_u, test_x, test_y, test_z, test_u = \
            load_hypertension_final_table_for_prescription(trial_number, test_ratio=test_ratio)

    test_u = np.concatenate(test_u, axis=0) # previous prescription

    # OLS imputation model
    ols_knn_impute = pickle.load(open(model_save_dir + 'ols_knn_impute_trial_' + str(trial_number) + '.pkl', 'rb'))
    _, imputed_outcome_ols = get_impute_outcome(test_x, test_y, ols_knn_impute)
    use_previous_ols = np.array([imputed_outcome_ols[i, test_u[i]] for i in range(len(imputed_outcome_ols))])

    # DRLR imputation model
    drlr_knn_impute = pickle.load(open(model_save_dir + 'drlr_knn_foo_impute_trial_' + str(trial_number) + '.pkl', 'rb'))
    test_x_flatten, imputed_outcome_drlr = get_impute_outcome(test_x, test_y, drlr_knn_impute)
    use_previous_drlr = np.array([imputed_outcome_drlr[i, test_u[i]] for i in range(len(imputed_outcome_drlr))])
    
    # Mixture of DRLR and OLS as imputation model
    imputed_outcome_mix = 0.5*imputed_outcome_ols+0.5*imputed_outcome_drlr
    use_previous_mix = np.array([imputed_outcome_mix[i, test_u[i]] for i in range(len(imputed_outcome_mix))])
    
    # Random Forest imputation model
    #rf_impute = pickle.load(open(model_save_dir + 'rf_trial_' + str(trial_number) + '.pkl', 'rb'))
    #_, imputed_outcome_rf = get_rf_impute_outcome(test_x, test_y, rf_impute)
    #use_previous_rf = np.array([imputed_outcome_rf[i, test_u[i]] for i in range(len(imputed_outcome_rf))])

    best_epsilon = json.load(open(model_save_dir + 'best_random_foo_trial_' + str(trial_number) + '.json', 'r'))
    previous_value = test_x_flatten[:, 0]

    # get prescription model evaluated
    # 1. OLS-kNN
    ols_model = pickle.load(open(model_save_dir + 'ols_knn_trial_' + str(trial_number) + '.pkl', 'rb'))
    ols_y_mean, ols_y_std, ols_y = return_prediction_and_std(test_x_flatten, ols_model)
    T_random_ols = find_prescription_threshold(ols_y_mean, ols_y_std, test_x_flatten[:, 0], 1, get_boltzman_policy(ols_y, best_epsilon['ols_knn_use_ols_knn']),ols_y)
    T_random_drlr = find_prescription_threshold(ols_y_mean, ols_y_std, test_x_flatten[:, 0], 1, get_boltzman_policy(ols_y, best_epsilon['ols_knn_use_drlr_knn']),ols_y)
    T_random_mix = find_prescription_threshold(ols_y_mean, ols_y_std, test_x_flatten[:, 0], 1, get_boltzman_policy(ols_y, best_epsilon['ols_knn_use_mix']),ols_y)
    #T_random_rf = find_prescription_threshold(ols_y_mean, ols_y_std, test_x_flatten[:, 0], 1, get_boltzman_policy(ols_y, best_epsilon['ols_knn_use_rf']),ols_y)
    T_fix = find_prescription_threshold(ols_y_mean, ols_y_std, test_x_flatten[:, 0], 0, get_boltzman_policy(ols_y, best_epsilon['ols_knn_use_drlr_knn']),ols_y)


    all_result_dict['ols_knn_random_use_ols_knn'] = \
        eval_random_policy(use_previous_ols, T_random_ols, ols_y, imputed_outcome_ols, best_epsilon['ols_knn_use_ols_knn'],
                           previous_value)
    all_result_dict['ols_knn_deterministic_use_ols_knn'] = \
        eval_deterministic_policy(use_previous_ols, T_fix, ols_y, imputed_outcome_ols, previous_value)

    all_result_dict['ols_knn_random_use_drlr_knn'] = \
        eval_random_policy(use_previous_drlr, T_random_drlr, ols_y, imputed_outcome_drlr, best_epsilon['ols_knn_use_drlr_knn'],
                           previous_value)
    all_result_dict['ols_knn_deterministic_use_drlr_knn'] = \
        eval_deterministic_policy(use_previous_drlr, T_fix, ols_y, imputed_outcome_drlr, previous_value)
        
    all_result_dict['ols_knn_random_use_mix'] = \
        eval_random_policy(use_previous_mix, T_random_mix, ols_y, imputed_outcome_mix, best_epsilon['ols_knn_use_mix'],
                           previous_value)
    all_result_dict['ols_knn_deterministic_use_mix'] = \
        eval_deterministic_policy(use_previous_mix, T_fix, ols_y, imputed_outcome_mix, previous_value)
        
    #all_result_dict['ols_knn_random_use_rf'] = \
        #eval_random_policy(use_previous_rf, T_random_rf, ols_y, imputed_outcome_rf, best_epsilon['ols_knn_use_rf'],
                           #previous_value)
    #all_result_dict['ols_knn_deterministic_use_rf'] = \
        #eval_deterministic_policy(use_previous_rf, T_fix, ols_y, imputed_outcome_rf, previous_value)

    # 2. DRLR-kNN
    drlr_model = pickle.load(open(model_save_dir + 'drlr_knn_foo_trial_' + str(trial_number) + '.pkl', 'rb'))
    drlr_y_mean, drlr_y_std, drlr_y = return_prediction_and_std(test_x_flatten, drlr_model)
    
    T_random_ols = find_prescription_threshold(drlr_y_mean, drlr_y_std, test_x_flatten[:, 0], 1, get_boltzman_policy(drlr_y, best_epsilon['drlr_knn_use_ols_knn']),drlr_y)
    T_random_drlr = find_prescription_threshold(drlr_y_mean, drlr_y_std, test_x_flatten[:, 0], 1, get_boltzman_policy(drlr_y, best_epsilon['drlr_knn_use_drlr_knn']),drlr_y)
    T_random_mix = find_prescription_threshold(drlr_y_mean, drlr_y_std, test_x_flatten[:, 0], 1, get_boltzman_policy(drlr_y, best_epsilon['drlr_knn_use_mix']),drlr_y)
    #T_random_rf = find_prescription_threshold(drlr_y_mean, drlr_y_std, test_x_flatten[:, 0], 1, get_boltzman_policy(drlr_y, best_epsilon['drlr_knn_use_rf']),drlr_y)
    
    T_fix = find_prescription_threshold(drlr_y_mean, drlr_y_std, test_x_flatten[:, 0], 0, get_boltzman_policy(drlr_y, best_epsilon['drlr_knn_use_drlr_knn']),drlr_y)


    all_result_dict['drlr_knn_random_use_ols_knn'] = \
        eval_random_policy(use_previous_ols, T_random_ols, drlr_y, imputed_outcome_ols, best_epsilon['drlr_knn_use_ols_knn'],
                           previous_value)
    all_result_dict['drlr_knn_deterministic_use_ols_knn'] = \
        eval_deterministic_policy(use_previous_ols, T_fix, drlr_y, imputed_outcome_ols, previous_value)

    all_result_dict['drlr_knn_random_use_drlr_knn'] = \
        eval_random_policy(use_previous_drlr, T_random_drlr, drlr_y, imputed_outcome_drlr, best_epsilon['drlr_knn_use_drlr_knn'],
                           previous_value)
    all_result_dict['drlr_knn_deterministic_use_drlr_knn'] = \
        eval_deterministic_policy(use_previous_drlr, T_fix, drlr_y, imputed_outcome_drlr, previous_value)
        
    all_result_dict['drlr_knn_random_use_mix'] = \
        eval_random_policy(use_previous_mix, T_random_mix, drlr_y, imputed_outcome_mix, best_epsilon['drlr_knn_use_mix'],
                           previous_value)
    all_result_dict['drlr_knn_deterministic_use_mix'] = \
        eval_deterministic_policy(use_previous_mix, T_fix, drlr_y, imputed_outcome_mix, previous_value)
        
    #all_result_dict['drlr_knn_random_use_rf'] = \
        #eval_random_policy(use_previous_rf, T_random_rf, drlr_y, imputed_outcome_rf, best_epsilon['drlr_knn_use_rf'],
                           #previous_value)
    #all_result_dict['drlr_knn_deterministic_use_rf'] = \
        #eval_deterministic_policy(use_previous_rf, T_fix, drlr_y, imputed_outcome_rf, previous_value)

    # 3. LASSO
    lasso_model = pickle.load(open(model_save_dir + 'lasso_trial_' + str(trial_number) + '.pkl', 'rb'))
    lasso_y_mean, lasso_y_std, lasso_y = return_prediction_and_std(test_x_flatten, lasso_model)
    #T = find_prescription_threshold(lasso_y, lasso_y_std, test_x_flatten[:, 0])
    
    T_random_ols = find_prescription_threshold(lasso_y_mean, lasso_y_std, test_x_flatten[:, 0], 1, get_boltzman_policy(lasso_y, best_epsilon['lasso_use_ols_knn']),lasso_y)
    T_random_drlr = find_prescription_threshold(lasso_y_mean, lasso_y_std, test_x_flatten[:, 0], 1, get_boltzman_policy(lasso_y, best_epsilon['lasso_use_drlr_knn']),lasso_y)
    T_random_mix = find_prescription_threshold(lasso_y_mean, lasso_y_std, test_x_flatten[:, 0], 1, get_boltzman_policy(lasso_y, best_epsilon['lasso_use_mix']),lasso_y)
    #T_random_rf = find_prescription_threshold(lasso_y_mean, lasso_y_std, test_x_flatten[:, 0], 1, get_boltzman_policy(lasso_y, best_epsilon['lasso_use_rf']),lasso_y)
    T_fix = find_prescription_threshold(lasso_y_mean, lasso_y_std, test_x_flatten[:, 0], 0, get_boltzman_policy(lasso_y, best_epsilon['lasso_use_drlr_knn']),lasso_y)

    all_result_dict['lasso_random_use_ols_knn'] = \
        eval_random_policy(use_previous_ols, T_random_ols, lasso_y, imputed_outcome_ols, best_epsilon['lasso_use_ols_knn'],
                           previous_value)
    all_result_dict['lasso_deterministic_use_ols_knn'] = \
        eval_deterministic_policy(use_previous_ols, T_fix, lasso_y, imputed_outcome_ols, previous_value)

    all_result_dict['lasso_random_use_drlr_knn'] = \
        eval_random_policy(use_previous_drlr, T_random_drlr, lasso_y, imputed_outcome_drlr, best_epsilon['lasso_use_drlr_knn'],
                           previous_value)
    all_result_dict['lasso_deterministic_use_drlr_knn'] = \
        eval_deterministic_policy(use_previous_drlr, T_fix, lasso_y, imputed_outcome_drlr, previous_value)
        
    all_result_dict['lasso_random_use_mix'] = \
        eval_random_policy(use_previous_mix, T_random_mix, lasso_y, imputed_outcome_mix, best_epsilon['lasso_use_mix'],
                           previous_value)
    all_result_dict['lasso_deterministic_use_mix'] = \
        eval_deterministic_policy(use_previous_mix, T_fix, lasso_y, imputed_outcome_mix, previous_value)
        
   # all_result_dict['lasso_random_use_rf'] = \
       # eval_random_policy(use_previous_rf, T_random_rf, lasso_y, imputed_outcome_rf, best_epsilon['lasso_use_rf'],
                           #previous_value)
    #all_result_dict['lasso_deterministic_use_rf'] = \
       # eval_deterministic_policy(use_previous_rf, T_fix, lasso_y, imputed_outcome_rf, previous_value)

    # 4. CART
    cart_model = pickle.load(open(model_save_dir + 'cart_trial_' + str(trial_number) + '.pkl', 'rb'))
    cart_y_mean, cart_y_std, cart_y = return_prediction_and_std(test_x_flatten, cart_model)
    #T = find_prescription_threshold(cart_y, cart_y_std, test_x_flatten[:, 0])
    
    T_random_ols = find_prescription_threshold(cart_y_mean, cart_y_std, test_x_flatten[:, 0], 1, get_boltzman_policy(cart_y, best_epsilon['cart_use_ols_knn']),cart_y)
    T_random_drlr = find_prescription_threshold(cart_y_mean, cart_y_std, test_x_flatten[:, 0], 1, get_boltzman_policy(cart_y, best_epsilon['cart_use_drlr_knn']),cart_y)
    T_random_mix = find_prescription_threshold(cart_y_mean, cart_y_std, test_x_flatten[:, 0], 1, get_boltzman_policy(cart_y, best_epsilon['cart_use_mix']),cart_y)
    #T_random_rf = find_prescription_threshold(cart_y_mean, cart_y_std, test_x_flatten[:, 0], 1, get_boltzman_policy(cart_y, best_epsilon['cart_use_rf']),cart_y)
    T_fix = find_prescription_threshold(cart_y_mean, cart_y_std, test_x_flatten[:, 0], 0, get_boltzman_policy(cart_y, best_epsilon['cart_use_drlr_knn']),cart_y)

    all_result_dict['cart_random_use_ols_knn'] = \
        eval_random_policy(use_previous_ols, T_random_ols, cart_y, imputed_outcome_ols, best_epsilon['cart_use_ols_knn'],
                           previous_value)
    all_result_dict['cart_deterministic_use_ols_knn'] = \
        eval_deterministic_policy(use_previous_ols, T_fix, cart_y, imputed_outcome_ols, previous_value)

    all_result_dict['cart_random_use_drlr_knn'] = \
        eval_random_policy(use_previous_drlr, T_random_drlr, cart_y, imputed_outcome_drlr, best_epsilon['cart_use_drlr_knn'],
                           previous_value)
    all_result_dict['cart_deterministic_use_drlr_knn'] = \
        eval_deterministic_policy(use_previous_drlr, T_fix, cart_y, imputed_outcome_drlr, previous_value)
        
    all_result_dict['cart_random_use_mix'] = \
        eval_random_policy(use_previous_mix, T_random_mix, cart_y, imputed_outcome_mix, best_epsilon['cart_use_mix'],
                           previous_value)
    all_result_dict['cart_deterministic_use_mix'] = \
        eval_deterministic_policy(use_previous_mix, T_fix, cart_y, imputed_outcome_mix, previous_value)
    
    #all_result_dict['cart_random_use_rf'] = \
        #eval_random_policy(use_previous_rf, T_random_rf, cart_y, imputed_outcome_rf, best_epsilon['cart_use_rf'],
                           #previous_value)
    #all_result_dict['cart_deterministic_use_rf'] = \
       # eval_deterministic_policy(use_previous_rf, T_fix, cart_y, imputed_outcome_rf, previous_value)

    all_result_dict['use_previous_ols_knn'] = np.mean(use_previous_ols - previous_value) # if use previous prescription
    all_result_dict['use_previous_drlr_knn'] = np.mean(use_previous_drlr - previous_value)
    all_result_dict['use_previous_mix'] = np.mean(use_previous_mix - previous_value)
    #all_result_dict['use_previous_rf'] = np.mean(use_previous_rf - previous_value)
    all_result_dict['use_doctor'] = np.mean(np.concatenate(test_y, axis=0) - previous_value)

    for key in all_result_dict.keys():
        all_result_dict[key] = float(all_result_dict[key])

    json.dump(all_result_dict, open(model_save_dir + 'final_results_foo_trial_' + str(trial_number) + '.json', 'w'),
              indent=True)


if __name__ == '__main__':
    main()
