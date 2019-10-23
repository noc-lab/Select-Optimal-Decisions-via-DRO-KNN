#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:47:38 2019

@author: admin
"""

import json 
import pandas as pd


with open('final_results_foo_trial_1.json') as file:
    hypertension_1 = json.load(file)
    
with open('final_results_foo_trial_2.json') as file:
    hypertension_2 = json.load(file)

with open('final_results_foo_trial_3.json') as file:
    hypertension_3 = json.load(file)
    
with open('final_results_foo_trial_4.json') as file:
    hypertension_4 = json.load(file)
    
with open('final_results_foo_trial_5.json') as file:
    hypertension_5 = json.load(file)
    
df = pd.DataFrame([hypertension_1, hypertension_2, hypertension_3, hypertension_4, hypertension_5])
mean_hypertension = dict(df.mean())

print(round(mean_hypertension['lasso_deterministic_use_mix'],2))
print(round(mean_hypertension['lasso_random_use_mix'],2))
print(round(mean_hypertension['cart_deterministic_use_mix'],2))
print(round(mean_hypertension['cart_random_use_mix'],2))
print(round(mean_hypertension['ols_knn_deterministic_use_mix'],2))
print(round(mean_hypertension['ols_knn_random_use_mix'],2))
print(round(mean_hypertension['drlr_knn_deterministic_use_mix'],2))
print(round(mean_hypertension['drlr_knn_random_use_mix'],2))
print(round(mean_hypertension['use_previous_mix'],2))
print(round(mean_hypertension['use_doctor'],2))


std_hypertension = dict(df.std())
print(round(std_hypertension['lasso_deterministic_use_mix'],2))
print(round(std_hypertension['lasso_random_use_mix'],2))
print(round(std_hypertension['cart_deterministic_use_mix'],2))
print(round(std_hypertension['cart_random_use_mix'],2))
print(round(std_hypertension['ols_knn_deterministic_use_mix'],2))
print(round(std_hypertension['ols_knn_random_use_mix'],2))
print(round(std_hypertension['drlr_knn_deterministic_use_mix'],2))
print(round(std_hypertension['drlr_knn_random_use_mix'],2))
print(round(std_hypertension['use_previous_mix'],2))
print(round(std_hypertension['use_doctor'],2))