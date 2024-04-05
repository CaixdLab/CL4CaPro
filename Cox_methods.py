import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from lifelines import CoxPHFitter
from sksurv.metrics import concordance_index_censored, integrated_brier_score
import numpy as np
import concurrent.futures
from pycox.evaluation import EvalSurv
from xgbse.converters import (
    convert_data_to_xgb_format,
    convert_to_structured
)
from sksurv.metrics import integrated_brier_score
from xgbse.metrics import concordance_index
import xgboost as xgb
from scipy.interpolate import interp1d
from cox_nnet import *
from Cal_IBS import BreslowEstimator
'''
For cox_nnet we reference in the 
Github repo: https://github.com/xinformatics/coxnnet_py3/tree/master 
Citation: Ching, Travers, Xun Zhu, and Lana X. Garmire. "Cox-nnet: an artificial neural network method for prognosis prediction of high-throughput omics data." PLoS computational biology 14.4 (2018): e1006076.
'''

def parse_option():
    parser = argparse.ArgumentParser('argument for analysis features extracted from trained models')

    parser.add_argument('--cancer_group', type=str, default='SM',
                        help='cancer group (SM/NGT/MSE/CCPRC/HC/GG/DS/LC)')
    parser.add_argument('--core', type=int, default=10,
                        help='cores reserved to run the script')
    parser.add_argument('--pooled', type=bool, default=False,
                        help='whether pooled the cancer group')
    parser.add_argument('--os_flag', type=bool, default=False,
                        help='whether using OS time')

    opt = parser.parse_args()

    return opt


def estimate_baseline_survival(train_preds, y):

    # Extract event times and event indicators from y_test
    event_times = y['Survival_in_days']
    event_occurred = y['Status']

    # Sort the data by time
    sorted_indices = np.argsort(event_times)
    sorted_times = event_times[sorted_indices]
    sorted_events = event_occurred[sorted_indices]
    sorted_preds = train_preds[sorted_indices]

    # Unique event times
    unique_times = np.unique(sorted_times)

    # Initialize the cumulative hazard
    cumulative_hazard = np.zeros_like(unique_times, dtype=float)

    # Calculate the cumulative baseline hazard
    for i, t in enumerate(unique_times):
        at_risk = sorted_times >= t
        risk_sum = np.sum(np.exp(sorted_preds[at_risk]))
        event_sum = np.sum(sorted_events[sorted_times == t])
        cumulative_hazard[i] = cumulative_hazard[i - 1] + event_sum / risk_sum if i > 0 else event_sum / risk_sum

    # Estimate the baseline survival function
    baseline_survival = np.exp(-cumulative_hazard)

    # Create a function to interpolate survival values for given times
    from scipy.interpolate import interp1d
    survival_function = interp1d(unique_times, baseline_survival, kind='previous', bounds_error=False, fill_value="extrapolate")

    return survival_function


# Function to interpolate cumulative hazard and convert to survival probabilities
def calculate_individual_survival_at_timepoints(Si, time_points):
    survival_probabilities = np.zeros((len(Si), len(time_points)))
    for i, s_func in enumerate(Si):
        for j, t in enumerate(time_points):
            survival_probabilities[i, j] = s_func(t)  # Calculate survival probability at time t
    return survival_probabilities

def build_y_data(y_data):
    y_time = y_data.iloc[:, 0].values.tolist()
    y_sta = y_data.iloc[:, 1].values.tolist()
    y = []
    for k in range(len(y_time)):
        y.append([y_sta[k], y_time[k]])

    # List of tuples
    aux = [(e1, e2) for e1, e2 in y]

    # Structured array
    new_data_y = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    return new_data_y

def process_each_pth(item, path, seed, cancer_group, pooled, method = 'coxEN'):
    store_pth = os.path.join(path, '{}_ibs_bench'.format(cancer_group))
    ana_list = []
    print('Processing:', item)
    data = pd.read_csv(path + item)

    # split x and y for cox regression
    if cancer_group == 'SM':
        CancerTypeList = ['BLCA', 'CESC', 'ESCA', 'HNSC', 'LUSC']
    elif cancer_group == 'NGT':
        CancerTypeList = ['GBM', 'LGG', 'PCPG']
    elif cancer_group == 'MSE':
        CancerTypeList = ['SKCM', 'UVM']
    elif cancer_group == 'CCPRC':
        CancerTypeList = ['KIRC', 'KIRP']
    elif cancer_group == 'HC':
        CancerTypeList = ['CHOL', 'LIHC']
    elif cancer_group == 'GG':
        CancerTypeList = ['COAD', 'READ', 'ESCA', 'STAD']
    elif cancer_group == 'DS':
        CancerTypeList = ['PAAD', 'STAD']
    elif cancer_group == 'LC':
        CancerTypeList = ['LUAD', 'LUSC']
    else:
        CancerTypeList = [cancer_group]

    for eachCancer in CancerTypeList:
        get_data = data.loc[data['gen_id'] == eachCancer]
        rest_data = data.loc[data['gen_id'] != eachCancer]  # get rest data samples
        get_data.sort_values(by=['PFItime'])
        data_len = get_data.shape[0]
        start = 0
        if len(get_data) < 80:
            piece_num = 5
        else:
            piece_num = 20

        end = start + int(data_len / piece_num)
        x_train_get = []
        y_train_get = []
        x_test_get = []
        y_test_get = []

        if pooled:
            labely = rest_data[['PFItime', 'PFI']]
            labelx = rest_data.iloc[:, 6:]
            x_train_get.append(labelx)
            y_train_get.append(labely)

        for i in range(piece_num):
            if i == piece_num - 1:
                end = data_len
            data_sample = get_data.iloc[start:end]
            start = end
            end = end + int(data_len / piece_num)
            labely = data_sample[['PFItime', 'PFI']]
            labelx = data_sample.iloc[:, 6:]
            x_train, x_test, y_train, y_test = train_test_split(labelx, labely, test_size=0.2,
                                                                random_state=seed)  # Pass seed 10, 74, 341, 925, 1036 #114
            x_train_get.append(x_train)
            y_train_get.append(y_train)
            x_test_get.append(x_test)
            y_test_get.append(y_test)

        x_train_combine = pd.concat(x_train_get)
        y_train_combine = pd.concat(y_train_get)
        x_test_combine = pd.concat(x_test_get)
        y_test_combine = pd.concat(y_test_get)

        x = x_train_combine
        x_test = x_test_combine

        y = build_y_data(y_train_combine)
        y_test = build_y_data(y_test_combine)

        print(item, ' Generate: ', eachCancer)

        # Calculate the time points
        upper = min(max(y_test['Survival_in_days']), max(y['Survival_in_days']))
        bottom = max(min(y_test['Survival_in_days']), min(y['Survival_in_days']))

        sorted_array = np.sort(y_test['Survival_in_days'])

        # For 90th percentile
        index_end_point_90 = int(0.9 * len(sorted_array))
        end_point_90 = sorted_array[index_end_point_90]
        times_90 = np.arange(bottom + 1, end_point_90)

        # For 80th percentile
        index_end_point_80 = int(0.8 * len(sorted_array))
        end_point_80 = sorted_array[index_end_point_80]
        times_80 = np.arange(bottom + 1, end_point_80)

        print(str(seed), bottom, upper, index_end_point_90, index_end_point_80)

        if method == 'coxEN':

            estimator = CoxnetSurvivalAnalysis(fit_baseline_model=True,
                                               normalize=True)  # alphas=alpha_get
            estimator.fit(x, y)
            scores = concordance_index_censored(y_test['Status'], y_test['Survival_in_days'], estimator.predict(x_test))
            c_index = round(scores[0], 6)

            train_preds = estimator.predict(x)

            baseline_model = BreslowEstimator().fit(train_preds, y['Status'], y['Survival_in_days'])
            survs = baseline_model.get_survival_function(test_pred)
            preds = np.asarray([[fn(t) for t in times] for fn in survs])
            scores = integrated_brier_score(y, y_test, preds, times)
            ibs = round(scores[0], 6)

            print(c_index, ibs)
            ana_list.append(['_'.join(item.split('_')[:-1]) + '_' + str(seed), eachCancer, c_index, ibs])

        elif method == 'coxxgb':

            PARAMS_COX = {
                'objective': 'survival:cox',
                'eval_metric': 'cox-nloglik',
                'tree_method': 'hist',
                'learning_rate': 5e-2,
                'max_depth': 3,
                'booster': 'gbtree',
                'subsample': 0.8,
                'min_child_weight': 50,
                'colsample_bynode': 0.5
            }

            dtrain = convert_data_to_xgb_format(x, y, 'survival:cox')
            dval = convert_data_to_xgb_format(x_test, y_test, 'survival:cox')

            bst = xgb.train(
                PARAMS_COX,
                dtrain,
                num_boost_round=1000,
                early_stopping_rounds=50,
                evals=[(dval, 'val')],
                verbose_eval=0
            )

            # predicting and evaluating
            test_preds = bst.predict(dval)
            train_preds = bst.predict(dtrain)

            scores = concordance_index_censored(y_test['Status'], y_test['Survival_in_days'], test_preds)
            c_index = round(scores[0], 10)

            baseline_model = BreslowEstimator().fit(train_preds, y['Status'], y['Survival_in_days'])
            survs = baseline_model.get_survival_function(test_pred)
            preds = np.asarray([[fn(t) for t in times] for fn in survs])
            scores = integrated_brier_score(y, y_test, preds, times)
            ibs = round(scores[0], 6)

            print(c_index, ibs)

            ana_list.append(['_'.join(item.split('_')[:-1]) + '_' + str(seed), eachCancer, c_index, ibs])

        elif method == 'coxnnet':

            # set parameters
            model_params = dict(node_map=None, input_split=None)
            search_params = dict(method="nesterov", learning_rate=0.01, momentum=0.9,
                                 max_iter=4000, stop_threshold=0.995, patience=1000, patience_incr=2,
                                 rand_seed=123, eval_step=23, lr_decay=0.9, lr_growth=1.0)
            L2_reg = 0.67

            model_params = dict(node_map=None, input_split=None, L2_reg=np.exp(L2_reg))
            print('Training Cox NN')
            model, cost_iter = trainCoxMlp(x, y['Survival_in_days'], y['Status'], model_params, search_params,
                                           verbose=True)

            # predicting and evaluating
            test_preds = model.predictNewData(x_test)
            train_preds = bst.predict(dtrain)

            scores = concordance_index_censored(y_test['Status'], y_test['Survival_in_days'], test_preds)
            c_index = round(scores[0], 10)

            baseline_model = BreslowEstimator().fit(train_preds, y['Status'], y['Survival_in_days'])
            survs = baseline_model.get_survival_function(test_pred)
            preds = np.asarray([[fn(t) for t in times] for fn in survs])
            scores = integrated_brier_score(y, y_test, preds, times)
            ibs = round(scores[0], 6)

            print(c_index, ibs)

            ana_list.append(['_'.join(item.split('_')[:-1]) + '_' + str(seed), eachCancer, c_index, ibs])

        print(item, ' Analysis: ', eachCancer)

    df = pd.DataFrame(ana_list)
    if pooled:
        df.to_csv(os.path.join(store_pth, ('Analysis_' + cancer_group + '_' + str(seed) + '_ibs.csv')), index=False)
    else:
        df.to_csv(os.path.join(store_pth, ('Analysis_' + cancer_group + '_' + str(seed) + '_ibs_unpooled.csv')), index=False)
    model_name = '_'.join(item.split('_')[:-1])

    return model_name + '_' + str(seed)

def process_feature_with_seed(path, core, cancer_group, seed_list, pooled, os_flag):
    if pooled:
        pth_file = 'CancerRNA_{}_WholeTimeSeq_6.txt'.format(cancer_group)
    elif os_flag:
        pth_file = 'CancerRNA_{}_WholeTimeSeq_3_OS.txt'.format(cancer_group)
    else:
        pth_file = 'CancerRNA_{}_WholeTimeSeq_3.txt'.format(cancer_group)
    processed_item = []
    store_pth = os.path.join(path, '{}_ibs_bench'.format(cancer_group))
    if not os.path.exists(store_pth):
        os.mkdir(store_pth)
    path_list = []
    cancer_group_list = []
    pooled_list = []
    for i in range(len(seed_list)):
        processed_item.append(pth_file)
        path_list.append(path)
        cancer_group_list.append(cancer_group)
        pooled_list.append(pooled)
    if core > 1:
        with concurrent.futures.ProcessPoolExecutor(core) as executor:
            for ana_out in executor.map(process_each_pth, processed_item, path_list, seed_list, cancer_group_list, pooled_list):
                print(ana_out)
    else:
        for k in range(len(processed_item)):
            ana_out = process_each_pth(processed_item[k], path_list[k], seed_list[k], cancer_group_list[k], pooled_list[k])
            print(ana_out)
    return

def main():
    seed_list = [10, 74, 341, 925, 1036, 1555, 1777, 2030, 2060, 2090, 2200,
                 2222, 2268, 2289, 2341, 2741, 2765, 2782, 2857, 2864, 2918,
                 2937, 2948, 2960, 2968, 3005, 3008, 3154, 3199, 3212, 3388,
                 3455, 3466, 3611, 3679, 3927, 4000, 4013, 4416, 4520]
    opt_get = parse_option()
    path = 'DataSet/'
    #path = 'SingleCancerDataSet/'
    print(opt_get.pooled)
    process_feature_with_seed(path, opt_get.core, opt_get.cancer_group, seed_list, opt_get.pooled, opt_get.os_flag)

if __name__ == '__main__':
    main()
