import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import scipy
from xgbse.converters import (
    convert_data_to_xgb_format,
    convert_to_structured
)
from xgbse.metrics import concordance_index
import xgboost as xgb
import numpy as np
import concurrent.futures
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def parse_option():
    parser = argparse.ArgumentParser('argument for analysis features extracted from trained models')

    parser.add_argument('--model_save_path', type=str, default='SupCLCP',
                        help='model save path')
    parser.add_argument('--seed', type=int, default=1036,
                        help='random seed')
    parser.add_argument('--cancer_group', type=str, default='SM',
                        help='cancer group (SM/NGT/MSE/CCPRC/HC/GG/DS/LC)')
    parser.add_argument('--core', type=int, default=10,
                        help='cores reserved to run the script')
    parser.add_argument('--validation', type=str, default='TCGA',
                        help='validation dataset using')

    opt = parser.parse_args()

    return opt


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

def process_each_pth(item, path, seed, cancer_group, pooled=False):
    ana_list = []
    print('Processing:', item)
    data = pd.read_csv(path + item)
    folder = item[:-4]
    save_path = path + folder + '_' + str(seed)
    try:
        os.mkdir(save_path)
    except:
        print('Path exists.')
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
        get_data = get_data.sort_values(by=['PFItime'])
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

        # Convert survival time from days to months
        survival_in_months = y['Survival_in_days'] / 30.44

        # Create a new dtype that includes Survival_in_months
        new_dtype = np.dtype(y.dtype.descr + [('Survival_in_months', '<f8')])

        # Create a new structured array with the new dtype
        new_y = np.zeros(y.shape, dtype=new_dtype)

        # Copy over the data from y
        for name in y.dtype.names:
            new_y[name] = y[name]

        # Add the new Survival_in_months data
        new_y['Survival_in_months'] = survival_in_months
        y_month = new_y

        # Convert survival time from days to months
        survival_in_months = y_test['Survival_in_days'] / 30.44

        # Create a new dtype that includes Survival_in_months
        new_dtype = np.dtype(y_test.dtype.descr + [('Survival_in_months', '<f8')])

        # Create a new structured array with the new dtype
        new_y = np.zeros(y_test.shape, dtype=new_dtype)

        # Copy over the data from y
        for name in y_test.dtype.names:
            new_y[name] = y_test[name]

        # Add the new Survival_in_months data
        new_y['Survival_in_months'] = survival_in_months
        y_test_month = new_y

        best_ratio = 0
        best_c = 0

        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # Use an appropriate random_state

        best_c_index = 0
        best_l_ratio = None
        best_model = None

        for l_ratio in np.linspace(0.05, 0.95, 19):  # Explore l_ratio from 0.05 to 0.95
            c_index_scores = []

            for train_index, test_index in kf.split(x):
                X_train, X_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = CoxnetSurvivalAnalysis(l1_ratio=l_ratio, fit_baseline_model=True, normalize=True)
                model.fit(X_train, y_train)

                predictions = model.predict(X_test)
                c_index = concordance_index_censored(y_test['Status'], y_test['Survival_in_days'], predictions)[0]
                c_index_scores.append(c_index)

            avg_c_index = np.mean(c_index_scores)

            if avg_c_index > best_c_index:
                best_c_index = avg_c_index
                best_l_ratio = l_ratio
                best_model = model

        test_preds = best_model.predict(x_test)
        train_preds = best_model.predict(x)

        # save model output
        np.savetxt(
            "./predictions_save/preds_{}_{}_{}_train.txt".format(eachCancer, seed,
                                                                 best_l_ratio)
            , train_preds, fmt="%f")
        np.savetxt(
            "./predictions_save/preds_{}_{}_{}_test.txt".format(eachCancer, seed,
                                                                best_l_ratio)
            , test_preds, fmt="%f")

        scores = concordance_index_censored(y_test['Status'], y_test['Survival_in_days'],
                                            test_preds)
        c_index = round(scores[0], 6)

        # Calculate median hazard ratio for training set
        mHR = np.median(train_preds)

        # Divide training set samples into two groups
        group1_indices = train_preds > mHR
        group2_indices = ~group1_indices

        # Divide training set samples into two groups
        group1_indices_test = test_preds > mHR
        group2_indices_test = ~group1_indices_test

        # Construct DataFrame
        data_HR = pd.DataFrame(y, columns=['Status', 'Survival_in_days'])
        data_HR['Group'] = 0
        data_HR.loc[group1_indices, 'Group'] = 1

        '''

        # Fit CoxPH model
        cph = CoxPHFitter()
        cph.fit(data_HR, duration_col='Survival_in_days', event_col='Status')

        #cph.print_summary()

        # Extract HR and its 95% CI
        hr = cph.summary.loc['Group', 'exp(coef)']
        lower_ci = cph.summary.loc['Group', 'exp(coef) lower 95%']
        upper_ci = cph.summary.loc['Group', 'exp(coef) upper 95%']

        HR_text_train = 'HR: ' + str(round(hr, 3)) + '(' + str(round(lower_ci, 3)) + '-' + str(round(upper_ci, 3)) + ')'

        # Construct DataFrame
        data_HR = pd.DataFrame(y_test, columns=['Status', 'Survival_in_days'])
        data_HR['Group'] = 0
        data_HR.loc[group1_indices_test, 'Group'] = 1

        # Fit CoxPH model
        cph = CoxPHFitter()
        cph.fit(data_HR, duration_col='Survival_in_days', event_col='Status')

        # Extract HR and its 95% CI
        hr = cph.summary.loc['Group', 'exp(coef)']
        lower_ci = cph.summary.loc['Group', 'exp(coef) lower 95%']
        upper_ci = cph.summary.loc['Group', 'exp(coef) upper 95%']

        HR_text_test = 'HR: ' + str(round(hr, 3)) + '(' + str(round(lower_ci, 3)) + '-' + str(
            round(upper_ci, 3)) + ')'
        
        '''

        # Kaplan-Meier curve estimation
        kmf = KaplanMeierFitter()

        # Create a subplot with 2 rows and 2 columns
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

        # Define groups and data
        scenarios = [
            {"data": y_month, "group1_indices": group1_indices, "group2_indices": group2_indices,
             "label": "Training Data", "ax": axes[0, 0], "ax_table": axes[1, 0]},
            {"data": y_test_month, "group1_indices": group1_indices_test, "group2_indices": group2_indices_test,
             "label": "Test Data", "ax": axes[0, 1], "ax_table": axes[1, 1]}
        ]

        # Iterate through scenarios (training and test data)
        for scenario in scenarios:
            # Define groups
            groups = [
                {"indices": scenario["group1_indices"], "label": "High Risk", "color": "red"},
                {"indices": scenario["group2_indices"], "label": "Low Risk", "color": "blue"}
            ]

            # Plot KM Curve
            for group in groups:
                kmf.fit(scenario["data"]['Survival_in_months'][group["indices"]],
                        event_observed=scenario["data"]['Status'][group["indices"]],
                        label=group["label"])
                kmf.plot(ax=scenario["ax"], ci_show=False, color=group["color"])

            scenario["ax"].set_title(scenario["label"], fontsize=20)
            scenario["ax"].set_ylabel('Survival Probability', fontsize=20)
            scenario["ax"].set_xlabel('Time (months)', fontsize=20)
            scenario["ax"].legend(fontsize=20)
            scenario["ax"].tick_params(axis='both', labelsize=20)
            scenario["ax"].grid(True)

            # Get x-axis tick locations
            #times = scenario["ax"].get_xticks().tolist()
            #times = [int(time) for time in times if time >= 0]
            #times.pop()

            # Create a table for number at risk
            num_at_risk = [[], [], []]  # Two empty lists for the two groups

            # Find the maximum time (last index) from the kmf.event_table
            max_time = kmf.event_table.index.max()

            # Define times as percentiles of the last index
            percentiles_of_interest = [0, 20, 40, 60, 80]
            times = [int(max_time * (p / 100)) for p in percentiles_of_interest]

            for time in times:
                num_at_risk[2].append(time)

            # Calculating number at risk for each group and time point
            for i, group in enumerate(groups):
                kmf.fit(scenario["data"]['Survival_in_months'][group["indices"]],
                        event_observed=scenario["data"]['Status'][group["indices"]],
                        label=group["label"])

                #print(kmf.event_table)

                for time in times:
                    # Find the first index that is >= time
                    indices = np.where(kmf.event_table.index >= time)[0]

                    #print(time, indices)#, kmf.event_table.iloc[indices[0]])

                    # Check if there are any indices that satisfy the condition
                    if indices.size > 0:
                        current_idx = indices[0]

                        # Check if it's the first index
                        if current_idx == 0:
                            at_risk = kmf.event_table.iloc[current_idx]["at_risk"]
                        # Else, check if the previous index < time
                        elif kmf.event_table.index[current_idx - 1] < time:
                            at_risk = kmf.event_table.iloc[current_idx]["at_risk"]
                        else:
                            at_risk = 0
                    else:
                        at_risk = 0

                    num_at_risk[i].append(at_risk)

            # Plot Table
            scenario["ax_table"].axis('tight')
            scenario["ax_table"].axis('off')
            row_labels = ["High Risk", "Low Risk", "Time"]
            cell_text = [num_at_risk[0], num_at_risk[1], num_at_risk[2]]
            table = scenario["ax_table"].table(cellText=cell_text, rowLabels=row_labels,
                                       cellLoc='center', loc='center')
            # Set fontsize
            table.auto_set_font_size(False)
            table.set_fontsize(16)

            # Adjusting width and height of the cells
            cells = table.get_celld()
            for i in range(len(row_labels)):
                for j in range(len(times)):  # assuming times is your columns
                    cells[i, j].set_height(0.2)  # adjust height as desired
                    cells[i, j].set_width(0.2)  # adjust width as desired

            # Adjusting the width and height of row label cells
            for i in range(len(row_labels)):
                cells[i, -1].set_width(0.2)  # adjust width of row labels as desired
                cells[i, -1].set_height(0.2)  # adjust height of row labels as desired

            for j in range(len(times) + 1):
                cells[0, j-1].get_text().set_color('red')
                cells[1, j-1].get_text().set_color('blue')

        # Log-rank test train
        results_train = logrank_test(
            y['Survival_in_days'][group1_indices], y['Survival_in_days'][group2_indices],
            event_observed_A=y['Status'][group1_indices],
            event_observed_B=y['Status'][group2_indices]
        )

        # Log-rank test test
        results_test = logrank_test(
            y_test['Survival_in_days'][group1_indices_test], y_test['Survival_in_days'][group2_indices_test],
            event_observed_A=y_test['Status'][group1_indices_test],
            event_observed_B=y_test['Status'][group2_indices_test]
        )

        # Adding p-value to the training data subplot
        if results_train.p_value < 0.0001:
            axes[0, 0].text(0.02, 0.12, f'p-value < 0.0001',
                            transform=axes[0, 0].transAxes,  # Use axis coordinate system
                            fontsize=16,
                            verticalalignment='bottom',
                            horizontalalignment='left',
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
        else:
            axes[0, 0].text(0.02, 0.12, f'p-value = {results_train.p_value:.4f}',
                            transform=axes[0, 0].transAxes,  # Use axis coordinate system
                            fontsize=16,
                            verticalalignment='bottom',
                            horizontalalignment='left',
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

        #axes[0, 0].text(0.02, 0.02, HR_text_train,
        #                transform=axes[0, 0].transAxes,  # Use axis coordinate system
        #                fontsize=16,
        #                verticalalignment='bottom',
        #                horizontalalignment='left',
        #                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

        # Adding p-value to the test data subplot
        if results_test.p_value < 0.0001:
            axes[0, 1].text(0.02, 0.12, f'p-value < 0.0001',
                            transform=axes[0, 1].transAxes,  # Use axis coordinate system
                            fontsize=16,
                            verticalalignment='bottom',
                            horizontalalignment='left',
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
        else:
            axes[0, 1].text(0.02, 0.12, f'p-value = {results_test.p_value:.4f}',
                            transform=axes[0, 1].transAxes,  # Use axis coordinate system
                            fontsize=16,
                            verticalalignment='bottom',
                            horizontalalignment='left',
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
        #axes[0, 1].text(0.02, 0.02, HR_text_test,
        #                transform=axes[0, 1].transAxes,  # Use axis coordinate system
        #                fontsize=16,
        #                verticalalignment='bottom',
        #                horizontalalignment='left',
        #                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

        # Add a super title for the entire figure
        plt.suptitle('Kaplan-Meier Survival Curves {}'.format(eachCancer), fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the suptitle

        if results_test.p_value < 0.05:

            plt.savefig('./KMC/kaplan_meier_curves_{}_{}_{}_{}_{}_{}.png'.format(eachCancer, item.replace('.txt', ''),
                                                                                 ratio_get, c_index,
                                                                                 results_train.p_value,
                                                                                 results_test.p_value))
        plt.close()  # Close the figure

        ana_list.append([item, eachCancer, best_ratio, c_index, results_train.p_value, results_test.p_value])
        print([item, eachCancer, best_ratio, c_index, results_train.p_value, results_test.p_value])

        if results_test.p_value < 0.05:
            np.savetxt(
                "./predictions/preds_ratio_{}_{}_{}_{}_{}_{}_test_h.txt".format(eachCancer, seed,
                                                                                best_ratio, c_index,
                                                                                round(results_train.p_value, 5),
                                                                                round(results_test.p_value, 5))
                , y_test[group1_indices_test], fmt="%f")
            np.savetxt(
                "./predictions/preds_ratio_{}_{}_{}_{}_{}_{}_test_l.txt".format(eachCancer, seed,
                                                                                best_ratio, c_index,
                                                                                round(results_train.p_value, 5),
                                                                                round(results_test.p_value, 5))
                , y_test[group2_indices_test], fmt="%f")
            np.savetxt(
                "./predictions/preds_ratio_{}_{}_{}_{}_{}_{}_train_h.txt".format(eachCancer, seed,
                                                                                 best_ratio, c_index,
                                                                                 round(results_train.p_value, 5),
                                                                                 round(results_test.p_value, 5))
                , y[group1_indices], fmt="%f")
            np.savetxt(
                "./predictions/preds_ratio_{}_{}_{}_{}_{}_{}_train_l.txt".format(eachCancer, seed,
                                                                                 best_ratio, c_index,
                                                                                 round(results_train.p_value, 5),
                                                                                 round(results_test.p_value, 5))
                , y[group2_indices], fmt="%f")

        print(item, ' Analysis: ', eachCancer)
        model_name = '_'.join(item.split('_')[:-1])
        layer_name = item.split('_')[-1].split('.')[0]
        print(model_name, layer_name)
    return ana_list


def process_feature_with_seed(path, seed, core, cancer_group, validation):
    all_files = os.listdir(path)
    pth_files = list(filter(lambda f: f.endswith('.txt'), all_files))
    total_ana_list = []
    processed_item = []
    path_list = []
    seed_list = []
    cancer_group_list = []
    validation_list = []
    for item in pth_files:
        if ('seed_' + str(seed) + '_') in item:
            if validation == 'Affy':
                if '13235' in item:
                    processed_item.append(item)
                    path_list.append(path)
                    seed_list.append(seed)
                    cancer_group_list.append(cancer_group)
            else:
                processed_item.append(item)
                path_list.append(path)
                seed_list.append(seed)
                cancer_group_list.append(cancer_group)


    if core > 1:
        with concurrent.futures.ProcessPoolExecutor(core) as executor:
            for ana_list in executor.map(process_each_pth, processed_item, path_list, seed_list, cancer_group_list):
                total_ana_list.extend(ana_list)
    else:
        for k in range(len(processed_item)):
            ana_list = process_each_pth(processed_item[k], path_list[k], seed_list[k], cancer_group_list[k])
            total_ana_list.extend(ana_list)
    return total_ana_list


def main():
    opt_get = parse_option()
    path = './save/{}_{}_{}/Features/'.format(opt_get.model_save_path, opt_get.cancer_group, opt_get.seed)
    ana_list = process_feature_with_seed(path, opt_get.seed, opt_get.core, opt_get.cancer_group, opt_get.validation)
    df = pd.DataFrame(ana_list)
    df.to_csv('Analysis_' + opt_get.cancer_group + '_' + str(opt_get.seed) + '.csv', index=False)


if __name__ == '__main__':
    main()
