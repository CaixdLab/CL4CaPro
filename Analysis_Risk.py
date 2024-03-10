import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, auc, roc_curve, roc_auc_score
import numpy as np
import random
import concurrent.futures
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


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

    opt = parser.parse_args()

    return opt

def divide_number(n, bing=10):
    if n < 20:
        return [n]
    else:
        if bing > 10:
            remainder = n % bing
            if n // bing > remainder:
                result = [bing] * (n // bing)
                for j in range(remainder):
                    result[-(j + 1)] += 1
                return result
            else:
                try:
                    min_remainder = 15
                    result = []
                    for k in range(bing, 16):
                        remainder = n % k
                        if remainder < min_remainder:
                            min_remainder = remainder
                            result = [k] * (n // k)
                    for j in range(min_remainder):
                        result[-(j + 1)] += 1
                    return result
                except:
                    return divide_number(n, bing - 1)
        else:
            min_remainder = 15
            result = []
            for k in range(10, 16):
                remainder = n % k
                if remainder < min_remainder:
                    min_remainder = remainder
                    result = [k] * (n // k)
            for j in range(min_remainder):
                result[-(j + 1)] += 1
            return result

def calculate_sensitivity_specificity(y_test, predictions):
    true_positive = np.sum((y_test == 1) & (predictions == 1))
    true_negative = np.sum((y_test == 0) & (predictions == 0))
    false_positive = np.sum((y_test == 0) & (predictions == 1))
    false_negative = np.sum((y_test == 1) & (predictions == 0))

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)

    return sensitivity, specificity

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
        try:
            get_data = data.loc[data['gen_id'] == eachCancer]
            rest_data = data.loc[data['gen_id'] != eachCancer]  # get rest data samples
            get_data = get_data.sort_values(by=['PFItime'])

            get_data_0 = get_data[get_data['PFItime'] < (3*365)]
            get_data_1 = get_data[get_data['PFItime'] >= (3*365)]
            data_len_0 = get_data_0.shape[0]
            data_len_1 = get_data_1.shape[0]
            if data_len_0 < data_len_1:
                get_data_0_split = divide_number(data_len_0)
                get_data_1_split = divide_number(data_len_1, get_data_0_split[0])
            else:
                get_data_1_split = divide_number(data_len_1)
                get_data_0_split = divide_number(data_len_0, get_data_1_split[0])
            print(data_len_0, get_data_0_split, data_len_1, get_data_1_split)

            x_train_get = []
            y_train_get = []
            x_test_get = []
            y_test_get = []
            x_train_get_split = [[], [], [], [], []]
            y_train_get_split = [[], [], [], [], []]

            if pooled:
                labely = rest_data['predicted_label']
                labelx = rest_data.iloc[:, 6:]
                x_train_get.append(labelx)
                y_train_get.append(labely)

            start = 0
            for split_get in get_data_0_split:
                end = start + split_get
                #print(start, end)
                data_sample = get_data.iloc[start:end]
                start = end
                labely = data_sample['predicted_label']
                labelx = data_sample.iloc[:, 6:]
                x_train, x_test, y_train, y_test = train_test_split(labelx, labely, test_size=0.2,
                                                                    random_state=seed)  # Pass seed 10, 74, 341, 925, 1036 #114
                x_train_get.append(x_train)
                y_train_get.append(y_train)
                x_test_get.append(x_test)
                y_test_get.append(y_test)

                sublists_x = []
                sublists_y = []

                data_format = list(zip(x_train.values.tolist(), y_train.values.tolist()))

                sublist_length = len(data_format) // 5

                # randomly shuffle the data list
                random.shuffle(data_format)

                # split the shuffled list into 5 sublists of equal length
                sublists = [data_format[i:i + sublist_length] for i in range(0, len(data_format), sublist_length)]
                if sublist_length * 5 < len(data_format):
                    sublists[-1] += (data_format[len(sublists) * sublist_length:])

                # separate the sublists back into x and y
                sublists_x += ([t[0] for t in sublist] for sublist in sublists)
                sublists_y += ([t[1] for t in sublist] for sublist in sublists)

                if len(sublists_y) > 5:
                    sublists_y[-2] += sublists_y[-1]
                    sublists_y.pop()
                    sublists_x[-2] += sublists_x[-1]
                    sublists_x.pop()

                for j in range(5):
                    x_train_get_split[j] += sublists_x[j]
                    y_train_get_split[j] += sublists_y[j]

            start = data_len_0
            for split_get in get_data_1_split:
                end = start + split_get
                # print(start, end)
                data_sample = get_data.iloc[start:end]
                start = end
                labely = data_sample['predicted_label']
                labelx = data_sample.iloc[:, 6:]
                x_train, x_test, y_train, y_test = train_test_split(labelx, labely, test_size=0.2,
                                                                    random_state=seed)  # Pass seed 10, 74, 341, 925, 1036 #114
                x_train_get.append(x_train)
                y_train_get.append(y_train)
                x_test_get.append(x_test)
                y_test_get.append(y_test)

                sublists_x = []
                sublists_y = []

                data_format = list(zip(x_train.values.tolist(), y_train.values.tolist()))

                sublist_length = len(data_format) // 5

                # randomly shuffle the data list
                random.shuffle(data_format)

                # split the shuffled list into 5 sublists of equal length
                sublists = [data_format[i:i + sublist_length] for i in range(0, len(data_format), sublist_length)]
                if sublist_length * 5 < len(data_format):
                    sublists[-1] += (data_format[len(sublists) * sublist_length:])

                # separate the sublists back into x and y
                sublists_x += ([t[0] for t in sublist] for sublist in sublists)
                sublists_y += ([t[1] for t in sublist] for sublist in sublists)

                if len(sublists_y) > 5:
                    sublists_y[-2] += sublists_y[-1]
                    sublists_y.pop()
                    sublists_x[-2] += sublists_x[-1]
                    sublists_x.pop()

                for j in range(5):
                    x_train_get_split[j] += sublists_x[j]
                    y_train_get_split[j] += sublists_y[j]


            x_train_combine = pd.concat(x_train_get)
            y_train_combine = pd.concat(y_train_get)
            x_test_combine = pd.concat(x_test_get)
            y_test_combine = pd.concat(y_test_get)

            x_train_combine_split = []
            y_train_combine_split = []
            for j in range(5):
                x_train_combine_split.append((x_train_get_split[j]))
                y_train_combine_split.append((y_train_get_split[j]))

            x = x_train_combine
            y = y_train_combine
            x_test = x_test_combine
            y_test = y_test_combine

            #eval_set = [(x, y), (x_test, y_test)]

            # time distribution CV
            param_max_depth = [3, 4, 5, 6]
            param_learning_rate = [0.3, 0.2, 0.1]
            param_n_estimators = [100]
            best_combine = []

            # Perform the five-fold cross-validation search for the best hyperparameters
            for i in range(5):

                best_d = 0
                best_l = 0
                best_n = 0
                best_score_auc_roc = 0

                # Define the training and testing data for this fold
                x_train_cv = np.vstack(x_train_combine_split[:i] + x_train_combine_split[i + 1:])
                y_train_cv = np.hstack(y_train_combine_split[:i] + y_train_combine_split[i + 1:])
                x_test_cv = x_train_combine_split[i]
                y_test_cv = y_train_combine_split[i]

                # Define the GridSearchCV object to search over the hyperparameter grid
                for d_para in param_max_depth:
                    for l_para in param_learning_rate:
                        for n_para in param_n_estimators:
                            xgb_model = XGBClassifier(max_depth=d_para, learning_rate=l_para, n_estimators=n_para)#, tree_method='gpu_hist', predictor='gpu_predictor')
                            xgb_model.fit(x_train_cv, y_train_cv)
                            prob_predictions = xgb_model.predict_proba(x_test_cv)[:, 1]
                            score_auc_roc = roc_auc_score(y_test_cv, prob_predictions)
                            if score_auc_roc > best_score_auc_roc:
                                best_score_auc_roc = score_auc_roc
                                best_d = d_para
                                best_l = l_para
                                best_n = n_para
                best_combine.append([best_d, best_l, best_n, best_score_auc_roc])

            # Define a dictionary to count the frequency of each combination of parameters
            params_count = {}

            # Define a dictionary to store the total score for each combination of parameters
            params_score = {}

            # Loop through each sublist in the params_list
            for sublist in best_combine:
                # Get the values of a, b, c, and score for the current sublist
                a, b, c, score = sublist

                # Check if the current combination of parameters is already in the params_count dictionary
                if (a, b, c) in params_count:
                    # If it is, increment the count and add the score to the params_score dictionary
                    params_count[(a, b, c)] += 1
                    params_score[(a, b, c)] += score
                else:
                    # If it is not, add the combination to the params_count dictionary with a count of 1 and add the score to the params_score dictionary
                    params_count[(a, b, c)] = 1
                    params_score[(a, b, c)] = score

            # Get the combination of parameters that appears the most in the params_list
            max_count = max(params_count.values())
            best_params = max([key for key, value in params_count.items() if value == max_count],
                              key=lambda x: params_score[x] / params_count[x])

            # Create an XGBClassifier object with best parameters
            model = XGBClassifier(max_depth=best_params[0], learning_rate=best_params[1], n_estimators=best_params[2])#**grid_search.best_params_)
            model.fit(x, y)
            #model.fit(x, y)#, eval_metric=["error", "logloss"], early_stopping_rounds=30, eval_set=eval_set, verbose=True)

            predictions = model.predict(x_test)
            prob_predictions = model.predict_proba(x_test)[:, 1]

            # Compute ROC curve and ROC area for each class
            fpr, tpr, _ = roc_curve(y_test, prob_predictions)
            auc_roc = roc_auc_score(y_test, prob_predictions)

            # Plot
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % auc_roc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.savefig('./ROCPlotGroup/ROCCurve_{}_{}_{}.png'.format(eachCancer, item, auc_roc))

            print(auc_roc)
            f1 = f1_score(y_test, predictions)
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            sensitivity, specificity = calculate_sensitivity_specificity(y_test, predictions)
            ana_list.append([item, eachCancer, auc_roc, f1, accuracy, precision, recall, sensitivity, specificity, best_params[0], best_params[1], best_params[2]])  # alphas=alpha_get,

            # Define the filename
            round_get = item.split('_')[-2]
            filename = 'ROCDataGroup/{}_round_{}_{}_{}_fpr_tpr_values.txt'.format(eachCancer, round_get, seed, auc_roc)

            # Open the file in write mode
            with open(filename, 'w') as file:
                # Write the header
                file.write('FPR,TPR\n')

                # Write the fpr and tpr values
                for fpr_value, tpr_value in zip(fpr, tpr):
                    file.write(f"{fpr_value},{tpr_value}\n")

            print(item, ' Analysis: ', eachCancer)
            model_name = '_'.join(item.split('_')[:-1])
            layer_name = item.split('_')[-1].split('.')[0]
            print(model_name, layer_name)
        except:
            print(item, 'can not Analysis: ', eachCancer)
    return ana_list

def process_feature_with_seed(path, seed, core, cancer_group):
    all_files = os.listdir(path)
    pth_files = list(filter(lambda f: f.endswith('.txt'), all_files))
    total_ana_list = []
    processed_item = []
    path_list = []
    seed_list = []
    cancer_group_list = []
    for item in pth_files:
        if ('seed_' + str(seed) + '_') in item:
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
    ana_list = process_feature_with_seed(path, opt_get.seed, opt_get.core, opt_get.cancer_group)
    df = pd.DataFrame(ana_list)
    df.to_csv('Analysis_' + opt_get.cancer_group + '_' + str(opt_get.seed) + '.csv', index=False)

if __name__ == '__main__':
    main()