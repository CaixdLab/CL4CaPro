import numpy as np
from sklearn.utils import check_consistent_length
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.metrics import integrated_brier_score, concordance_index_censored
import concurrent.futures

class StepFunction:
    """Callable step function.

    .. math::

        f(z) = a * y_i + b,
        x_i \\leq z < x_{i + 1}

    Parameters
    ----------
    x : ndarray, shape = (n_points,)
        Values on the x axis in ascending order.

    y : ndarray, shape = (n_points,)
        Corresponding values on the y axis.

    a : float, optional, default: 1.0
        Constant to multiply by.

    b : float, optional, default: 0.0
        Constant offset term.
    """
    def __init__(self, x, y, a=1., b=0.):
        check_consistent_length(x, y)
        self.x = x
        self.y = y
        self.a = a
        self.b = b

    def __call__(self, x):
        """Evaluate step function.

        Parameters
        ----------
        x : float|array-like, shape=(n_values,)
            Values to evaluate step function at.

        Returns
        -------
        y : float|array-like, shape=(n_values,)
            Values of step function at `x`.
        """
        x = np.atleast_1d(x)
        if not np.isfinite(x).all():
            raise ValueError("x must be finite")
        if np.min(x) < self.x[0] or np.max(x) > self.x[-1]:
            raise ValueError(
                "x must be within [%f; %f]" % (self.x[0], self.x[-1]))
        i = np.searchsorted(self.x, x, side='left')
        not_exact = self.x[i] != x
        i[not_exact] -= 1
        value = self.a * self.y[i] + self.b
        if value.shape[0] == 1:
            return value[0]
        return value

    def __repr__(self):
        return "StepFunction(x=%r, y=%r, a=%r, b=%r)" % (self.x, self.y, self.a, self.b)

def _compute_counts(event, time, order=None):
    """Count right censored and uncensored samples at each unique time point.

    Parameters
    ----------
    event : array
        Boolean event indicator.

    time : array
        Survival time or time of censoring.

    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.

    Returns
    -------
    times : array
        Unique time points.

    n_events : array
        Number of events at each time point.

    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.

    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = np.argsort(time, kind="mergesort")

    uniq_times = np.empty(n_samples, dtype=time.dtype)
    uniq_events = np.empty(n_samples, dtype=int)
    uniq_counts = np.empty(n_samples, dtype=int)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    times = np.resize(uniq_times, j)
    n_events = np.resize(uniq_events, j)
    total_count = np.resize(uniq_counts, j)
    n_censored = total_count - n_events

    # offset cumulative sum by one
    total_count = np.r_[0, total_count]
    n_at_risk = n_samples - np.cumsum(total_count)

    return times, n_events, n_at_risk[:-1], n_censored

class BreslowEstimator:
    """Breslow's estimator of the cumulative hazard function.

    Attributes
    ----------
    cum_baseline_hazard_ : :class:`sksurv.functions.StepFunction`
        Cumulative baseline hazard function.

    baseline_survival_ : :class:`sksurv.functions.StepFunction`
        Baseline survival function.
    """

    def fit(self, linear_predictor, event, time):
        """Compute baseline cumulative hazard function.

        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.

        event : array-like, shape = (n_samples,)
            Contains binary event indicators.

        time : array-like, shape = (n_samples,)
            Contains event/censoring times.

        Returns
        -------
        self
        """
        risk_score = np.exp(linear_predictor)
        order = np.argsort(time, kind="mergesort")
        risk_score = risk_score[order]
        uniq_times, n_events, n_at_risk, _ = _compute_counts(event, time, order)

        divisor = np.empty(n_at_risk.shape, dtype=float)
        value = np.sum(risk_score)
        divisor[0] = value
        k = 0
        for i in range(1, len(n_at_risk)):
            d = n_at_risk[i - 1] - n_at_risk[i]
            value -= risk_score[k:(k + d)].sum()
            k += d
            divisor[i] = value

        assert k == n_at_risk[0] - n_at_risk[-1]

        y = np.cumsum(n_events / divisor)
        self.cum_baseline_hazard_ = StepFunction(uniq_times, y)
        self.baseline_survival_ = StepFunction(self.cum_baseline_hazard_.x,
                                               np.exp(- self.cum_baseline_hazard_.y))
        return self

    def get_cumulative_hazard_function(self, linear_predictor):
        """Predict cumulative hazard function.

        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.

        Returns
        -------
        cum_hazard : ndarray, shape = (n_samples,)
            Predicted cumulative hazard functions.
        """
        risk_score = np.exp(linear_predictor)
        n_samples = risk_score.shape[0]
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = StepFunction(x=self.cum_baseline_hazard_.x,
                                    y=self.cum_baseline_hazard_.y,
                                    a=risk_score[i])
        return funcs

    def get_survival_function(self, linear_predictor):
        """Predict survival function.

        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.

        Returns
        -------
        survival : ndarray, shape = (n_samples,)
            Predicted survival functions.
        """
        risk_score = np.exp(linear_predictor)
        n_samples = risk_score.shape[0]
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = StepFunction(x=self.baseline_survival_.x,
                                    y=np.power(self.baseline_survival_.y, risk_score[i]))
        return funcs

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

def compute_ibs(cancer_group, seed, pooled=False):
    IBS_record = []
    if pooled:
        data = pd.read_csv(os.path.join(dataset_pth, 'CancerRNA_{}_WholeTimeSeq_6.txt'.format(cancer_group)))
    else:
        data = pd.read_csv(os.path.join(dataset_pth, 'CancerRNA_{}_WholeTimeSeq_3.txt'.format(cancer_group)))
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
    for cancer_get in CancerTypeList:
        get_data = data.loc[data['gen_id'] == cancer_get]
        rest_data = data.loc[data['gen_id'] != cancer_get]  # get rest data samples
        get_data.sort_values(by=['PFItime'])
        data_len = get_data.shape[0]
        start = 0
        if len(get_data) < 80:
            piece_num = 5
        else:
            piece_num = 20

        end = start + int(data_len / piece_num)
        y_train_get = []
        y_test_get = []

        if pooled:
            labely = rest_data[['PFItime', 'PFI']]
            y_train_get.append(labely)

        for i in range(piece_num):
            if i == piece_num - 1:
                end = data_len
            data_sample = get_data.iloc[start:end]
            start = end
            end = end + int(data_len / piece_num)
            labely = data_sample[['PFItime', 'PFI']]
            labelx = data_sample.iloc[:, 6:]
            _, _, y_train, y_test = train_test_split(labelx, labely, test_size=0.2, random_state=seed)
            y_train_get.append(y_train)
            y_test_get.append(y_test)

        y_train_combine = pd.concat(y_train_get)
        y_test_combine = pd.concat(y_test_get)

        y = build_y_data(y_train_combine)
        y_test = build_y_data(y_test_combine)

        combined_time_test_list = list(zip(y_test['Status'], y_test['Survival_in_days']))
        combined_time_train_list = list(zip(y['Status'], y['Survival_in_days']))
        sorted_combined_time_test_list = sorted(combined_time_test_list, key=lambda x: x[1])
        sorted_combined_time_train_list = sorted(combined_time_train_list, key=lambda x: x[1])
        sorted_status_test, sorted_time_test = zip(*sorted_combined_time_test_list)
        sorted_status_train, sorted_time_train = zip(*sorted_combined_time_train_list)

        last_true_index = -1
        num_thre = 30
        for index, status in reversed(list(enumerate(sorted_status_train))):
            if status and index <= len(sorted_status_train) - (num_thre + 1):
                last_true_index = index
                break
        sorted_time_train_end = sorted_time_train[last_true_index]

        last_true_index = -1
        num_thre = 30
        for index, status in reversed(list(enumerate(sorted_status_test))):
            if status and index <= len(sorted_status_test) - (num_thre + 1):
                last_true_index = index
                break
        sorted_time_test_end = sorted_time_train[last_true_index]

        #sorted_time_test_end = sorted_time_test[int(0.8 * len(sorted_time_test))]
        sorted_time_test_start = sorted_time_test[int(0 * len(sorted_time_test))]
        #sorted_time_train_end = sorted_time_train[int(0.8 * len(sorted_time_train))]
        sorted_time_train_start = sorted_time_train[int(0 * len(sorted_time_train))]
        times = np.arange(max(sorted_time_test_start, sorted_time_train_start), min(sorted_time_test_end, sorted_time_train_end))

        for item in os.listdir(pth):
            if item.split('_')[-1] == 'test.txt' and cancer_get in item and '_' + str(seed) + '_' in item:
                cancer_name, seed, ratio = item.split('_')[1], int(item.split('_')[2]), item.split('_')[3]

                train_pred = []
                test_pred = []

                with open(os.path.join(pth, item), 'r') as file:
                    for line in file:
                        # Convert each line to a float and append to the list
                        test_pred.append(float(line.strip()))

                with open(os.path.join(pth, item.replace('test', 'train')), 'r') as file:
                    for line in file:
                        # Convert each line to a float and append to the list
                        train_pred.append(float(line.strip()))

                try:
                    scores = concordance_index_censored(y_test['Status'], y_test['Survival_in_days'],
                                                        test_pred)
                    c_index = round(scores[0], 6)
                    baseline_model = BreslowEstimator().fit(train_pred, y['Status'], y['Survival_in_days'])
                    survs = baseline_model.get_survival_function(test_pred)
                    preds = np.asarray([[fn(t) for t in times] for fn in survs])
                    #print(len(y), len(y_test), len(preds), len(times))
                    scores = integrated_brier_score(y, y_test, preds, times)

                    IBS_record.append([cancer_name, seed, ratio, scores, c_index])
                    print(cancer_name, seed, ratio, scores, c_index)
                except:
                    print(cancer_name, seed, ratio, ' ERROR!')
    return IBS_record

#pth = 'predictions_save_single'
pth = 'predictions_save'
dataset_pth = 'DataSet'
#dataset_pth = 'SingleCancerDataSet'
pooled = False
IBS_record_total = []

#cancer_list = ["BLCA", "BRCA", "CESC", "COAD", "GBM", "HNSC", "KIRC", "KIRP", "LGG", "LIHC", "LUAD", "LUSC", "OV", "PRAD", "SARC", "SKCM", "STAD", "THCA", "UCEC"]
#cancer_list = ['PRAD', 'THCA', 'BRCA']
cancer_list = ["BLCA", "CESC", "COAD", "GBM", "HNSC", "KIRC", "KIRP", "LGG", "LIHC", "LUAD", "LUSC", "OV", "SARC", "SKCM", "STAD", "UCEC"]
#cancer_group_list = ['SM', 'NGT', 'MSE', 'CCPRC', 'HC', 'GG', 'DS', 'LC']
seed_list = [10, 74, 341, 925, 1036, 1555, 1777, 2030, 2060, 2090, 2200,
                 2222, 2268, 2289, 2341, 2741, 2765, 2782, 2857, 2864, 2918,
                 2937, 2948, 2960, 2968, 3005, 3008, 3154, 3199, 3212, 3388,
                 3455, 3466, 3611, 3679, 3927, 4000, 4013, 4416, 4520]
core = 10
for cancer_get in cancer_list:
    cancer_get_list = [cancer_get] * 40
    pooled_list = [pooled] * 40
    if core > 1:
        with concurrent.futures.ProcessPoolExecutor(core) as executor:
            for IBS_record_get in executor.map(compute_ibs, cancer_get_list, seed_list, pooled_list):
                IBS_record_total.extend(IBS_record_get)
    else:
        for seed in seed_list:
            IBS_record_get = compute_ibs(cancer_get, seed, pooled)
            IBS_record_total.extend(IBS_record_get)

df = pd.DataFrame(IBS_record_total)
df.to_csv('IBS_CLtest.csv')
