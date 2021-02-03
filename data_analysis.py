import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind, chi2_contingency
import numpy as np
from tqdm import tqdm

seed = 18
np.random.seed(seed)

dataset_path = "data/kidney.csv"

df = pd.read_csv(dataset_path)


columns = df.columns.tolist()
features, target = columns[:-1], columns[-1]

feature_clf = {'age': 'num', 'bp': 'num', 'sg':'cat', 'al': 'cat', 'su': 'cat', 'rbc': 'cat', 'pc': 'cat', 'pcc': 'cat', 'ba': 'cat', 'bgr': 'num', 'bu': 'num', 'sc': 'num', 'sod': 'num', 'pot': 'num', 'hemo': 'num', 'pcv': 'num', 'wbcc': 'num', 'rbcc': 'num', 'htn': 'cat', 'dm': 'cat', 'cad': 'cat', 'appet': 'cat', 'pe': 'cat', 'ane': 'cat', 'class': 'cat'}

label_dict = {'normal': 0, 'abnormal': 1, 'notpresent': 0, 'present': 1, 'yes': 1, 'no': 0, 'ckd': 1, 'notckd': 0, 'good': 0, 'poor':1}

df_processed = preprocess_columns(df)
df_train, df_test = train_test_split(df_processed, test_size=0.2, random_state=seed)

def preprocess_columns(df):
    processed_features = {}
    columns = df.columns.tolist()
    for feature in columns:
        if df[feature].dtype in [int, float]:
            processed_features[feature] = df[feature].tolist()
        elif df[feature].dtype == object:
            processed_features[feature] = [label_dict[sample] for sample in df[feature]]
    return pd.DataFrame(processed_features)

def test_column_significance(feature, target):
    if feature_clf[feature.name] == 'num':
        feature_true = feature[target == 1]
        feature_false = feature[target == 0]
        tval, pval = ttest_ind(feature_true, feature_false)
        bt_diff_means = bootstrap_diff_means(feature, target)
        diff_means, boot_means, boot_ci = compute_confidence_interval(feature, bt_diff_means)
        stat = {'diff_means': diff_means, 'boot_means': boot_means, 'boot_ci': boot_ci, 'tval': tval, 'pval': pval, 'bt_diff_means': bt_diff_means}
    elif feature_clf[feature.name] == 'cat':
        contingency_table = pd.concat([feature, target], axis=1).pivot_table(index=feature.name, columns=target.name, aggfunc=len).fillna(0).copy().astype(int)
        g, pval, dof, expected = chi2_contingency(contingency_table)
        stat = {'g': g, 'pval': pval, 'dof': dof, 'expected': expected, 'cont': contingency_table}
    else:
        print(f'COLUMN NOT CLASSIFIED IN feature_clf: {feature.name}')
    return stat


def bootstrap_diff_means(feature, target, sample_size=1000, bt_size=1000):
    bt_diff_means = []
    for bootstrap_iter in range(sample_size):
        boot_index = np.random.randint(len(target), size = bt_size)
        boot_feature, boot_target = feature[boot_index], target[boot_index]
        true_boot_feature = boot_feature[boot_target == 1]
        false_boot_feature = boot_feature[boot_target == 0]
        bt_diff_means.append(true_boot_feature.mean() - false_boot_feature.mean())
    return bt_diff_means

def compute_confidence_interval(feature, bt_diff_means):
    diff_means = feature.mean()
    boot_means = np.mean(bt_diff_means)
    boot_ci = np.quantile(bt_diff_means, q=[0.025, 0.975])
    return diff_means, boot_means, boot_ci

res_stats = {}
for feature in tqdm(features):
    res_stats[feature] = test_column_significance(df_processed[feature], df_processed[target])

pval_threshold = 0.01
significant_features = [feature for feature in features if res_stats[feature]['pval'] < pval_threshold]

