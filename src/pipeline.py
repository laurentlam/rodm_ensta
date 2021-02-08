# -*- coding: utf-8 -*-
"""SOD322-pipeline.ipynb

# SOD322: Recherche Opérationnelle et Données Massives

## Projet

Laurent Lam & Ilyes El-Rammach

### Import libraries
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split
from tqdm import tqdm

"""### Configuration variables"""

seed = 18
np.random.seed(seed)

test_size = 1 / 3

bootstrap_size = 1000
boot_iter = 1000
pval_threshold = 0.05
max_bins = 5

dataset_path = "../data/kidney.csv"

"""## Load dataset"""

df = pd.read_csv(dataset_path)

columns = df.columns.tolist()
features, target = columns[:-1], columns[-1]

feature_clf = {
    "age": "num",
    "bp": "num",
    "sg": "cat",
    "al": "cat",
    "su": "cat",
    "rbc": "cat",
    "pc": "cat",
    "pcc": "cat",
    "ba": "cat",
    "bgr": "num",
    "bu": "num",
    "sc": "num",
    "sod": "num",
    "pot": "num",
    "hemo": "num",
    "pcv": "num",
    "wbcc": "num",
    "rbcc": "num",
    "htn": "cat",
    "dm": "cat",
    "cad": "cat",
    "appet": "cat",
    "pe": "cat",
    "ane": "cat",
    "class": "cat",
}
label_dict = {"normal": 0, "abnormal": 1, "notpresent": 0, "present": 1, "yes": 1, "no": 0, "ckd": 1, "notckd": 0, "good": 0, "poor": 1}

"""### Preprocess/Format features"""


def preprocess_columns(df):
    processed_features = {}
    columns = df.columns.tolist()
    for feature in columns:
        if df[feature].dtype in [int, float]:
            processed_features[feature] = df[feature].tolist()
        elif df[feature].dtype == object:
            processed_features[feature] = [label_dict[sample] for sample in df[feature]]
    return pd.DataFrame(processed_features)


df_processed = preprocess_columns(df)

"""## Train test split"""

df_train, df_test = train_test_split(df_processed, test_size=test_size, random_state=seed)

"""## Training split analysis

### Independant analysis
"""


def test_column_significance(feature, target):
    # T-Test
    if feature_clf[feature.name] == "num":
        feature_true = feature[target == 1]
        feature_false = feature[target == 0]
        tval, pval = ttest_ind(feature_true, feature_false)
        bt_diff_means = bootstrap_diff_means(feature, target)
        diff_means, boot_means, boot_ci = compute_confidence_interval(feature, bt_diff_means)
        stat = {
            "diff_means": diff_means,
            "boot_means": boot_means,
            "boot_ci": boot_ci,
            "tval": tval,
            "pval": pval,
            "bt_diff_means": bt_diff_means,
        }
    # Chi-2 Test
    elif feature_clf[feature.name] == "cat":
        contingency_table = (
            pd.concat([feature, target], axis=1)
            .pivot_table(index=feature.name, columns=target.name, aggfunc=len)
            .fillna(0)
            .copy()
            .astype(int)
        )
        g, pval, dof, expected = chi2_contingency(contingency_table)
        stat = {"g": g, "pval": pval, "dof": dof, "expected": expected, "cont": contingency_table}
    else:
        print(f"COLUMN NOT CLASSIFIED IN feature_clf: {feature.name}")
    return stat


def bootstrap_diff_means(feature, target, sample_size=boot_iter, bt_size=bootstrap_size):
    bt_diff_means = []
    for bootstrap_iter in range(sample_size):
        boot_index = np.random.choice(target.index, size=bt_size)
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
    res_stats[feature] = test_column_significance(df_train[feature], df_train[target])

significant_features = [feature for feature in features if res_stats[feature]["pval"] < pval_threshold]

df_train = df_train[significant_features + ["class"]]
df_test = df_test[significant_features + ["class"]]

"""### Relationship Analysis # TODO"""


"""### Categorial variables: Modalities Aggregation"""


def get_aggregated_modalities(contingency_table, ineq="zero"):
    modalities = contingency_table.index.tolist()
    discr = False
    aggr = False
    aggr_mod = []
    aggr_mod_index = []
    for modality_index, modality in enumerate(modalities):
        if aggr is False:
            tmp_mod = []
            tmp_mod_index = []
        false, true = contingency_table.iloc[modality_index].to_numpy()
        if ineq == "zero":
            if aggr is True and (false != 0 and true != 0):
                aggr_mod.append(tmp_mod)
                aggr_mod_index.append(tmp_mod_index)
                tmp_mod = []
                tmp_mod_index = []
                aggr = False
            if false == 0 or true == 0:
                aggr = True
        tmp_mod.append(modality)
        tmp_mod_index.append(modality_index)
        if not aggr:
            aggr_mod.append(tmp_mod)
            aggr_mod_index.append(tmp_mod_index)
    if aggr_mod[-1] != tmp_mod:
        aggr_mod.append(tmp_mod)
        aggr_mod_index.append(tmp_mod_index)
    return aggr_mod, aggr_mod_index


def compute_aggr_mod_bins(res_stats):
    new_mod_bins = {}
    for feature in features:
        if feature_clf[feature] == "cat":
            if res_stats[feature]["dof"] > 2:
                aggregated_mod, aggregated_mod_index = get_aggregated_modalities(res_stats[feature]["cont"])
                if len(aggregated_mod) < res_stats[feature]["dof"]:
                    # new_mod_bins[feature] = [np.min(mod) for mod in aggregated_mod]
                    new_mod_bins[feature] = aggregated_mod
    return new_mod_bins


def reduce_mod_features(df, new_mod_bins):
    new_modalities_dict = {}
    for feature, mod_bins in new_mod_bins.items():
        new_modalities_dict[feature] = {modality: bin_index for bin_index, mod_bin in enumerate(mod_bins) for modality in mod_bin}
        df[feature] = df[feature].apply(lambda x: new_modalities_dict[feature][x])
    return df, new_modalities_dict


new_mod_bins = compute_aggr_mod_bins(res_stats)
df_train_reprocessed, new_modalities_dict = reduce_mod_features(df_train.copy(deep=True), new_mod_bins)

"""### Continuous variables : Binning"""

ft_num = set([ft for ft in features if feature_clf[ft] == "num"]).intersection(significant_features)


def assign_val2bin(value, bins):
    if value <= bins[0].left:
        return 0
    for bin_index, bin in enumerate(bins):
        if value in bin:
            return bin_index
    return len(bins) - 1


def assign_bins(df, binning_intervals):
    for ft in df.columns:
        if ft in binning_intervals:
            df[ft] = df[ft].apply(lambda x: assign_val2bin(x, binning_intervals[ft]))
    return df


"""#### Basic binning: Quantiles

"
binning_intervals = {}
for ft in ft_num:
    tmp = pd.qcut(df_train_reprocessed[ft], q=max_bins, duplicates='drop')
    binning_intervals[ft] = tmp.unique().tolist()
    df_train_reprocessed[ft] = pd.qcut(df_train_reprocessed[ft], q=max_bins, labels=range(len(binning_intervals[ft])), duplicates='drop')

#### Sick population's Quantiles + Aggregation
"""

df_true = df_train_reprocessed[df_train_reprocessed["class"] == 1]
df_false = df_train_reprocessed[df_train_reprocessed["class"] == 0]

continuous_bins = {ft: pd.qcut(df_true[ft], q=max_bins, duplicates="drop").values.categories for ft in ft_num}

df_bins = assign_bins(df_train_reprocessed, continuous_bins)


def compute_aggr_cont_bins(df, cont_bins):
    new_cont_bins = {}
    for feature in ft_num:
        contingency_table = contingency_table = (
            pd.concat([df[feature], df[target]], axis=1)
            .pivot_table(index=feature, columns=target, aggfunc=len)
            .fillna(0)
            .copy()
            .astype(int)
        )
        aggregated_cont, aggregated_cont_index = get_aggregated_modalities(contingency_table)
        if len(aggregated_cont) < len(cont_bins[feature]):
            new_cont_bins[feature] = aggregated_cont
    return new_cont_bins


new_cont_bins = compute_aggr_cont_bins(df_bins, continuous_bins)

df_aggr_bins, new_cont_dict = reduce_mod_features(df_bins.copy(deep=True), new_cont_bins)

"""### Binarization"""


def binarize_ordinal(df):
    df_binary = pd.DataFrame([])
    for ft in df.columns:
        n_cols = df[ft].nunique() - 1
        ft_cols = np.zeros((df.shape[0], n_cols))
        df_binary[[f"{ft}_{index}" for index in range(n_cols)]] = df[ft].apply(
            lambda x: pd.Series([int(x > index) for index in range(n_cols)])
        )
    return df_binary


df_train_reprocessed = df_aggr_bins.astype(int)
df_train_binary = binarize_ordinal(df_train_reprocessed)

"""## Format/Convert testing split"""

# Statistically significant features
df_test = df_test[significant_features + ["class"]]
# Modalities aggregation
df_test_reprocessed, _ = reduce_mod_features(df_test.copy(deep=True), new_mod_bins)
# Binning
## Continuous variables
df_test_bins = assign_bins(df_test_reprocessed, continuous_bins)
df_test_bins_cont, _ = reduce_mod_features(df_test_bins.copy(deep=True), new_cont_bins)
# Binarizing
df_test_bins_cont = df_test_bins_cont.astype(int)
df_test_binary = binarize_ordinal(df_test_bins_cont)

"""## Write to CSV files"""

df_train_binary.to_csv("/".join(dataset_path.split("/")[:-1]) + "/kidney_train.csv", index=False)
df_test_binary.to_csv("/".join(dataset_path.split("/")[:-1]) + "/kidney_test.csv", index=False)
