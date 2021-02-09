import logging
from functools import partial
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

from config import *

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


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


def get_aggregated_modalities(contingency_table):
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


def create_aggr_mod_bins(res_stats):
    new_mod_bins = {}
    for feature in res_stats:
        if feature_clf[feature] == "cat":
            if res_stats[feature]["dof"] > 2:
                aggregated_mod, aggregated_mod_index = get_aggregated_modalities(res_stats[feature]["cont"])
                if len(aggregated_mod) < res_stats[feature]["dof"]:
                    new_mod_bins[feature] = aggregated_mod
    return new_mod_bins


def reduce_mod_features(df, new_mod_bins):
    new_modalities_dict = {}
    for feature, mod_bins in new_mod_bins.items():
        new_modalities_dict[feature] = {modality: bin_index for bin_index, mod_bin in enumerate(mod_bins) for modality in mod_bin}
        df[feature].replace(new_modalities_dict[feature], inplace=True)
    return df


def assign_val2bin(value, bins):
    if value <= bins[0].left:
        return 0
    for bin_index, bin in enumerate(bins):
        if value in bin:
            return bin_index
    return len(bins) - 1


def assign_bins(df, binning_intervals):

    for ft in binning_intervals:
        assign_val2bin_ft = partial(assign_val2bin, bins=binning_intervals[ft])
        df[ft] = df[ft].apply(assign_val2bin_ft)
    return df


def create_bins(df, target):
    columns = df.columns.tolist()
    features, target = columns[:-1], columns[-1]
    ft_num = set([ft for ft in features if feature_clf[ft] == "num"])
    df_true = df[df[target] == 1]
    df_false = df[df[target] == 0]
    return {ft: pd.qcut(df_true[ft], q=max_bins, duplicates="drop").values.categories for ft in ft_num}


def create_aggr_cont_bins(df, continuous_bins):
    new_cont_bins = {}
    columns = df.columns.tolist()
    features, target = columns[:-1], columns[-1]
    ft_num = set([ft for ft in features if feature_clf[ft] == "num"])
    for feature in ft_num:
        contingency_table = contingency_table = (
            pd.concat([df[feature], df[target]], axis=1)
            .pivot_table(index=feature, columns=target, aggfunc=len)
            .fillna(0)
            .copy()
            .astype(int)
        )
        aggregated_cont, aggregated_cont_index = get_aggregated_modalities(contingency_table)
        if len(aggregated_cont) < len(continuous_bins[feature]):
            new_cont_bins[feature] = aggregated_cont
    return new_cont_bins


def binarize_ordinal(df):
    df = df.astype(int)
    df_binary = pd.DataFrame([])
    for ft in df.columns:
        n_cols = df[ft].nunique() - 1
        ft_cols = np.zeros((df.shape[0], n_cols))
        df_binary[[f"{ft}_{index}" for index in range(n_cols)]] = df[ft].apply(
            lambda x: pd.Series([int(x > index) for index in range(n_cols)])
        )
    return df_binary