import logging
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

from config import get_config


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(funcName)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def preprocess_columns(df, cfg_dict):
    processed_features = {}
    df = df.replace("?", np.nan)
    df = df.replace(" ?", np.nan)
    df = df.dropna()
    columns = df.columns.tolist()
    for feature in columns:
        if df[feature].dtype in [int, float]:
            processed_features[feature] = df[feature].tolist()
        elif df[feature].dtype == object:
            if set(df[feature].unique()).issubset(set(cfg_dict["label_dict"])):
                processed_features[feature] = [cfg_dict["label_dict"][sample] for sample in df[feature]]
            else:
                sample_list = sorted(df[feature].unique().tolist())
                processed_features[feature] = [sample_list.index(sample) for sample in df[feature]]
    return pd.DataFrame(processed_features)


def preprocess_adult(df, cfg_dict):
    df_processed = df.copy(deep=True)
    df_processed["education_cat"] = pd.cut(df_processed["education_num"], bins=[0, 8, 12, 16], labels=range(3)).astype(int)
    df_processed["workclass_gov"] = df_processed["workclass"].replace(cfg_dict["workclass_gov"], inplace=False).astype(int)
    df_processed["workclass_private"] = df_processed["workclass"].replace(cfg_dict["workclass_private"], inplace=False).astype(int)
    df_processed["marital_status_cat"] = df_processed["marital_status"].replace(cfg_dict["marital_status_cat"], inplace=False).astype(int)
    # df_processed["occupation_cat"] = df_processed["occupation"].replace(cfg_dict["occupation_cat"], inplace=False).astype(int)
    ft_to_delete = ["education", "relationship", "education_num", "native_country", "workclass", "marital_status", "occupation", "race"]
    df_processed = df_processed[[ft for ft in df_processed.columns if ft not in ft_to_delete]]
    df_processed = add_column_modalities(df_processed, cfg_dict)
    return preprocess_columns(df_processed, cfg_dict)


def add_column_modalities(df, cfg_dict):
    new_df = df[[ft for ft in df.columns if ft not in cfg_dict["ft_mod_dict"]]].copy(deep=True)
    for ft, modalities in cfg_dict["ft_mod_dict"].items():
        for modality in modalities:
            new_df[f"{ft}_{modality.strip()}"] = df[ft].apply(lambda x: int(x == modality))
            cfg_dict["feature_clf"][f"{ft}_{modality.strip()}"] = "class"
    return new_df


def test_column_significance(feature, target, cfg_dict):
    # T-Test
    if cfg_dict["feature_clf"][feature.name] == "num":
        feature_true = feature[target == 1]
        feature_false = feature[target == 0]
        tval, pval = ttest_ind(feature_true, feature_false)
        bt_diff_means = bootstrap_diff_means(feature, target, cfg_dict)
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
    elif cfg_dict["feature_clf"][feature.name] in ["cat", "class"]:
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


def bootstrap_diff_means(feature, target, cfg_dict):
    bt_diff_means = []
    for bootstrap_iter in range(cfg_dict["boot_iter"]):
        boot_index = np.random.choice(target.index, size=cfg_dict["bootstrap_size"])
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


def get_correlated_variables(df, threshold):
    columns = df.columns.tolist()
    corr_var = dict()
    for method in ["pearson", "spearman", "kendall"]:
        corr = df.corr(method=method)
        pair_index = np.where(corr.abs().values > threshold)
        unique_pairs = []
        for ft1, ft2 in zip(*pair_index):
            if ft1 != ft2:
                if (columns[ft2], columns[ft1]) not in corr_var and (columns[ft1], columns[ft2]) not in corr_var:
                    corr_var[(columns[ft1], columns[ft2])] = [abs(corr.values[ft1, ft2])]
                elif (columns[ft2], columns[ft1]) not in corr_var:
                    corr_var[(columns[ft1], columns[ft2])].append(abs(corr.values[ft1, ft2]))
                else:
                    corr_var[(columns[ft2], columns[ft1])].append(abs(corr.values[ft1, ft2]))
    return corr_var


def get_corr_features(corr_dict, target):
    to_remove = []
    for (ft1, ft2), corr in corr_dict.items():
        if ft1 == target:
            continue
        elif ft2 == target:
            continue
        # If ft1 != target and ft2 != target
        # If ft1 correlated to target
        elif (ft1, target) in corr_dict:
            # If ft2 correlated to target
            if (ft2, target) in corr_dict:
                # If ft2 more correlated to target than ft1
                if np.mean(corr_dict[(ft1, target)]) < np.mean(corr_dict[(ft2, target)]):
                    to_remove.append(ft1)
                else:
                    to_remove.append(ft2)
            elif (target, ft2) in corr_dict:
                if np.mean(corr_dict[(ft1, target)]) < np.mean(corr_dict[(target, ft2)]):
                    to_remove.append(ft1)
                else:
                    to_remove.append(ft2)
            else:
                to_remove.append(ft2)
        elif (target, ft1) in corr_dict:
            if (ft2, target) in corr_dict:
                if np.mean(corr_dict[(target, ft1)]) < np.mean(corr_dict[(ft2, target)]):
                    to_remove.append(ft1)
                else:
                    to_remove.append(ft2)
            elif (target, ft2) in corr_dict:
                if np.mean(corr_dict[(target, ft1)]) < np.mean(corr_dict[(target, ft2)]):
                    to_remove.append(ft1)
                else:
                    to_remove.append(ft2)
            else:
                to_remove.append(ft2)
    return to_remove


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


def create_aggr_mod_bins(res_stats, cfg_dict):
    new_mod_bins = {}
    for feature in res_stats:
        if cfg_dict["feature_clf"][feature] == "cat":
            if res_stats[feature]["dof"] > 2:
                aggregated_mod, aggregated_mod_index = get_aggregated_modalities(res_stats[feature]["cont"])
                if len(aggregated_mod) < res_stats[feature]["dof"]:
                    new_mod_bins[feature] = aggregated_mod
    return new_mod_bins


def reduce_mod_features(df, new_mod_bins):
    new_modalities_dict = {}
    columns = df.columns.tolist()
    for feature, mod_bins in new_mod_bins.items():
        if feature in columns:
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


def create_bins(df, target, cfg_dict):
    columns = df.columns.tolist()
    features, target = columns[:-1], columns[-1]
    ft_num = set([ft for ft in features if cfg_dict["feature_clf"][ft] == "num"])
    df_true = df[df[target] == 1]
    df_false = df[df[target] == 0]
    return {ft: pd.qcut(df_true[ft], q=cfg_dict["max_bins"], duplicates="drop").values.categories for ft in ft_num}


def create_aggr_cont_bins(df, continuous_bins, cfg_dict):
    new_cont_bins = {}
    columns = df.columns.tolist()
    features, target = columns[:-1], columns[-1]
    ft_num = set([ft for ft in features if cfg_dict["feature_clf"][ft] == "num"])
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


def binarize_ordinal(df, df_binary, cols_to_binarize):
    for ft in cols_to_binarize:
        n_cols = df[ft].nunique() - 1
        df_binary[[f"{ft}_{index}" for index in range(n_cols)]] = df[ft].apply(
            lambda x: pd.Series([int(x > index) for index in range(n_cols)])
        )
    return df_binary


def binarize_classical(df, df_binary, cols_to_binarize):
    for ft in cols_to_binarize:
        n_cols = df[ft].nunique() - 1
        df_binary[[f"{ft}_{index}" for index in range(n_cols)]] = df[ft].apply(
            lambda x: pd.Series([int(x == index + 1) for index in range(n_cols)])
        )
    return df_binary


def binarize(df, cfg_dict):
    df_binary = pd.DataFrame([])
    ordinal_cols = [ft for ft in df.columns.tolist()[:-1] if cfg_dict["feature_clf"][ft] != cfg_dict["target"]]
    class_cols = [ft for ft in df.columns.tolist()[:-1] if ft not in ordinal_cols] + [cfg_dict["target"]]
    df_binary = binarize_ordinal(df, df_binary, ordinal_cols)
    df_binary = binarize_classical(df, df_binary, class_cols)
    return df_binary
