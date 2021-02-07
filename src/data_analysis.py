import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split
from tqdm import tqdm

seed = 18
np.random.seed(seed)

# Load data
dataset_path = "../data/kidney.csv"

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


# Preprocess datatypes into int, float
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


def test_column_significance(feature, target):
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


def bootstrap_diff_means(feature, target, sample_size=1000, bt_size=1000):
    bt_diff_means = []
    for bootstrap_iter in range(sample_size):
        boot_index = np.random.randint(len(target), size=bt_size)
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


# Filter out non-statistically significant features
res_stats = {}
for feature in tqdm(features):
    res_stats[feature] = test_column_significance(df_processed[feature], df_processed[target])

pval_threshold = 0.05
significant_features = [feature for feature in features if res_stats[feature]["pval"] < pval_threshold]

df_reprocessed = df_processed[significant_features + ["class"]]


def get_aggregated_modalities(contingency_table):
    modalities = contingency_table.index.tolist()
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


# Aggregate bins for nominal variables
new_mod_bins = compute_aggr_mod_bins(res_stats)

df_reprocessed, new_modalities_dict = reduce_mod_features(df_reprocessed.copy(deep=True), new_mod_bins)


# Basic binning of continuous variables - quantile-based
ft_num = [ft for ft in features if feature_clf[ft] == "num"]
for ft in ft_num:
    tmp = pd.qcut(df_reprocessed[ft], q=5, duplicates="drop").nunique()
    df_reprocessed[ft] = pd.qcut(df_reprocessed[ft], q=5, labels=range(tmp), duplicates="drop")


# Binarization for ordinal classification
df_reprocessed = df_reprocessed.astype(int)


def binarize_ordinal(df):
    df_binary = pd.DataFrame([])
    for ft in df.columns:
        n_cols = df[ft].nunique() - 1
        ft_cols = np.zeros((df.shape[0], n_cols))
        df_binary[[f"{ft}_{index}" for index in range(n_cols)]] = df_reprocessed[ft].apply(
            lambda x: pd.Series([int(x > index) for index in range(n_cols)])
        )
    return df_binary


df_binary = binarize_ordinal(df_reprocessed)

# Train test split
df_train, df_test = train_test_split(df_binary, test_size=1 / 3, random_state=seed)

# Write to csv files
df_train.to_csv("/".join(dataset_path.split("/")[:-1]) + "/kidney_train.csv", index=False)
df_test.to_csv("/".join(dataset_path.split("/")[:-1]) + "/kidney_test.csv", index=False)
