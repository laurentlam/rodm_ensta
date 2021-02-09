# -*- coding: utf-8 -*-
"""SOD322-pipeline.ipynb

# SOD322: Recherche Opérationnelle et Données Massives

## Projet

Laurent Lam & Ilyes El-Rammach
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from config import *
from functions import *

logger = get_logger()


def create_features(df):
    # Collect features and target
    columns = df.columns.tolist()
    features, target = columns[:-1], columns[-1]

    # Independent statistical analysis
    logger.info('Independant statistical analysis...')
    stats_features = {feature: test_column_significance(df[feature], df[target]) for feature in features}
    significant_features = [feature for feature in features if stats_features[feature]["pval"] < pval_threshold]
    logger.info(f'Found {len(significant_features)}/{len(features)} significant features.')
    df_filtered = df[significant_features + [target]]

    # Nominal variables: Modalities aggregation
    logger.info('Nominal features: Aggregating modalities...')
    aggr_mod_bins = create_aggr_mod_bins(stats_features)
    logger.info(f'Nominal features: Found modalities to aggregate from {len(aggr_mod_bins)} features.')
    df_aggr_mod = reduce_mod_features(df_filtered, aggr_mod_bins)

    # Continuous variables: Binning
    logger.info('Continuous features: Creating bins...')
    continuous_bins = create_bins(df_aggr_mod, target)
    logger.info('Continuous features: Assigning bins...')
    df_bins = assign_bins(df_aggr_mod, continuous_bins)
    # Continuous variables: Bins aggregation
    logger.info('Continuous features: Aggregating bins...')
    aggr_cont_bins = create_aggr_cont_bins(df_bins, continuous_bins)
    logger.info(f'Continuous features: Found bins to aggregate from {len(aggr_cont_bins)} features.')
    df_aggr_bins = reduce_mod_features(df_bins, aggr_cont_bins)

    # Binarization
    logger.info('Binarizing features...')
    df_train_binary = binarize_ordinal(df_aggr_bins)

    # Collect features markers
    features_markers = {
        "target": target,
        "significant_features": significant_features,
        "aggr_mod_bins": aggr_mod_bins,
        "continuous_bins": continuous_bins,
        "aggr_cont_bins": aggr_cont_bins,
    }
    logger.info('Created features markers for features binarization.')
    return df_train_binary, features_markers


def transform_features(df, features_markers):
    # Statistically significant features
    logger.info('Filtering out non-significant features...')
    df_filtered = df[features_markers["significant_features"] + [features_markers["target"]]]
    # Modalities aggregation
    logger.info('Reducing nominal modalities...')
    df_aggr_mod = reduce_mod_features(df_filtered, features_markers["aggr_mod_bins"])
    # Binning
    ## Continuous variables
    logger.info('Assigning bins...')
    df_bins = assign_bins(df_aggr_mod, features_markers["continuous_bins"])
    logger.info('Reducing bins modalities...')
    df_aggr_bins = reduce_mod_features(df_bins, features_markers["aggr_cont_bins"])
    # Binarizing
    logger.info('Binarizing features...')
    df_binary = binarize_ordinal(df_aggr_bins)
    logger.info('Transformed features into binary features.')
    return df_binary


def write_csv_files(dataset, dataset_path, df_train, df_test):
    folder_path = "/".join(dataset_path.split("/")[:-1])
    logger.info(f"Writing train split into CSV file at {folder_path + f'/{dataset}_train.csv'}...")
    df_train.to_csv(folder_path + f'/{dataset}_train.csv', index=False)
    logger.info(f"Writing test split into CSV file at {folder_path + f'/{dataset}_test.csv'}...")
    df_test.to_csv(folder_path + f'/{dataset}_test.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="dataset name",
    )
    args = vars(parser.parse_args())
    dataset = args["dataset"]
    dataset_path = f"./data/{dataset}.csv"
    # Load dataset
    logger.info(f'Loading {dataset} dataset at {dataset_path}.')
    df = pd.read_csv(dataset_path)
    logger.info(f'Loaded {dataset}: {df.shape[0]} individuals with {df.shape[1] - 1} features.')
    # Preprocess columns
    logger.info(f'Pre-processing {dataset} columns...')
    df_processed = preprocess_columns(df)
    # Train test split
    logger.info(f'Splitting into train test with test_size: {round(test_size, 2)}.')
    df_train, df_test = train_test_split(df_processed, test_size=test_size, random_state=seed)
    # Create features
    logger.info('Creating features from training set...')
    df_train_binary, features_markers = create_features(df_train)
    # Transform features
    df_test_binary = transform_features(df_test, features_markers)
    # Write to CSV files
    logger.info('Writing into CSV files...')
    write_csv_files(dataset, dataset_path, df_train_binary, df_test_binary)
    logger.info('Pre-processing pipeline done.')
