import os
import yaml
import argparse
import pandas as pd
import numpy as np


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--conf_path', type=os.path.abspath, default="random_features.yaml")
    args = parser.parse_args(args)

    with open(args.conf_path, 'rb') as f:
        conf = yaml.load(f.read(), Loader=yaml.FullLoader)

    return conf

def add_features(config):
    def add_random_features(df, features, user_id_col):
        df_ids = df[user_id_col].unique()
        for feature, values in features.items():
            df[feature] = np.nan
            for user_id in df_ids:
                df[feature][df[user_id_col] == user_id] = np.random.randint(low=values["min"], high=values["max"])
        return df

    def add_arange_features(df, user_id_col, date_col, cycled_arange):
        df["arange"] = np.nan
        df['date'] = pd.to_datetime(df[date_col], format="%d%b%y:%H:%M:%S")
        df = df.sort_values(by=['date'])
        for user_id in df[user_id_col].unique():
            if cycled_arange:
                df.arange[df[user_id_col] == user_id] = np.arange(len(df[df[user_id_col] == user_id]), dtype=int)%100 + 1
            else:
                df.arange[df[user_id_col] == user_id] = np.arange(len(df[df[user_id_col] == user_id]), dtype=int)
        df = df.sort_index()
        return df 
        
    np.random.seed(config["seed"])
    train = pd.read_csv(config["data_train"])
    test = pd.read_csv(config["data_test"])
    
    if config["add_random_features"]:
        train = add_random_features(train, config["random_features"], config["user_id_col"])
        test = add_random_features(test, config["random_features"], config["user_id_col"])
    if config["add_arange_features"]:
        train = add_arange_features(train, config["user_id_col"], config["date_col"], config["cycled_arange"])
        test = add_arange_features(test, config["user_id_col"], config["date_col"], config["cycled_arange"])

    # print(len(train['arange'].unique()))
    train.to_csv(config["train_out"])
    test.to_csv(config["test_out"])


if __name__ == '__main__':
    config = parse_args()
    add_features(config)