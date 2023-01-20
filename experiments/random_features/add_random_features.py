import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path=".", config_name="random_features")
def add_features(config: DictConfig):
    def add_random_features(df, features, user_id_col, full_random_features):
        df_ids = df[user_id_col].unique()
        for feature, values in features.items():
            if full_random_features:
                df[feature] = np.random.randint(low=values["min"], high=values["max"], size=len(df))
            else:
                random_column = pd.Series(np.random.randint(low=values["min"], high=values["max"], size=len(df_ids)), index=df_ids)
                df[feature] = random_column.reindex(df[user_id_col].values).values
        return df

    def add_arange_features(df, user_id_col, date_col, cycled_arange, datetime_format):
        if datetime_format:
            df['date'] = pd.to_datetime(df[date_col], format=datetime_format)
            df = df.sort_values(by=['date'])
        if cycled_arange:
            df['arange'] = df.groupby([user_id_col]).cumcount()%100 + 1
        else:
            df['arange'] = df.groupby([user_id_col]).cumcount() + 1
        df = df.sort_index()
        return df 
        
    np.random.seed(config["seed"])
    train = pd.read_csv(config["data_train"])
    test = pd.read_csv(config["data_test"])
    
    if config["add_random_features"]:
        train = add_random_features(train, config["random_features"], config["user_id_col"], config["full_random_features"])
        test = add_random_features(test, config["random_features"], config["user_id_col"], config["full_random_features"])
    if config["add_arange_features"]:
        train = add_arange_features(train, config["user_id_col"], config["date_col"], config["cycled_arange"], config["datetime_format"])
        test = add_arange_features(test, config["user_id_col"], config["date_col"], config["cycled_arange"], config["datetime_format"])

    # print(len(train['arange'].unique()))
    train.to_csv(config["train_out"])
    test.to_csv(config["test_out"])


if __name__ == '__main__':
    add_features()