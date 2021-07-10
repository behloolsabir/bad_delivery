import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import pickle as pkl
import datetime

def loadData(sample=True):
    if sample:
        return pd.read_csv('../data/processed_data/data_sample.csv',
                           parse_dates=['created_timestamp_local'])
    return pd.read_json("../../location_task/location_task_no_nulls.json",
                        convert_dates=['created_timestamp_local'])


def featureEngineering(data_df, filename='treated_data_sample.pkl'):
    """
    Feature Engg
    1. Time stamp converted to day of the week, day and hour
    2. Geohash converted into seperate 8 bits
    """
    try:
        print(f'Looking for an existing pickle {filename}')
        return pkl.load(open(f'../data/processed_data/{filename}', 'rb'))
    except FileNotFoundError:
        print(f'Pickle {filename} does not exist.\nPreparing data now...')
        data_df = data_df.loc[data_df.logistics_dropoff_distance > 0]
        data_df[
            'created_timestamp_local_dayofweek'] = data_df.created_timestamp_local.dt.dayofweek
        data_df[
            'created_timestamp_local_day'] = data_df.created_timestamp_local.dt.day
        data_df[
            'created_timestamp_local_hour'] = data_df.created_timestamp_local.dt.hour

        for idx in range(1, 9):
            data_df[
                f'delivery_geohash_precision8_{idx}'] = data_df.delivery_geohash_precision8.str.split(
                    '').str[idx]
        pkl.dump(data_df, open(f'../data/processed_data/{filename}', 'wb'))
        return data_df


def labelEncoding(train_df, test_df, encode_features):
    d = defaultdict(LabelEncoder)
    train_df[encode_features] = train_df[encode_features].apply(
        lambda x: d[x.name].fit_transform(x))
    test_df[encode_features] = test_df[encode_features].apply(
        lambda x: d[x.name].transform(x))
    return train_df[encode_features], test_df[encode_features]


def trainTestSplit(data_df):
    train_data = data_df[
        data_df.created_timestamp_local.dt.date < datetime.date(2021, 3, 25)]
    test_data = data_df[
        data_df.created_timestamp_local.dt.date >= datetime.date(2021, 3, 25)]
    return train_data, test_data