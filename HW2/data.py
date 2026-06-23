import pandas as pd
from datetime import datetime


def load_data(path):
    """reads and returns the pandas DataFrame"""
    return pd.read_csv(path)


def add_new_columns(df):
    """adds columns to df and returns the new df"""
    all_seasons = ['spring','summer','fall','winter']
    df['season_name'] =df['season'].apply(lambda x: all_seasons[x])

    df['hour'] = df['timestamp'].apply(lambda x: x.hour)
    df['day'] = df['timestamp'].apply(lambda x: x.day)
    df['month'] = df['timestamp'].apply(lambda x: x.month)
    df['year'] = df['timestamp'].apply(lambda x: x.year)

    df['is_weekend_holiday'] = df.apply(get_weekend_holiday_num, axis=1)

    df['t_diff'] = df.apply(lambda row: row['t2'] - row['t1'], axis=1)
    return df


def get_weekend_holiday_num(row):
    holiday = row['is_holiday']
    weekend = row['is_weekend']

    if holiday == 0 and weekend == 0:
        return 1
    elif holiday == 0 and weekend == 1:
        return 2
    elif holiday == 1 and weekend == 0:
        return 3
    elif holiday == 1 and weekend == 1:
        return 4


def data_analysis(df):
    """prints statistics on the transformed df"""
    print('describe output: ')
    print(df.describe().to_string())
    print()
    print('corr output: ')
    corr = df.corr(numeric_only=True)
    print(corr.to_string())
    print()

    """" 7 """
    all_columns_corr = {}
    columns = corr.columns
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            col1, col2 = columns[i], columns[j]
            all_columns_corr[(col1, col2)] = corr.loc[col1, col2]
    
    highest_corr = sorted(all_columns_corr.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
    print('Highest correlated are:')
    for idx, (pair, val) in enumerate(highest_corr, 1):
        print(f"{idx}. ('{pair[0]}', '{pair[1]}') with {round(val, 6)}")

    lowest_corr = sorted(all_columns_corr.items(), key=lambda item: abs(item[1]), reverse=False)[:5]
    print('\nLowest correlated are:')
    for idx, (pair, val) in enumerate(lowest_corr, 1):
        print(f"{idx}. ('{pair[0]}', '{pair[1]}') with {round(val, 6)}")
    print()

    """" 8 """
    all_seasons = ['fall', 'spring', 'summer', 'winter']
    all_seasons_average = df.groupby('season_name')['t_diff'].mean().round(2)
    for season in all_seasons:
        if season in all_seasons_average.index:
            print(f"{season} average t_diff is {all_seasons_average[season]}")

    overall_average = df['t_diff'].mean()
    print(f"All average t_diff is {round(overall_average, 2)}")