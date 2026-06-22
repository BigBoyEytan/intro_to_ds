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
    """base of 7 and 8"""
    """prints statistics on the transformed df"""
    print('describe output: ')
    print(df.describe().to_string())
    print()
    print('corr output: ')
    corr = df.corr(numeric_only=True)
    print(corr.to_string())
    print()

    d1 ={}
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2 and (not (f'{col1},{col2}' in d1)):
                d1[f'{col1},{col2}'] = df[col1].corr(df[col2])


    """" 7 """
    print('Highest correlated are:')
    used_best = []
    dict_best = {}
    for i in range(5):
        l1 = find_best_cor(d1, used_best)
        dict_best[l1[0]] = l1[1]

    for key, value in dict_best.items():
        print(f'({key}) ' + f'with {round(value, 6)}')


    print('Lowest correlated are: ')
    used_worst = []
    dict_worst = {}
    for i in range(5):
        l2 = find_worst_cor(d1, used_worst)
        dict_worst[l2[0]] = l2[1]

    for key, value in dict_worst.items():
        print(f'({key}) ' + f'with {round(value, 6)}')

    """" 8 """
    season_avgs = df.groupby('season_name')['t_diff'].mean()
    l_of_unique_seasons = df['season_name'].unique().tolist()
    for season in l_of_unique_seasons:
        avg_val = season_avgs[season]
        print(f"{season} average t_diff is {round(avg_val, 2)}")

    overall_avg = df['t_diff'].mean()
    print(f"All average t_diff is {overall_avg}")


def find_best_cor(d1,used_best):
    max =0
    best_key =''
    for key, value in d1.items():
        if key not in used_best and value > max:
            max = value
            best_key = key
            used_best.append(key)

    return [best_key,max]


def find_worst_cor(d1,used_worst):
    min =1
    worst_key =''
    for key, value in d1.items():
        if key not in used_worst and value < min:
            min = value
            worst_key = key
            used_worst.append(key)

    return [worst_key,min]