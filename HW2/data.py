import pandas as pd
from datetime import datetime



def load_data(path):
    """reads and returns the pandas DataFrame"""
    return pd.read_csv(path)



def add_new_columns(df):
    """adds columns to df and returns the new df"""
    df['season_name'] =df['season'].apply(to_season)

    df['hour'] = df['timestamp'].apply(get_hour)
    df['day'] = df['timestamp'].apply(get_day)
    df['month'] = df['timestamp'].apply(get_month)
    df['year'] = df['timestamp'].apply(get_year)

    df['is_weekend_holiday'] = df.apply(get_weekend_holiday_num, axis=1)
    df['t_diff'] = df.apply(get_t2_to_t1_dif, axis=1)

    return df

def data_analysis(df):
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

    df.groupby('Event').mean(numeric_only=True)


def to_season(x):
    if(x == 0):
        return 'spring'
    elif(x == 1):
        return 'summer'
    elif(x == 2):
        return 'fall'
    elif(x == 3):
        return 'winter'

def get_hour(text):
    text.split(' ')
    return text[1]

def get_day(text):
    t1= text.split(' ')[0].split('/')
    return t1[0]

def get_month(text):
    t1 = text.split(' ')[0].split('/')
    return t1[1]

def get_year(text):
    t1 = text.split(' ')[0].split('/')
    return t1[2]


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


def get_t2_to_t1_dif(row):
    t_1 = row['t1']
    t_2 = row['t2']
    return t_1 - t_2


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