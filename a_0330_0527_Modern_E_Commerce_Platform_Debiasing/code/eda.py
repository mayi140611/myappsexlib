# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/0EDA.ipynb (unless otherwise specified).

__all__ = ['get_user_data', 'get_item_data', 'load_whole_click_data']

# Cell

import numpy as np
from loguru import logger
import os
import pandas as pd
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)  # 设置显示数据的最大列数，防止出现省略号…，导致数据显示不全
pd.set_option('expand_frame_repr', False)  # 当列太多时不自动换行
from .config import args


# Cell

def get_user_data():
    col_str = 'user_id user_age_level user_gender user_city_level'.split()
    train_user_df = pd.read_csv(os.path.join(args.DATA_DIR, 'data_gen/underexpose_train/underexpose_user_feat.csv'), names=['user_id','user_age_level','user_gender','user_city_level'],
                               converters={c:str for c in col_str})
    train_user_df = train_user_df.drop_duplicates('user_id')

    return train_user_df


# Cell
def get_item_data():
    train_item_df = pd.read_csv(os.path.join(args.DATA_DIR, 'data_gen/underexpose_train/underexpose_item_feat.csv'), sep=r',\s+|,\[|\],\[',
                                names=['item_id']+list(range(256)),
                                converters={'item_id':str})
    train_item_df.iloc[:, -1] = train_item_df.iloc[:, -1].str.replace(']', '').map(float)
    return train_item_df

# Cell

def load_whole_click_data(now_phase, base_dir):
    """
    """
#     now_phase = 2
    train_path = os.path.join(base_dir, 'data_gen/underexpose_train')
    test_path = os.path.join(base_dir, 'data_gen/underexpose_test')
    recom_item = []

    whole_click = pd.DataFrame()
    click_train = pd.DataFrame()
    click_test = pd.DataFrame()
    test_qtime = pd.DataFrame()
    click_test_val = pd.DataFrame()

    all_click_df = []
    for c in range(now_phase + 1):
        logger.info(f'phase: {c}')
        cols_str = 'user_id item_id time'.split()
        click_train1 = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'], converters={c: str for c in cols_str})
        click_test1 = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(c, c), header=None,  names=['user_id', 'item_id', 'time'], converters={c: str for c in cols_str})
        test_qtime1 = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_qtime-{}.csv'.format(c, c), header=None,  names=['user_id','time'], converters={c: str for c in cols_str})
#         test_qtime1['item_id'] = -1
#         click_test1_val = click_test1.sort_values(['user_id', 'time']).drop_duplicates(subset=['user_id'],keep='last')

#         click_test1 = click_test1[~click_test1.index.isin(click_test1_val.index)]
        click_train1['phase'] = c
        click_test1['phase'] = c
        test_qtime1['phase'] = c
        all_click = click_train1.append(click_test1)
#                     .append(test_qtime1)
        click_test = click_test.append(click_test1)
        whole_click = whole_click.append(all_click)
        test_qtime = test_qtime.append(test_qtime1)
#         click_test_val = click_test_val.append(click_test1_val, ignore_index=True)

    whole_click = whole_click.sort_values('time').drop_duplicates(['user_id', 'item_id', 'time'], keep='last').reset_index(drop=True)
#     logger.info(f'去重前whole_click 共{whole_click.shape[0]}条')
#     whole_click = pd.merge(whole_click, test_qtime, how='left').fillna(100)
#     whole_filter = whole_click[whole_click.time > whole_click.query_time]

#     logger.info(f'filter click data that time > query_time 共{whole_filter.shape[0]}条')
#     whole_click = whole_click[whole_click.time <= whole_click.query_time]
#     del whole_click['query_time']
    # 只保留一个user_id购买最后一次的item_id
    whole_click_val = whole_click[whole_click.user_id.isin(test_qtime.user_id)].drop_duplicates(['user_id'], keep='last')
    whole_click_train = whole_click[~whole_click.index.isin(whole_click_val.index)]
    logger.info(f'whole_click_train: {whole_click_train.shape}, whole_click_val: {whole_click_val.shape}, click_test: {click_test.shape}, test_qtime: {test_qtime.shape}')
    return whole_click_train, whole_click_val, click_test, test_qtime