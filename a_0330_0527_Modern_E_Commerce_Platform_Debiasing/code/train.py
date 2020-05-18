# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/3_model_new.ipynb (unless otherwise specified).

__all__ = ['trace', 'recall_from_user_item_matr', 'recall_from_cocur_matr', 'parser', 'args', 'whole_click', 'save_dir',
           'sr1']

# Cell
import os
import sys
sys.path.append('../nbdevlib/')
from algo.rs.matrix import *
from .val import main as val
from .eda import *
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import math
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

trace = logger.add('runtime.log')

# Cell

# topk=100

def recall_from_user_item_matr(sim_item_corr, topk, user_pred_list):
    rs = pd.Series()
    for u in tqdm(user_pred_list):
        items = items_list[u][-10:]
        len_items = len(items)
#         print(len_items)
        sr = pd.Series()
        for i, item in enumerate(items):
            sr = sr.append(sim_item_corr[item].sort_values()[-100:] * (len_items+i))
        sr = sr.reset_index()
        sr.columns = ['item_id', 'score']

        sr = sr.groupby('item_id')['score'].sum().reset_index()

        sr[~sr.item_id.isin(items_list[u])]
        rs.loc[u] = sr.sort_values('score', ascending=False)[:topk]['item_id'].to_list()
#                     break
    return rs

def recall_from_cocur_matr(matr, topk, user_pred_list):
    rs = pd.Series()
    for u in tqdm(user_pred_list):
        items = items_list[u][-20:]
        len_items = len(items)
#         print(len_items)
        sr = pd.Series()
        for i, item in enumerate(items):
            sr = sr.append(pd.Series(matr[item2id[item]].toarray()[0]).sort_values()[-100:] * (len_items+i))
        sr = sr.reset_index()
        sr.columns = ['id', 'score']

        sr = sr.groupby('id')['score'].sum().reset_index()
        sr['item_id'] = sr['id'].map(id2item)

        sr[~sr.item_id.isin(items_list[u])]
        rs.loc[u] = sr.sort_values('score', ascending=False)[:topk]['item_id'].to_list()
#                     break
    return rs

# Cell

parser = argparse.ArgumentParser(description='t')
parser.add_argument('--now_phase', type=int, default=6, help='')
parser.add_argument('--alpha', type=float, default=0, help='已经买过的item权重衰减系数')
parser.add_argument('--time_decay', type=float, default=7/8, help='时间衰减')
parser.add_argument('--submit_fp', type=str, default='/Users/luoyonggui/Downloads/baseline1_itemcf3.csv', help='提交文件生成位置')
parser.add_argument('--mode', type=str, default='train', help='train test')
parser.add_argument('--topk', type=int, default=200, help='每种召回策略召回的样本数')

args = parser.parse_args(args=[])
logger.info(args)

# Cell

whole_click_train, whole_click_val, test_qtime = load_whole_click_data(args.now_phase, BASE_DIR)

# 完整的click数据
whole_click = whole_click_train.append(whole_click_val)

whole_click.shape

# Cell
if args.mode == 'train':
    items_list = whole_click_train.groupby('user_id')['item_id'].agg(list)
elif args.mode == 'test':
    items_list = whole_click.groupby('user_id')['item_id'].agg(list)

logger.info('recall_from_cocur_matr')
save_dir=f'matrix_{args.mode}'
if not os.path.exists(save_dir):
    train_data_matrix, id2item, item2id = build_co_occurance_matrix(items_list.tolist(), window=20,
                                                      penalty1=True, penalty2=True,
                                                      penalty3=0.9, save_dir=f'matrix_{args.mode}')
else:
    train_data_matrix, id2item, item2id = load_co_occurance_matrix(save_dir=f'matrix_{args.mode}',
                                                      penalty1=True, penalty2=True,
                                                      penalty3=0.9)
sr1 = recall_from_cocur_matr(train_data_matrix, args.topk, test_qtime.user_id.tolist())

if args.mode == 'train':
    r_series = pd.Series()

    pred_num = whole_click_val.shape[0]
    for line in whole_click_val.itertuples():
        try:
            i = sr[line.user_id].index(line.item_id)
        except:
            i = 99999

        r_series = r_series.append(i)

    for i in range(50, args.topk+1, 50):
        logger.info(f'recall{i}:{r_series[r_series<i].shape[0]/pred_num}')
elif args.mode == 'test':
    sub = pd.DataFrame()

    for i in range(50):
        sub[f'c{i}'] = sr.str.get(i)

    sub.to_csv(args.submit_fp, header=False)