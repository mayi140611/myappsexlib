# AUTOGENERATED! DO NOT EDIT! File to edit: algo_rs_matrix.ipynb (unless otherwise specified).

__all__ = ['build_co_occurance_matrix', 'load_co_occurance_matrix', 'build_user_item_matrix',
           'build_item_item_cosine_matrix']

# Cell
import scipy.sparse as sp
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Cell

def build_co_occurance_matrix(items_list,
                              window=9999,
                              penalty1=False,
                              penalty2=False,
                              penalty3=1,
                              penalty4=False,
                              save_dir=None):
    """

    :items_list:
        [
            [item1, item2],
            [item1, item2, item4],
            ...
        ]
    : window: int
        只有在window内才算共现
    : penalty1:
        对距离惩罚，距离越远，相关性越小
    : penalty2:
        对list长度惩罚，长度越长，对共现的价值越小
    : penalty3: float
        对seq方向惩罚，方向为正 不惩罚，否则惩罚
        1表示不惩罚
    : penalty4:
        对item出现次数做惩罚，出现越多，对共现的价值越小

    :return:


    """
    from collections import Counter
    item_list_flat = [ii for i in items_list for ii in i]
    item_num_dict = dict(Counter(item_list_flat))  # 每个item出现次数的字典

    items = pd.Series(list(item_num_dict.keys()))
    item2id = pd.Series(items.index, items)

    n_items = items.shape[0]
    print(f'n_items: {n_items}')
    train_data_matrix = sp.lil_matrix((n_items, n_items), dtype=np.float)
    for items_ in tqdm(items_list):
        for i, item in enumerate(items_):
            for j, related_item in enumerate(items_):
                distance = np.abs(i-j)
                if (item != related_item) and (distance<window):
                    vt = 1
                    if penalty1:
                        vt /= np.sqrt(np.log2(distance+1))
                    if penalty2:
                        vt /= np.log10(len(items_)+9)
                    if i < j:
                        vt *= penalty3
                    train_data_matrix[item2id.loc[item], item2id.loc[related_item]] += vt
    if penalty4:
        for r in tqdm(range(train_data_matrix.shape[0])):
            for c in train_data_matrix.rows[r]:
                train_data_matrix[r,c] /= (np.log(item_num_dict[items[r]]+1)*np.log(item_num_dict[items[c]]+1))

    if save_dir:
        if not os.path.exists(save_dir):
            print(f'create matrix dir {save_dir}')
            os.mkdir(save_dir)
        items.to_pickle(os.path.join(save_dir, f'id2item_series_{penalty1}_{penalty2}_{penalty3}.pkl'))
        item2id.to_pickle(os.path.join(save_dir, f'item2id_series_{penalty1}_{penalty2}_{penalty3}.pkl'))
        sp.save_npz(os.path.join(save_dir, f'item_item_matrix_{penalty1}_{penalty2}_{penalty3}.npz'), train_data_matrix.tocsc())
        print(f'save matrix to {save_dir}, finished')
    return train_data_matrix, items, item2id

# Cell
def load_co_occurance_matrix(save_dir,penalty1,penalty2,penalty3):
    id2item = pd.read_pickle(os.path.join(save_dir, f'id2item_series_{penalty1}_{penalty2}_{penalty3}.pkl'))
    item2id = pd.read_pickle(os.path.join(save_dir, f'item2id_series_{penalty1}_{penalty2}_{penalty3}.pkl'))
    co_occurance_matrix = sp.load_npz(os.path.join(save_dir, f'item_item_matrix_{penalty1}_{penalty2}_{penalty3}.npz'))
    return co_occurance_matrix, id2item, item2id

# Cell
def build_user_item_matrix(df, user_col, item_col):
    """
    使用pd.crosstab(df[user_col], df[item_col])可以直接达到目的，但是当items很大时，会报异常:
    ValueError: Unstacked DataFrame is too big, causing int32 overflow
    """

    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    id2user = df[user_col].drop_duplicates().reset_index(drop=True)
    user2id = pd.Series(id2user.index, id2user)
    id2item = df[item_col].drop_duplicates().reset_index(drop=True)
    item2id = pd.Series(id2item.index, id2item)
    print(f'n_users: {n_users}, n_items: {n_items}')
    train_data_matrix = sp.lil_matrix((n_users, n_items))
    for line in df[[user_col, item_col]].itertuples():
        train_data_matrix[user2id[line[1]], item2id[line[2]]] += 1
    train_data_matrix = train_data_matrix.tocsc()
    train_data_matrix.data = np.log(train_data_matrix.data + 1)
    return train_data_matrix, id2user, user2id, id2item, item2id

# Cell
def build_item_item_cosine_matrix(matr, Y=None):
    """
    由于item一般数据很多(数10w)，需要很大的内存
    """
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(matr, Y)