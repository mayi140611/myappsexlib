"""
日期:2020-04-25 13:49:56排名: 无  只是itemcf phase3

score:0.1855
hitrate_50_full:0.4485
ndcg_50_full:0.1855
hitrate_50_half:0.3359
ndcg_50_half:0.1382

日期:2020-04-19 00:25:12排名: 无  对user点击item按时间做指数加权处理

score:0.1275
hitrate_50_full:0.3071
ndcg_50_full:0.1275
hitrate_50_half:0.2171
ndcg_50_half:0.0952

日期:2020-04-18 20:54:35排名: 无  对user点击item按时间做线性加权处理

score:0.1353
hitrate_50_full:0.3291
ndcg_50_full:0.1353
hitrate_50_half:0.2454
ndcg_50_half:0.1010

日期:2020-04-18 18:48:13排名: 无 phase2 use_iif=False

score:0.1195
hitrate_50_full:0.3001
ndcg_50_full:0.1195
hitrate_50_half:0.2228
ndcg_50_half:0.0857

日期:2020-04-18 18:41:10排名: 无 phase2

score:0.1264
hitrate_50_full:0.3139
ndcg_50_full:0.1264
hitrate_50_half:0.2331
ndcg_50_half:0.0934

日期:2020-04-17 16:07:36排名: 无 phase1

score:0.0841
hitrate_50_full:0.2068
ndcg_50_full:0.0841
hitrate_50_half:0.1581
ndcg_50_half:0.0632
改进：
用所有的数据预测phasen
初始版本是用phase0-n的数据预测phasen，而没有用到后面的数据

日期:2020-04-17 15:22:28排名: 无
修改了一行bug
for j, wij in sorted(sim_item_corr[i].items(), reverse=True)[:top_k]:

score:0.0787
hitrate_50_full:0.1996
ndcg_50_full:0.0787
hitrate_50_half:0.1383
ndcg_50_half:0.0556

高手开源baseline
https://tianchi.aliyun.com/forum/postDetail?postId=103530
日期:2020-04-17 10:42:13

score:0.0707
hitrate_50_full:0.1784
ndcg_50_full:0.0707
hitrate_50_half:0.1405
ndcg_50_half:0.0558
"""
import numpy as np
import pandas as pd  
from tqdm import tqdm  
from collections import defaultdict  
import math  
from sklearn.metrics.pairwise import cosine_similarity
import time

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)  # 设置显示数据的最大列数，防止出现省略号…，导致数据显示不全
pd.set_option('expand_frame_repr', False)  # 当列太多时不自动换行

t0 = time.time()
  
def get_sim_item(df, user_col, item_col, use_iif=False, mode='item_item'): 
    """
    use_iif:
        True: 把用户点击过的item总数考虑进去，
            如果用户1和用户2都点击了item[1, 2]，但是用户1一共就点击过10个item，而用户2一共点击过10000个item
            那么在计算item[1, 2]的共现系数时，用户1肯定比用户2的价值大
    """
    user_item_dict = None
    if mode == 'item_item':
        user_item_ = df.groupby(user_col)[item_col].agg(set).reset_index()  
        user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))  

        sim_item = {}  # sim_item[i][j] 表示item_i和item_j共现的次数
        item_cnt = defaultdict(int)  # 统计某个item有多少个user购买过
        for user, items in tqdm(user_item_dict.items()):  
            for i in items:  
                item_cnt[i] += 1  
                sim_item.setdefault(i, {})  
                for relate_item in items:  
                    if i == relate_item:  
                        continue  
                    sim_item[i].setdefault(relate_item, 0)  
                    if not use_iif:  
                        sim_item[i][relate_item] += 1  
                    else:  
                        sim_item[i][relate_item] += 1 / math.log(1 + len(items))  

        sim_item_corr = sim_item.copy()  
        for i, related_items in tqdm(sim_item.items()):  
            for j, cij in related_items.items():  
                # item_i和item_j共现的次数/各自出现的次数之积
                sim_item_corr[i][j] = cij/math.sqrt(item_cnt[i]*item_cnt[j])  
    elif mode == 'user_item':
        click_df = df
        click_df = click_df.drop_duplicates()

        n_users = click_df.user_id.nunique()

        n_items = click_df.item_id.nunique()

        print('gen train_data_matrix begin...')
        train_data_matrix = np.zeros((n_users, n_items))
        user_id_dict, item_id_dict = dict(), dict()
        user_id_dict_inv, item_id_dict_inv = dict(), dict()
        u_cnt, i_cnt = 0, 0
        for line in click_df.itertuples():
            if line.user_id not in user_id_dict:
                user_id_dict[line.user_id] = u_cnt
                user_id_dict_inv[u_cnt] = line.user_id
                u_cnt += 1
            if line.item_id not in item_id_dict:
                item_id_dict[line.item_id] = i_cnt
                item_id_dict_inv[i_cnt] = line.item_id
                i_cnt += 1
#             train_data_matrix[user_id_dict[line.user_id], item_id_dict[line.item_id]] += 1
#         train_data_matrix = np.log(train_data_matrix + 1)
        
        print('cal cosine_similarity begin...')  
        # 所有test中出现的user购买过的item_id
        all_test_item_id = df.loc[df.user_id.isin(click_test.user_id.unique().tolist()), 'item_id'].unique().tolist()
        # 把item_id转换为matrix index
        tt = [item_id_dict[i] for i in all_test_item_id]    
        print(len(tt))
#         t = cosine_similarity(train_data_matrix.T, train_data_matrix[:, tt].T)
        
        print('cal cosine_similarity end')
#         dft = pd.DataFrame(t)
#         dft.to_pickle('dft.pkl')
        dft = pd.read_pickle('dft.pkl')
        print(dft.shape)
        sim_item_corr = dict()
        for i in tqdm(range(dft.shape[1])):
            sim_item_corr[item_id_dict_inv[tt[i]]] = {item_id_dict_inv[ii]: v for ii, v in dft.iloc[:, i].sort_values(ascending=False).iloc[:501].items()}
    return sim_item_corr, user_item_dict  
  

def recommend(sim_item_corr, user_item_dict, user_id, top_k, item_num, weight_mode='linear'):  
    """
    向user_id推荐item_num个该user_id没有买过的item
    """
    rank = {}  
    # 该user_id购买过的items
    dft = whole_click[whole_click.user_id == user_id].sort_values('time').drop_duplicates('item_id', keep='last')
    if weight_mode=='linear':
        dft['t'] = range(dft.shape[0], dft.shape[0] * 2)
        dft['t'] = dft['t'] - dft.shape[0] // 2 - dft.shape[0] // 4
    elif weight_mode=='exp':
        dft['t'] = ((dft['time'] - 0.9837) * 10000).map(math.exp) # 最大值是最小值的15倍左右，score:0.1275
        
    interacted_items = dft['item_id'].tolist()    
    weights = dft['t'].tolist()
#     print(dft.shape[0], dft.head())
    # 遍历该user购买过的items，
    cnt = 0
    for i in interacted_items:  
        # 遍历该item共现最高的top_k个item，把其中用户没有买过的加入推荐列表
#         for j, wij in sorted(sim_item_corr[i].items(), reverse=True)[:top_k]:  
        for j, wij in sorted(sim_item_corr[i].items(), key=lambda d: d[1], reverse=True)[:top_k]:  
            if j not in interacted_items:  
                rank.setdefault(j, 0)  
                rank[j] += wij * weights[cnt]
        cnt += 1
    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]  
  
def get_predict(df, pred_col, top_fill):  
    """
    fill user to 50 items
    逻辑就是如果推荐给用户的items少于50个，就用点击数最高的item补足
    """
    top_fill = [int(t) for t in top_fill.split(',')]  
    scores = [-1 * i for i in range(1, len(top_fill) + 1)]  
    ids = list(df['user_id'].unique())  
    fill_df = pd.DataFrame(ids * len(top_fill), columns=['user_id'])  
    fill_df.sort_values('user_id', inplace=True)  
    fill_df['item_id'] = top_fill * len(ids)  
    fill_df[pred_col] = scores * len(ids)  
    df = df.append(fill_df)  
    df.sort_values(pred_col, ascending=False, inplace=True)  
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')  
    df['rank'] = df.groupby('user_id')[pred_col].rank(method='first', ascending=False)  
    df = df[df['rank'] <= 50]  
    df = df.groupby('user_id')['item_id'].apply(lambda x: ','.join([str(i) for i in x])).str.split(',', expand=True).reset_index()  
    return df  

def load_data(now_phase):
#     now_phase = 2  
    train_path = './data_origin/underexpose_train'  
    test_path = './data_origin/underexpose_test'  
    recom_item = []  

    whole_click = pd.DataFrame()  
    click_train = pd.DataFrame()   
    click_test = pd.DataFrame()  
    test_qtime = pd.DataFrame()  
    for c in range(now_phase + 1):  
        print('phase:', c)  
        click_train1 = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
        click_test1 = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(c, c), header=None,  names=['user_id', 'item_id', 'time'])  
        test_qtime1 = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_qtime-{}.csv'.format(c, c), header=None,  names=['user_id','query_time'])  

        click_train = click_train.append(click_train1) 
    #     all_click = click_train.append(click_test1)  
        click_test = click_test.append(click_test1) 
        test_qtime = test_qtime.append(test_qtime1) 

#     # 去掉 train中time>query_time的数据    
#     click_train = pd.merge(click_train, test_qtime, how='left').fillna(10)  
#     click_train = click_train[click_train.time <= click_train.query_time]
#     del click_train['query_time']
    whole_click = click_train.append(click_test)  
    whole_click = whole_click.drop_duplicates()
    return whole_click, click_train, click_test, test_qtime


if __name__ == '__main__':
    now_phase = 3
    whole_click, click_train, click_test, test_qtime = load_data(now_phase)
    item_sim_list, user_item = get_sim_item(whole_click, 'user_id', 'item_id', use_iif=True, mode='item_item')  


    recom_item = [] 
    for i in tqdm(click_test['user_id'].unique()):  
        rank_item = recommend(item_sim_list, user_item, i, 500, 50)  
        for j in rank_item:  
            recom_item.append([i, j[0], j[1]])  
    # find most popular items  
    top50_click = whole_click['item_id'].value_counts().index[:50].values  
    top50_click = ','.join([str(i) for i in top50_click])  

    recom_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'sim'])  
    result = get_predict(recom_df, 'sim', top50_click)  

    result.to_csv('/Users/luoyonggui/Downloads/baseline1_itemcf3.csv', index=False, header=None)

    t1 = time.time()
    print(f'{t1-t0}s')
    print('complete!!!')