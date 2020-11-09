# AUTOGENERATED! DO NOT EDIT! File to edit: algo-ml-tree-catboost.ipynb (unless otherwise specified).

__all__ = ['val', 'train_val', 'predict', 'explain']

# Cell
from sklearn.metrics import classification_report

from catboost import Pool, CatBoostClassifier

import pandas as pd
import matplotlib.pyplot as plt


# Cell
def val(model, df_val, y_val, cat_cols, threshold=0.5):
    test_data = Pool(data=df_val,
                  cat_features=cat_cols)
    dfr = pd.DataFrame(y_val)
    dfr.columns = ['true_label']
    y_test_hat = model.predict_proba(test_data)[:, 1]
    dfr['score'] = y_test_hat
    dfr['predict_label'] = 0
    dfr.loc[dfr.score>=threshold, 'predict_label'] = 1
    print(classification_report(y_val, dfr['predict_label']))
    print(dfr['predict_label'].value_counts())
    dfr = dfr.sort_values('score', ascending=False)
    dfr['order'] = range(1, dfr.shape[0] + 1)
    print(dfr[(dfr.true_label == 1)|(dfr.predict_label==1)])
    return dfr

# Cell
def train_val(X_train, y_train, cat_cols=[], params=None, valset=None, plot=False, threshold=0.5):
    """
    :threshold: 预测的阈值
    """

    train_data = Pool(data=X_train,
                   label=y_train,
                   cat_features=cat_cols)
    if valset:
        X_val, y_val = valset
        val_data = Pool(data=X_val, label=y_val, cat_features=cat_cols)
    else: val_data = None
    if params is None:
        params = {
        'iterations': 500,
        'learning_rate': 0.05,
        'random_seed': 144,
        'custom_metric': 'F1',
        'loss_function': 'Logloss',
#         'class_weights': [1, 8],
        }
    print(params)
    model = CatBoostClassifier(**params)
    r = model.fit(train_data, eval_set=val_data, verbose=False, plot=plot)
    df_features_importance = pd.DataFrame({'name': model.feature_names_,
                                        'value': model.feature_importances_})
    df_features_importance = df_features_importance.sort_values('value', ascending=False)

    df_features_importance.reset_index(drop=True, inplace=True)

    if plot:

        fea_ = df_features_importance.sort_values('value')[df_features_importance.value > 0].value
        fea_name = df_features_importance.sort_values('value')[df_features_importance.value > 0].name
        plt.figure(figsize=(10, 20))
        plt.barh(fea_name, fea_, height=0.5)
        plt.show()
    else: print(df_features_importance.head(20))
    if valset:
        dfr = val(model, X_val, y_val, cat_cols, threshold)
    else: dfr=None
    return model, df_features_importance, r.best_iteration_


# Cell
def predict(model,X_test,cat_cols, threshold=0.5):
    test_data=Pool(data=X_test,cat_features=cat_cols)
    dfr=pd.DataFrame(index=X_test.index)
    y_test_hat=model.predict_proba(test_data)[:,1]
    dfr['score']=y_test_hat
    dfr['predict_label'] = 0
    dfr.loc[dfr.score>=threshold, 'predict_label']=1
#     dfr.sort_values("score", ascending=False, inplace=True)
    print('--------------------------------------------------')
    s=dfr['predict_label'].value_counts()
    print(s)
    return dfr

# Cell
def explain(model,df_predict,cat_cols,dfr):
    test_data=Pool(data=df_predict,cat_features=cat_cols)
    shap_values=model.get_feature_importance(test_data,type='ShapValues')
    dfs=pd.DataFrame(shap_values[:,:-1],columns=df_predict.columns,index=df_predict['CHANGE_ID'])
    dfs_T=dfs.T
    ss=[]
    for i in range(dfs_T.shape[1]):
        ss.append(dfs_T.iloc[:,i].copy().sort_values(ascending=False).iloc[:5])
    count=0
    rr=[]
    for line in dfr[dfr.predict_label==1].itertuples():
        rr.append({"change_id":line.CHANGE_ID,"FS_SC_NM":"个险模型","FS_SC_SCORE":round(line.score,2),"FS_SC_EXPLAIN":','.join([f'{i[0]}:{round(i[1], 2)}' for i in list(zip(ss[count].index,ss[count].values))])})
    count+=1
    print(rr)
    return rr