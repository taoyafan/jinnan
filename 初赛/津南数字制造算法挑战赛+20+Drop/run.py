
# In[1]:


import numpy as np 
import pandas as pd 
import lightgbm as lgb
import xgboost as xgb
import warnings
import re
import argparse, sys

# In[2]:


from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid


# In[3]:


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 100)




# # 数据清洗

# ## 删除缺失率高的特征
# 
# + __删除缺失值大于 th_high 的值__
# + __缺失值在 th_low 和 th_high 之间的特征根据是否缺失增加新特征__
#   
#   如 B10 缺失较高，增加新特征 B10_null，如果缺失为1，否则为0

# In[6]:


class del_nan_feature(BaseEstimator, TransformerMixin):
    
    def __init__(self, th_high=0.85, th_low=0.02):
        self.th_high = th_high
        self.th_low = th_low
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print('-'*30, ' '*5, 'del_nan_feature', ' '*5, '-'*30, '\n')
        print("shape before process = {}".format(X.shape))

        # 删除高缺失率特征
        X.dropna(axis=1, thresh=(1-self.th_high)*X.shape[0], inplace=True)
        
        
        # 缺失率较高，增加新特征
        for col in X.columns:
            if col == 'target':
                continue
            
            miss_rate = X[col].isnull().sum()/ X.shape[0]
            if miss_rate > self.th_low:
                print("Missing rate of {} is {:.3f} exceed {}, adding new feature {}".
                     format(col, miss_rate, self.th_low, col+'_null'))
                X[col+'_null'] = 0
                X.loc[X[pd.isnull(X[col])].index, [col+'_null']] = 1
        print("shape = {}".format(X.shape))

        return X


# ## 处理字符时间（段）

# In[7]:


# 处理时间
def timeTranSecond(t):
    try:
        h,m,s=t.split(":")
    except:

        if t=='1900/1/9 7:00':
            return 7*3600/3600
        elif t=='1900/1/1 2:30':
            return (2*3600+30*60)/3600
        elif pd.isnull(t):
            return np.nan
        else:
            return 0

    try:
        tm = (int(h)*3600+int(m)*60+int(s))/3600
    except:
        return (30*60)/3600

    return tm


# In[8]:


# 处理时间差
def getDuration(se):
    try:
        sh,sm,eh,em=re.findall(r"\d+",se)
#         print("sh, sm, eh, em = {}, {}, {}, {}".format(sh, em, eh, em))
    except:
        if pd.isnull(se):
            return np.nan, np.nan, np.nan

    try:
        t_start = (int(sh)*3600 + int(sm)*60)/3600
        t_end = (int(eh)*3600 + int(em)*60)/3600
        
        if t_start > t_end:
            tm = t_end - t_start + 24
        else:
            tm = t_end - t_start
    except:
        if se=='19:-20:05':
            return 19, 20, 1
        elif se=='15:00-1600':
            return 15, 16, 1
        else:
            print("se = {}".format(se))


    return t_start, t_end, tm


# In[9]:


class handle_time_str(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print('-'*30, ' '*5, 'handle_time_str', ' '*5, '-'*30, '\n')

        for f in ['A5','A7','A9','A11','A14','A16','A24','A26','B5','B7']:
            try:
                X[f] = X[f].apply(timeTranSecond)
            except:
                print(f,'应该在前面被删除了！')


        for f in ['A20','A28','B4','B9','B10','B11']:
            try:
                start_end_diff = X[f].apply(getDuration)
                
                X[f+'_start'] = start_end_diff.apply(lambda x: x[0])
                X[f+'_end'] = start_end_diff.apply(lambda x: x[1])
                X[f] = start_end_diff.apply(lambda x: x[2])

            except:
                print(f,'应该在前面被删除了！')
        return X


# ## 计算时间差

# In[ ]:





# In[10]:


def t_start_t_end(t):
    if pd.isnull(t[0]) or pd.isnull(t[1]):
#         print("t_start = {}, t_end = {}, id = {}".format(t[0], t[1], t[2]))
        return np.nan
        
    if t[1] < t[0]:
        t[1] += 24
        
    dt = t[1] - t[0]

    if(dt > 24 or dt < 0):
#         print("dt error, t_start = {}, t_end = {}, id = {}".format(t[0], t[1], t[2]))
        return np.nan
    
    return dt


# In[11]:


class calc_time_diff(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print('-'*30, ' '*5, 'calc_time_diff', ' '*5, '-'*30, '\n')

        # t_start 为时间的开始， tn 为中间的时间，减去 t_start 得到时间差
        t_start = ['A9', 'A24', 'B5']
        tn = {'A9':['A11', 'A14', 'A16'], 'A24': ['A26'], 'B5':['B7']}
        
        # 计算时间差
        for t_s in t_start:
            for t_e in tn[t_s]:
                X[t_e+'-'+t_s] = X[[t_s, t_e, 'target']].apply(t_start_t_end, axis=1)
                
        # 所有结果保留 3 位小数
        X = X.apply(lambda x: round(x, 3))
        
        print("shape = {}".format(X.shape))
        
        return X


# ## 处理异常值

# + __单一类别个数小于 threshold 的值视为异常值, 改为 nan__

# In[12]:


class handle_outliers(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=2):
        self.th = threshold
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print('-'*30, ' '*5, 'handle_outliers', ' '*5, '-'*30, '\n')
        category_col = [col for col in X if col not in ['id', 'target']]
        for col in category_col:
            label = X[col].value_counts(dropna=False).index.tolist()
            for i, num in enumerate(X[col].value_counts(dropna=False).values):
                if num <= self.th:
#                     print("Number of label {} in feature {} is {}".format(label[i], col, num))
                    X.loc[X[col]==label[i], [col]] = np.nan
        
        print("shape = {}".format(X.shape))
        return X


# ## 删除单一类别占比过大特征

# In[13]:


class del_single_feature(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.98):
        # 删除单一类别占比大于 threshold 的特征
        self.th = threshold
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print('-'*30, ' '*5, 'del_single_feature', ' '*5, '-'*30, '\n')
        category_col = [col for col in X if col not in ['target']]
        
        for col in category_col:
            rate = X[col].value_counts(normalize=True, dropna=False).values[0]
            
            if rate > self.th:
                print("{} 的最大类别占比是 {}, drop it".format(col, rate))
                X.drop(col, axis=1, inplace=True)

        print("shape = {}".format(X.shape))
        return X


# # 特征工程

# ## 获得训练集与测试集

# In[14]:


def split_data(pipe_data, target_name='target'):
   
    # 特征列名
    category_col = [col for col in pipe_data if col not in ['target',target_name]]
    
    # 训练、测试行索引
    train_idx = pipe_data[np.logical_not(pd.isnull(pipe_data[target_name]))].index
    test_idx = pipe_data[pd.isnull(pipe_data[target_name])].index
    
    # 获得 train、test 数据
    X_train = pipe_data.loc[train_idx, category_col].values.astype(np.float64)
    y_train = np.squeeze(pipe_data.loc[train_idx, [target_name]].values.astype(np.float64))
    X_test = pipe_data.loc[test_idx, category_col].values.astype(np.float64)
    
    return X_train, y_train, X_test, test_idx


# ## xgb(用于特征 nan 值预测)

# In[15]:


##### xgb
def xgb_predict(X_train, y_train, X_test, params=None, verbose_eval=100):
    
    if params == None:
        xgb_params = {'eta': 0.05, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8, 
                  'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}
    else:
        xgb_params = params

    folds = KFold(n_splits=10, shuffle=True, random_state=2018)
    oof_xgb = np.zeros(len(X_train))
    predictions_xgb = np.zeros(len(X_test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        if(verbose_eval):
            print("fold n°{}".format(fold_+1))
            print("len trn_idx  {}".format(len(trn_idx)))
            
        trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
        val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

        watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
        clf = xgb.train(dtrain=trn_data,
                        num_boost_round=20000,
                        evals=watchlist,
                        early_stopping_rounds=200,
                        verbose_eval=verbose_eval,
                        params=xgb_params)
        
        
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
        predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

    if(verbose_eval):
        print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train)))
    return oof_xgb, predictions_xgb


# ## 根据 B14 构建新特征

# In[16]:


class add_new_features(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print('-'*30, ' '*5, 'add_new_features', ' '*5, '-'*30, '\n')

        # 经过测试,只有 B14 / B12 有用
        
#         X['B14/A1'] = X['B14'] / X['A1']
#         X['B14/A3'] = X['B14'] / X['A3']
#         X['B14/A4'] = X['B14'] / X['A4']
#         X['B14/A19'] = X['B14'] / X['A19']
#         X['B14/B1'] = X['B14'] / X['B1']
#         X['B14/B9'] = X['B14'] / X['B9']

        X['B14/B12'] = X['B14'] / X['B12']
        
        print("shape = {}".format(X.shape))
        return X


# ## 选择特征, nan 值填充
# 
# + __选择可能有效的特征__   (只是为了加快选择时间)
# 
# + __利用其他特征预测 nan，取最近值填充__

# In[17]:


def get_closest(indexes, predicts):
    print("From {}".format(predicts))

    for i, one in enumerate(predicts):
        predicts[i] = indexes[np.argsort(abs(indexes - one))[0]]

    print("To {}".format(predicts))
    return predicts
    

def value_select_eval(pipe_data, selected_features):
    
    # 经过多次测试, 只选择可能是有用的特征
    cols_with_nan = [col for col in pipe_data.columns 
                     if pipe_data[col].isnull().sum()>0 and col in selected_features]

    for col in cols_with_nan:
        X_train, y_train, X_test, test_idx = split_data(pipe_data, target_name=col)
        oof_xgb, predictions_xgb = xgb_predict(X_train, y_train, X_test, verbose_eval=False)
        
        print("-"*100, end="\n\n")
        print("CV normal MAE scores of predicting {} is {}".
              format(col, mean_absolute_error(oof_xgb, y_train)/np.mean(y_train)))
        
        pipe_data.loc[test_idx, [col]] = get_closest(pipe_data[col].value_counts().index,
                                              predictions_xgb)

    pipe_data = pipe_data[selected_features+['target']]

    return pipe_data

# pipe_data = value_eval(pipe_data)


# In[18]:


class selsected_fill_nans(BaseEstimator, TransformerMixin):

    def __init__(self, selected_features = ['A3_null', 'A6', 'A16', 'A25', 'A28', 'A28_end',
                                           'B5', 'B10_null', 'B11_null', 'B14', 'B14/B12', 'id']):
        self.selected_fearutes = selected_features
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print('-'*30, ' '*5, 'selsected_fill_nans', ' '*5, '-'*30, '\n')

        X = value_select_eval(X, self.selected_fearutes)

        print("shape = {}".format(X.shape))
        return X


# In[19]:


def modeling_cross_validation(data):
    X_train, y_train, X_test, test_idx = split_data(data,
                                                    target_name='target')
    oof_xgb, _ = xgb_predict(X_train, y_train, X_test, verbose_eval=False)
    print('-'*100, end='\n\n')
    return mean_squared_error(oof_xgb, y_train)


def featureSelect(data):

    init_cols = [f for f in data.columns if f not in ['target']]
    best_cols = init_cols.copy()
    best_score = modeling_cross_validation(data[best_cols+['target']])
    print("初始 CV score: {:<8.8f}".format(best_score))

    for col in init_cols:
        best_cols.remove(col)
        score = modeling_cross_validation(data[best_cols+['target']])
        print("当前选择特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}".
              format(col, score, best_score), end=" ")
        
        if best_score - score > 0.0000004:
            best_score = score
            print("有效果,删除！！！！")
        else:
            best_cols.append(col)
            print("保留")

    print('-'*100)
    print("优化后 CV score: {:<8.8f}".format(best_score))
    return best_cols, best_score


# ## 后向选择特征

# In[20]:


class select_feature(BaseEstimator, TransformerMixin):

    def __init__(self, init_features = None):
        self.init_features = init_features
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print('-'*30, ' '*5, 'select_feature', ' '*5, '-'*30, '\n')
        
        if self.init_features:
            X = X[self.init_features + ['target']]
            best_features = self.init_features
        else:
            best_features = [col for col in X.columns]
        
        last_feartues = []
        iteration = 0
        equal_time = 0
        
        best_CV = 1
        best_CV_feature = []
        
        # 打乱顺序,但是使用相同种子,保证每次运行结果相同
        np.random.seed(2018)
        while True:
            print("Iteration = {}\n".format(iteration))
            best_features, score = featureSelect(X[best_features + ['target']])
            
            # 保存最优 CV 的参数
            if score < best_CV:
                best_CV = score
                best_CV_feature = best_features
                print("Found best score :{}, with features :{}".format(best_CV, best_features))
                
            np.random.shuffle(best_features)
            print("\nCurrent fearure length = {}".format(len(best_features)))
            
            # 最终 3 次迭代相同，则终止迭代
            if len(best_features) == len(last_feartues):
                equal_time += 1
                if equal_time == 3:
                    break
            else:
                equal_time = 0
            
            last_feartues = best_features
            iteration = iteration + 1

            print("\n\n\n")
            
        return X[best_features + ['target']]


# # 训练

# ## 构建 pipeline, 处理数据

# In[21]:





# ## 自动调参

# In[22]:


def find_best_params(pipe_data, predict_fun, param_grid):
    
    # 获得 train 和 test, 归一化
    X_train, y_train, X_test, test_idx = split_data(pipe_data, target_name='target')
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    best_score = 1

    # 遍历所有参数,寻找最优
    for params in ParameterGrid(param_grid):
        print('-'*100, "\nparams = \n{}\n".format(params))

        oof, predictions = predict_fun(X_train, y_train, X_test, params=params, verbose_eval=False)
        score = mean_squared_error(oof, y_train)
        print("CV score: {}, current best score: {}".format(score, best_score))

        if best_score > score:
            print("Found new best score: {}".format(score))
            best_score = score
            best_params = params


    print('\n\nbest params: {}'.format(best_params))
    print('best score: {}'.format(best_score))
    
    return best_params


# ## lgb

# In[23]:


def lgb_predict(X_train, y_train, X_test, params=None, verbose_eval=100):
    
    if params == None:
        lgb_param = {'num_leaves': 20, 'min_data_in_leaf': 2, 'objective':'regression', 'max_depth': 4,
         'learning_rate': 0.06, "min_child_samples": 3, "boosting": "gbdt", "feature_fraction": 1,
         "bagging_freq": 0.7, "bagging_fraction": 1, "bagging_seed": 11, "metric": 'mse', "lambda_l2": 0.003,
         "verbosity": -1}
    else :
        lgb_param = params
        
    folds = KFold(n_splits=10, shuffle=True, random_state=2018)
    oof_lgb = np.zeros(len(X_train))
    predictions_lgb = np.zeros(len(X_test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        if verbose_eval:
            print("fold n°{}".format(fold_+1))
            
        trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
        val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

        num_round = 10000
        clf = lgb.train(lgb_param, trn_data, num_round, valid_sets = [trn_data, val_data],
                        verbose_eval=verbose_eval, early_stopping_rounds = 100)
        
        oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)

        predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

        if verbose_eval:
            print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, y_train)))
    
    return oof_lgb, predictions_lgb


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_type', type=str,
                        help='Can be B or C, meaning running code with either test B or test C')
    args = parser.parse_args()
    return args


def read_data(train_file_name, test_file_name):
    # 读取数据， 改名
    train = pd.read_csv(train_file_name, encoding='gb18030')
    test = pd.read_csv(test_file_name, encoding='gb18030')
    train.rename(columns={'样本id': 'id', '收率': 'target'}, inplace=True)
    test.rename(columns={'样本id': 'id', '收率': 'target'}, inplace=True)
    target_name = 'target'

    # 存在异常数据，改为 nan
    train.loc[1304, 'A25'] = np.nan
    train['A25'] = train['A25'].astype(float)

    # 去掉 id 前缀
    train['id'] = train['id'].apply(lambda x: int(x.split('_')[1]))
    test['id'] = test['id'].apply(lambda x: int(x.split('_')[1]))

    train.drop(train[train[target_name] < 0.87].index, inplace=True)
    full = pd.concat([train, test], ignore_index=True)
    return full


def feature_processing(full):

    selected_features = ['A3_null', 'A6', 'A16', 'A25', 'A28', 'A28_end',
                         'B5', 'B10_null', 'B11_null', 'B14', 'B14/B12', 'id']
    
    pipe = Pipeline([
        ('del_nan_feature', del_nan_feature()),
        ('handle_time_str', handle_time_str()),
        ('calc_time_diff', calc_time_diff()),
        ('Handle_outliers', handle_outliers(2)),
        ('del_single_feature', del_single_feature(1)),
        ('add_new_features', add_new_features()),
        ('selsected_fill_nans', selsected_fill_nans(selected_features)),
        ('select_feature', select_feature(selected_features)),
    ])
    
    pipe_data = pipe.fit_transform(full.copy())
    print(pipe_data.shape)
    return pipe_data


def train_predict(pipe_data):

    # lgb
    param_grid = [
        {'num_leaves': [20], 'min_data_in_leaf': [2, 3], 'objective': ['regression'],
         'max_depth': [3, 4, 5], 'learning_rate': [0.06, 0.12, 0.24], "min_child_samples": [3],
         "boosting": ["gbdt"], "feature_fraction": [0.7], "bagging_freq": [1],
         "bagging_fraction": [1], "bagging_seed": [11], "metric": ['mse'],
         "lambda_l2": [0.0003, 0.001, 0.003], "verbosity": [-1]}
    ]
    
    lgb_best_params = find_best_params(pipe_data, lgb_predict, param_grid)

    X_train, y_train, X_test, test_idx = split_data(pipe_data, target_name='target')
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    oof_lgb, predictions_lgb = lgb_predict(X_train, y_train, X_test, params=lgb_best_params, verbose_eval=200)  #

    # xgb
    param_grid = [
        {'silent': [1],
         'nthread': [4],
         'eval_metric': ['rmse'],
         'eta': [0.03],
         'objective': ['reg:linear'],
         'max_depth': [4, 5, 6],
         'num_round': [1000],
         'subsample': [0.4, 0.6, 0.8, 1],
         'colsample_bytree': [0.7, 0.9, 1],
         }
    ]
    
    xgb_best_params = find_best_params(pipe_data, xgb_predict, param_grid)

    X_train, y_train, X_test, test_idx = split_data(pipe_data, target_name='target')
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    oof_xgb, predictions_xgb = xgb_predict(X_train, y_train, X_test, params=xgb_best_params, verbose_eval=200)  #
    
    # 模型融合 stacking
    train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
    test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()
    
    folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
    oof_stack = np.zeros(train_stack.shape[0])
    predictions = np.zeros(test_stack.shape[0])
    
    for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, y_train)):
        print("fold {}".format(fold_))
        trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
        val_data, val_y = train_stack[val_idx], y_train[val_idx]
        
        clf_3 = BayesianRidge()
        clf_3.fit(trn_data, trn_y)
        
        oof_stack[val_idx] = clf_3.predict(val_data)
        predictions += clf_3.predict(test_stack) / 10
    
    final_score = mean_squared_error(y_train, oof_stack)
    print(final_score)
    return predictions


def gen_submit(test_file_name, result_name, predictions):
    # 生成提交结果
    sub_df = pd.read_csv(test_file_name, encoding='gb18030')
    sub_df = sub_df[['样本id', 'A1']]
    sub_df['A1'] = predictions
    sub_df['A1'] = sub_df['A1'].apply(lambda x: round(x, 3))
    print("Generating a submit file : {}".format(result_name))
    sub_df.to_csv(result_name, header=0, index=0)

if __name__ == '__main__':
    args = init_config()
    print(args, file=sys.stderr)
    
    if args.test_type in ['B', 'b']:
        test_file_name = 'data/jinnan_round1_testB_20190121.csv'
        result_name = 'submit_B.csv'
    elif args.test_type in ['C', 'c']:
        test_file_name = 'data/jinnan_round1_test_20190121.csv'
        result_name = 'submit_C.csv'
    else:
        raise RuntimeError('Need config of test_type, can be only B or C for example: --test_type=B')

    # 设定文件名, 读取文件
    train_file_name = 'data/jinnan_round1_train_20181227.csv'
    
    print("Training file named {} and testing file named {}".format(train_file_name, test_file_name))

    # read file
    full = read_data(train_file_name, test_file_name)
    
    # feature processing
    pipe_data = feature_processing(full)

    # train and predict
    predictions = train_predict(pipe_data)

    # generate a submit file
    gen_submit(test_file_name, result_name, predictions)
    
    print("Finished")
