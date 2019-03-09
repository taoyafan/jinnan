# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import warnings
import re
import argparse, sys
import pickle
import os

from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

def init_config():
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_type', type=str,
                      help='Can be B or C, meaning running code with either test B or test C')
  args = parser.parse_args()
  return args


def pkl_load(file_name):
  with open(file_name, "rb") as f:
    return pickle.load(f)


def pkl_save(fname, data, protocol=3):
  with open(fname, "wb") as f:
    pickle.dump(data, f, protocol)
    
    
def load_models():
  lgb_models = pkl_load("models/lgb_models.pkl")
  xgb_models = pkl_load("models/xgb_models.pkl")
  stack_models = pkl_load("models/stack_models.pkl")
  min_max_scaler = pkl_load("models/min_max_scaler.pkl")
  return lgb_models, xgb_models, stack_models, min_max_scaler


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
  _full = pd.concat([train, test], ignore_index=True)
  return _full


class del_nan_feature(BaseEstimator, TransformerMixin):

  def __init__(self, th_high=0.85, th_low=0.02):
    self.th_high = th_high
    self.th_low = th_low

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    print('-' * 30, ' ' * 5, 'del_nan_feature', ' ' * 5, '-' * 30, '\n')
    print("shape before process = {}".format(X.shape))
  
    # 删除高缺失率特征
    X.dropna(axis=1, thresh=(1 - self.th_high) * X.shape[0], inplace=True)
  
    # 缺失率较高，增加新特征
    for col in X.columns:
      if col == 'target':
        continue
    
      miss_rate = X[col].isnull().sum() / X.shape[0]
      if miss_rate > self.th_low:
        print("Missing rate of {} is {:.3f} exceed {}, adding new feature {}".
              format(col, miss_rate, self.th_low, col + '_null'))
        X[col + '_null'] = 0
        X.loc[X[pd.isnull(X[col])].index, [col + '_null']] = 1
    print("shape = {}".format(X.shape))
  
    return X
  
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


# 处理时间差
def getDuration(se):
  try:
    sh, sm, eh, em = re.findall(r"\d+", se)
  #         print("sh, sm, eh, em = {}, {}, {}, {}".format(sh, em, eh, em))
  except:
    if pd.isnull(se):
      return np.nan, np.nan, np.nan
  
  try:
    t_start = (int(sh) * 3600 + int(sm) * 60) / 3600
    t_end = (int(eh) * 3600 + int(em) * 60) / 3600
    
    if t_start > t_end:
      tm = t_end - t_start + 24
    else:
      tm = t_end - t_start
  except:
    if se == '19:-20:05':
      return 19, 20, 1
    elif se == '15:00-1600':
      return 15, 16, 1
    else:
      print("se = {}".format(se))
  
  return t_start, t_end, tm


class handle_time_str(BaseEstimator, TransformerMixin):

  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    print('-' * 30, ' ' * 5, 'handle_time_str', ' ' * 5, '-' * 30, '\n')
  
    for f in ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']:
      try:
        X[f] = X[f].apply(timeTranSecond)
      except:
        print(f, '应该在前面被删除了！')
  
    for f in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
      try:
        start_end_diff = X[f].apply(getDuration)
      
        X[f + '_start'] = start_end_diff.apply(lambda x: x[0])
        X[f + '_end'] = start_end_diff.apply(lambda x: x[1])
        X[f] = start_end_diff.apply(lambda x: x[2])
    
      except:
        print(f, '应该在前面被删除了！')
    return X


def t_start_t_end(t):
  if pd.isnull(t[0]) or pd.isnull(t[1]):
    #         print("t_start = {}, t_end = {}, id = {}".format(t[0], t[1], t[2]))
    return np.nan

  if t[1] < t[0]:
    t[1] += 24

  dt = t[1] - t[0]

  if (dt > 24 or dt < 0):
    #         print("dt error, t_start = {}, t_end = {}, id = {}".format(t[0], t[1], t[2]))
    return np.nan

  return dt


class calc_time_diff(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    print('-' * 30, ' ' * 5, 'calc_time_diff', ' ' * 5, '-' * 30, '\n')
  
    # t_start 为时间的开始， tn 为中间的时间，减去 t_start 得到时间差
    t_start = ['A9', 'A24', 'B5']
    tn = {'A9': ['A11', 'A14', 'A16'], 'A24': ['A26'], 'B5': ['B7']}
  
    # 计算时间差
    for t_s in t_start:
      for t_e in tn[t_s]:
        X[t_e + '-' + t_s] = X[[t_s, t_e, 'target']].apply(t_start_t_end, axis=1)
  
    # 所有结果保留 3 位小数
    X = X.apply(lambda x: round(x, 3))
  
    print("shape = {}".format(X.shape))
  
    return X


class handle_outliers(BaseEstimator, TransformerMixin):

  def __init__(self, threshold=2):
    self.th = threshold

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    print('-' * 30, ' ' * 5, 'handle_outliers', ' ' * 5, '-' * 30, '\n')
    category_col = [col for col in X if col not in ['id', 'target']]
    for col in category_col:
      label = X[col].value_counts(dropna=False).index.tolist()
      for i, num in enumerate(X[col].value_counts(dropna=False).values):
        if num <= self.th:
          #                     print("Number of label {} in feature {} is {}".format(label[i], col, num))
          X.loc[X[col] == label[i], [col]] = np.nan
  
    print("shape = {}".format(X.shape))
    return X


def split_data(pipe_data, target_name='target', unused_feature=[], min_max_scaler=None):

  # 特征列名
  category_col = [col for col in pipe_data if col not in ['target'] + [target_name] + unused_feature]

  # 训练、测试行索引
  # 训练集只包括存在 target 和 target_name 的数据
  train_idx = pipe_data[np.logical_not(
    np.logical_or(pd.isnull(pipe_data[target_name]), pd.isnull(pipe_data['target']))
  )].index

  test_idx = pipe_data[pd.isnull(pipe_data[target_name])].index

  # 获得 train、test 数据
  X_train = pipe_data.loc[train_idx, category_col].values.astype(np.float64)
  y_train = np.squeeze(pipe_data.loc[train_idx, [target_name]].values.astype(np.float64))
  X_test = pipe_data.loc[test_idx, category_col].values.astype(np.float64)

  # 归一化
  if (min_max_scaler == None):
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
  else:
    X_train = min_max_scaler.transform(X_train)
  X_test = min_max_scaler.transform(X_test)

  return X_train, y_train, X_test, test_idx, min_max_scaler


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
  xgb_models = []
  
  for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    if (verbose_eval):
      print("fold n°{}".format(fold_ + 1))
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
    
    xgb_models.append(clf)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
  
  if (verbose_eval):
    print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train)))
  return oof_xgb, predictions_xgb, xgb_models


def lgb_predict(X_train, y_train, X_test, params=None, verbose_eval=100):

  if params == None:
    lgb_param = {'num_leaves': 20, 'min_data_in_leaf': 2, 'objective': 'regression', 'max_depth': 5,
                 'learning_rate': 0.24, "min_child_samples": 3, "boosting": "gbdt", "feature_fraction": 0.7,
                 "bagging_freq": 1, "bagging_fraction": 1, "bagging_seed": 11, "metric": 'mse', "lambda_l2": 0.003,
                 "verbosity": -1}
  else:
    lgb_param = params

  folds = KFold(n_splits=10, shuffle=True, random_state=2018)
  oof_lgb = np.zeros(len(X_train))
  predictions_lgb = np.zeros(len(X_test))
  lgb_models = []

  for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    if verbose_eval:
      print("fold n°{}".format(fold_ + 1))
  
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])
  
    num_round = 10000
    clf = lgb.train(lgb_param, trn_data, num_round, valid_sets=[trn_data, val_data],
                    verbose_eval=verbose_eval, early_stopping_rounds=100)
  
    lgb_models.append(clf)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
  
    if verbose_eval:
      print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, y_train)))

  return oof_lgb, predictions_lgb, lgb_models


class add_new_features(BaseEstimator, TransformerMixin):

  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    print('-' * 30, ' ' * 5, 'add_new_features', ' ' * 5, '-' * 30, '\n')
  
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


def predict(data, lgb_models, xgb_models, stack_models, min_max_scaler):
  
  def _predict(X):
    lgb_result = 0
    xgb_result = 0
    stack_result = 0
    
    for clf in lgb_models:
      lgb_result += clf.predict(X, num_iteration=clf.best_iteration) / len(lgb_models)
    
    for clf in xgb_models:
      xgb_result += clf.predict(xgb.DMatrix(X), ntree_limit=clf.best_ntree_limit) / len(xgb_models)
    
    test_stack = np.vstack([lgb_result, xgb_result]).transpose()
    for clf in stack_models:
      stack_result += clf.predict(test_stack) / len(stack_models)
    
    return stack_result

  _, _, X_test, test_idx, _ = split_data(data, min_max_scaler=min_max_scaler)
  result = _predict(X_test)
  return result


def feature_processing(full, outlier_th = 3):
  selected_features = ['A22', 'A28', 'A20_end', 'B10', 'B11_start', 'A5', 'A10', 'B14/B12', 'B14']
  pipe = Pipeline([
                  ('del_nan_feature', del_nan_feature()),
                  ('handle_time_str', handle_time_str()),
                  ('calc_time_diff', calc_time_diff()),
                  ('Handle_outliers', handle_outliers(outlier_th)),
                  ('add_new_features', add_new_features()),
                  ])
  
  pipe_data = pipe.fit_transform(full.copy())[selected_features+['target']]
  print(pipe_data.shape)
  return pipe_data


def gen_submit(test_file_name, result_name, predictions):
  # 生成提交结果
  sub_df = pd.read_csv(test_file_name, encoding='gb18030')
  sub_df = sub_df[['样本id', 'A1']]
  sub_df['A1'] = predictions
  sub_df['A1'] = sub_df['A1'].apply(lambda x: round(x, 3))
  print("Generating a submit file : {}".format(result_name))
  sub_df.to_csv(result_name, header=0, index=0)


def find_best_params(pipe_data, predict_fun, param_grid):

  # 获得 train 和 test, 归一化
  X_train, y_train, X_test, test_idx, _ = split_data(pipe_data, target_name='target')
  best_score = 1

  # 遍历所有参数,寻找最优
  for params in ParameterGrid(param_grid):
    print('-' * 100, "\nparams = \n{}\n".format(params))
  
    oof, predictions, _ = predict_fun(X_train, y_train, X_test, params=params, verbose_eval=False)
    score = mean_squared_error(oof, y_train)
    print("CV score: {}, current best score: {}".format(score, best_score))
  
    if best_score > score:
      print("Found new best score: {}".format(score))
      best_score = score
      best_params = params

  print('\n\nbest params: {}'.format(best_params))
  print('best score: {}'.format(best_score))

  return best_params


def stacking_predict(oof_lgb, oof_xgb, predictions_lgb, predictions_xgb, y_train, verbose_eval=1):
  # stacking
  train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
  test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

  folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
  oof_stack = np.zeros(train_stack.shape[0])
  predictions = np.zeros(test_stack.shape[0])
  stack_models = []

  for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, y_train)):
    if verbose_eval:
      print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
    val_data, val_y = train_stack[val_idx], y_train[val_idx]
  
    clf_3 = BayesianRidge()
    clf_3.fit(trn_data, trn_y)
    stack_models.append(clf_3)
  
    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10

  final_score = mean_squared_error(y_train, oof_stack)
  if verbose_eval:
    print(final_score)
  return oof_stack, predictions, final_score, stack_models


def train_predict(pipe_data, lgb_best_params, xgb_best_params, verbose_eval=200):
  X_train, y_train, X_test, test_idx, min_max_scaler = split_data(pipe_data, target_name='target')

  oof_lgb, predictions_lgb, lgb_models = lgb_predict(X_train, y_train, X_test,
                                                     params=lgb_best_params, verbose_eval=verbose_eval)
  if verbose_eval:
    print('-' * 100)
  oof_xgb, predictions_xgb, xgb_models = xgb_predict(X_train, y_train, X_test,
                                                     params=xgb_best_params, verbose_eval=verbose_eval)
  if verbose_eval:
    print('-' * 100)
  oof_stack, predictions, final_score, stack_models = stacking_predict(oof_lgb, oof_xgb,
                                                                       predictions_lgb, predictions_xgb, y_train,
                                                                       verbose_eval=verbose_eval)

  return oof_stack, predictions, final_score, lgb_models, xgb_models, stack_models, min_max_scaler


def train_models():
  full = read_data('data/jinnan_round1_train_20181227.csv', 'data/jinnan_round1_testB_20190121.csv')
  pipe_data = feature_processing(full)

  param_grid = [
    {'num_leaves': [20], 'min_data_in_leaf': [2, 3], 'objective': ['regression'],
     'max_depth': [3, 4, 5], 'learning_rate': [0.06, 0.12, 0.24], "min_child_samples": [3],
     "boosting": ["gbdt"], "feature_fraction": [0.7], "bagging_freq": [1],
     "bagging_fraction": [1], "bagging_seed": [11], "metric": ['mse'],
     "lambda_l2": [0.0003, 0.001, 0.003], "verbosity": [-1]}
  ]

  lgb_best_params = find_best_params(pipe_data, lgb_predict, param_grid)

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

  oof_stack, predictions, final_score, lgb_models, xgb_models, stack_models, min_max_scaler = train_predict(
      pipe_data, lgb_best_params, xgb_best_params)

 
  if not os.path.exists('models'):
    os.makedirs('models')
    
  pkl_save("models/lgb_models.pkl", lgb_models)
  pkl_save("models/xgb_models.pkl", xgb_models)
  pkl_save("models/stack_models.pkl", stack_models)
  pkl_save("models/min_max_scaler.pkl", min_max_scaler)

if __name__ == '__main__':
  args = init_config()
  print(args, file=sys.stderr)
  outlier_th = 3
  
  if args.test_type in ['B', 'b']:
    test_file_name = 'data/jinnan_round1_testB_20190121.csv'
    result_name = 'submit_B.csv'
  elif args.test_type in ['C', 'c']:
    test_file_name = 'data/jinnan_round1_test_20190201.csv'
    result_name = 'submit_C.csv'
  elif args.test_type in ['A', 'a']:
    test_file_name = 'data/jinnan_round1_testA_20181227.csv'
    result_name = 'submit_A.csv'
  elif args.test_type in ['fusai', 'FuSai', 'Fusai', 'fuSai']:
    test_file_name = 'data/FuSai.csv'
    result_name = 'submit_FuSai.csv'
  elif args.test_type in ['gen', 'Gen', 'GEN']:
    test_file_name = 'data/optimize.csv'
    result_name = 'submit_optimize.csv'
    outlier_th = 0
  else:
    raise RuntimeError('Need config of test_type, can be only B or C for example: --test_type=B')

  # 设定文件名, 读取文件
  train_file_name = 'data/jinnan_round1_train_20181227.csv'

  print("Training file named {} and testing file named {}".format(train_file_name, test_file_name))

  print("Generating training models")
  train_models()
  print("Saving training models to file: \'models\'")

  full = read_data(train_file_name, test_file_name)
  lgb_models, xgb_models, stack_models, min_max_scaler = load_models()
  # feature processing
  pipe_data = feature_processing(full, outlier_th=outlier_th)

  # train and predict
  predictions = predict(pipe_data, lgb_models, xgb_models, stack_models, min_max_scaler)

  # generate a submit file
  gen_submit(test_file_name, result_name, predictions)

  print("Finished")

