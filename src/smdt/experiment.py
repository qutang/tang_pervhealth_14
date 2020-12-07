"""
  Module to run each designed experiment

  It will provide a function to do grid search on any other experiment functions

  Other functions will provide a evaluation summary, detail and customized evaluation
  results will be handled with another module

  Several helper functions would be used to set up sclearn environment
"""

import sys
sys.path.append("../")
from sklearn.externals import joblib
import smdt.info as s_info
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import *
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import numpy.random as random
import pandas as pd
from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from operator import itemgetter

def _split_rest_from_the_remaining(data_df, class_df):
  no_rest_flag = class_df[s_info.classname_col] != 'rest'
  data_df_no_rest = data_df[no_rest_flag]
  class_df_no_rest = class_df[no_rest_flag]
  return data_df_no_rest, class_df_no_rest

#===============================================================================

def load_dataset(session, sensor, window_size):
  """
    1. Load from pkl
    2. fill NA in data dataframe
    3. exclude rest
  """
  if session == 'all':
    data_pkl = s_info.pkl_folder + "/session_all_" + sensor + "." + str(window_size) + ".data.pkl"
    class_pkl = s_info.pkl_folder + "/session_all_" + sensor + "." + str(window_size) + ".class.pkl"
  else:
    data_pkl = s_info.pkl_folder + "/session" + str(session) + "_" + sensor + "." + str(window_size) + ".data.pkl"
    class_pkl = s_info.pkl_folder + "/session" + str(session) + "_" + sensor + "." + str(window_size) + ".class.pkl"
  data_df = joblib.load(data_pkl)
  class_df = joblib.load(class_pkl)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
  #fill na with zero
  data_df = data_df.fillna(value=0.)
  # fill -inf with 1, -inf appears to R square when the sequence is a perfect horizontal line,
  # in this case, it would surely be perfect fit
  # try:
  #   data_df['R squared'] = data_df['R squared'].replace(-np.inf, 1)
  #   data_df['R squared'] = data_df['R squared'].replace(np.inf, 1)
  # except:
  #   data_df['left']['R squared'] = data_df['left']['R squared'].replace(-np.inf, 1)
  #   data_df['right']['R squared'] = data_df['right']['R squared'].replace(-np.inf, 1)
  #   data_df['left']['R squared'] = data_df['left']['R squared'].replace(np.inf, 1)
  #   data_df['right']['R squared'] = data_df['right']['R squared'].replace(np.inf, 1)

  # exclude rest
  print "original dataset size: %d" % len(data_df)
  data_df, class_df = _split_rest_from_the_remaining(data_df, class_df)
  print "dataset size without rest: %d" % len(data_df)
  print "sessions: %s" % class_df[s_info.session_col].unique()
  print "sensors: %s" % class_df[s_info.sensor_col].unique()
  print "window size: " + str(window_size)
  return data_df, class_df

def get_class_num_and_names(class_df):
  class_nums = class_df.target.unique()
  class_names = class_df[s_info.classname_col].unique()
  # sort from low number to high number
  inds = class_nums.argsort()
  class_names = class_names[inds]
  class_nums = class_nums[inds]

  print "class numbers: %s" % class_nums
  print "class names: %s" % class_names
  return class_nums, class_names

def shuffle_dataset(data_df, class_df, random_state=None):
  print "============shuffle dataset starts==================================="
  random.seed(random_state)
  new_index = np.copy(data_df.index)
  random.shuffle(new_index)
  data_df = data_df.copy(deep=True).ix[new_index,:]
  class_df = class_df.copy(deep=True).ix[new_index,:]
  return data_df, class_df

def balance_dataset(X, y, class_nums, class_names, method='subsample', random_state=None, resampling_rate=1):
  """
    Don't use it in testing but only training set, thus the index doesn't matter
  """
  print "===========resampling dataset starts================================="
  n_samples = len(y[y==class_nums[-1]])*resampling_rate
  from sklearn.utils import resample
  Xs = []
  ys = []
  for num, name in zip(class_nums, class_names):
    print name + ": " + str(len(y[y==num])) + ", " + str(len(y[y==num])/float(len(y)))
    if len(y[y==num]) < n_samples:
      Xt = X[y==num,:] 
      yt = y[y==num]
    else:
      try:
        Xt, yt = resample(X[y==num,:], y[y==num], 
                                replace=False, n_samples=n_samples, 
                                random_state=random_state)
        print len(X[y==num,:])
      except:
        sys.exit(1)
    Xs.append(Xt)
    ys.append(yt)
  #combine back
  # print Xs
  # print ys
  resampled_X = np.vstack(tuple(Xs))
  resampled_y = np.hstack(tuple(ys))
  # print resampled_X
  print ""
  for num, name in zip(class_nums, class_names):
    print name + ": " + str(len(resampled_y[resampled_y==num])) + ", " + str(len(resampled_y[resampled_y==num])/float(len(resampled_y)))
  # print "After resampling: others=%d, acts=%d, total=%d" % (len(resampled_y_others), len(y_acts), len(resampled_y))
  return resampled_X, resampled_y

def scale_dataset(X):
  """
  Used on training set
  """
  scaler = MinMaxScaler(copy=True)
  X = scaler.fit_transform(X)
  print scaler
  return X, scaler

def set_classifier(model='svc', params=None):
  """
  For SVC:
  C : float, optional (default=1.0)
      Penalty parameter C of the error term.
  
  gamma : float, optional (default=0.0)
          Kernel coefficient for 'rbf' and 'poly'. If gamma is 0.0 then 1/n_features will be used instead.
  
  For RandomForest:
  n_estimators : integer, optional (default=10)
                 The number of trees in the forest.
  criterion : string, optional (default="gini")
              The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain. Note: this parameter is tree-specific.
  """
  print "============set up classifier===================================="
  if params == None:
    if model == 'svc':
      params = {'C':1, 
                'gamma':0.1,
                'kernel':'rbf'
                }
    elif model == 'rf':
      params = {'n_estimators': 20
               }
  print params
  print "classifier is: " + model
  if model == 'svc':
    result = SVC(probability=True, tol=1e-5)
  elif model == 'rf':
    result = RandomForestClassifier(max_features='auto', compute_importances=True, n_jobs=-1, max_depth=None)
  else:
    result = None
    return
  result.set_params(**params)
  print result
  return result

def set_validation(X, y, method='kfold', params=None):
  """
    method: kfold, skfold, loso, test
    params:
      kfold/skfold: n_folds: number of instances
      kfold: random_state: random seed for shuffle
      loso: labels: labels used to split
  """
  print "===============initialize validation splits=========================="
  if params == None:
    if method == 'kfold':
      params = {'n_folds': 10, 'random_state': 1}
    elif method == 'skfold':
      params = {'n_folds': 10}
    elif method == 'loso':
      params = {'labels': None}
    elif method == 'test':
      params = {'test_size': 0.2, 'random_state': 1}
  print params
  print "validation method is: " + method
  if method == 'kfold':
    cv = KFold(len(y), indices=True, shuffle=False, **params)
  elif method == 'skfold':
    cv = StratifiedKFold(y, indices=True, **params)
  elif method == 'loso':
    print "labels are: %s" % np.unique(params['labels'])
    cv = LeaveOneLabelOut(indices=True, **params)
  else:
    cv = None
  print cv
  return cv

def run_classification(X_train, y_train, X_test, clf):
  """A single run of classification
  """
  # scale train set first
  X_train, scaler = scale_dataset(X_train)
  feature_rank = []
  print X_train.shape
  print X_test.shape
  X_test = scaler.transform(X_test)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  y_log_prob = clf.predict_log_proba(X_test)
  y_prob = clf.predict_proba(X_test)
  feature_rank = clf.feature_importances_
  return y_pred, y_log_prob, y_prob, feature_rank

def run_cross_validation(X, y, cv, class_names, class_nums, clf, balance=False, proto=False, prototypical_labels=None):
  """
  Return:
    y_pred, y_true, y_log_prob, y_prob
  """
  y_tests = []
  y_preds = []
  y_probs = []
  y_log_probs = []
  average_feature_rank = []
  test_inds = []
  c = 1
  for train_ind, test_ind in cv:
    print "============run " + str(c) + " fold validation=================="
    print train_ind
    print test_ind
    test_inds.append(test_ind)
    X_train, X_test = X[train_ind], X[test_ind]
    y_train, y_test = y[train_ind], y[test_ind]
    # balance train split
    if balance:
      X_train, y_train = balance_dataset(X_train, y_train, class_nums, class_names, random_state=None)
    if proto:
      prototypicals = prototypical_labels[train_ind]
      X_train = X_train[np.logical_or(prototypicals!=0, np.logical_and(prototypicals==0, y_train!=9))]
      y_train = y_train[np.logical_or(prototypicals!=0, np.logical_and(prototypicals==0, y_train!=9))]
    from sklearn.base import clone
    y_pred, y_log_prob, y_prob, feature_rank = run_classification(X_train, y_train, X_test, clone(clf))
    y_preds.append(y_pred)
    y_tests.append(y_test)
    y_probs.append(y_prob.T)
    y_log_probs.append(y_log_prob.T)
    average_feature_rank.append(feature_rank)
    c+=1
  print np.shape(y_probs[0]), np.shape(y_probs[1])
  y_pred = np.hstack(tuple(y_preds))
  y_true = np.hstack(tuple(y_tests))
  y_prob = np.hstack(tuple(y_probs))
  y_log_prob = np.hstack(tuple(y_log_probs))
  test_inds = np.hstack(tuple(test_inds))

  sorted_inds = test_inds.argsort()
  print sorted_inds
  print test_inds[sorted_inds]

  y_pred = y_pred[sorted_inds]
  y_true = y_true[sorted_inds]
  y_prob = y_prob[:,sorted_inds]
  y_log_prob = y_log_prob[:,sorted_inds]

  average_feature_rank = np.mean(average_feature_rank, axis=0)
  return y_pred, y_true, y_log_prob, y_prob, average_feature_rank

def get_metric(y_true, y_pred, class_names=None, class_nums=None, prototypical_labels=None, method='f1_score'):
  """
    A single run of metrics computation
  """
  if hasattr(metrics, method) | hasattr(sys.modules[__name__], method):
      if method == 'f1_score':
        score_arr = getattr(metrics, method)(y_true, y_pred, average=None, labels=class_nums)
        score_arr = pd.DataFrame([score_arr], columns=class_names, index=['f1_score',])
      elif method=='puff_f1_score':
        score_arr = puff_f1_score(y_true, y_pred)
      elif method == 'classification_report':
        score_arr = getattr(metrics, method)(y_true, y_pred, target_names=class_names, labels=class_nums)
      elif method == 'confusion_matrix':
        score_arr = getattr(metrics, method)(y_true, y_pred, labels=class_nums)
        score_arr = pd.DataFrame(score_arr, columns=class_names, index=class_names)
      elif method == 'accuracy_score':
        score_arr = getattr(metrics, method)(y_true, y_pred)
        score_arr = pd.Series([score_arr], index=['overall accuracy'])
      elif method == 'nonproto_puff_accuracy':
        score_arr = nonproto_puff_accuracy(y_true, y_pred, prototypical_labels=prototypical_labels)
        score_arr = pd.DataFrame([score_arr], columns=["non-prototpycail accuracy", "prototypical accuracy"])
      elif method == 'nonproto_puff_confusion':
        score_arr1, score_arr2 = nonproto_puff_confusion(y_true, y_pred, prototypical_labels=prototypical_labels, class_names=class_names, class_nums=class_nums)
        score_arr = []
        score_arr.append(pd.DataFrame(score_arr1, columns=class_names, index=class_names))
        score_arr.append(pd.DataFrame(score_arr2, columns=class_names, index=class_names))
      else:
        print method + "Not implemented"
        sys.exit(1)
  else:
    print "Can't find metrics function: " + method
    sys.exit(1)
  return score_arr

def print_dataset_information(X, y, class_nums, class_names):
  print "total size: %d" % len(y)
  info_dict = {'size': [], 'proportion':[]}
  for n, name in zip(class_nums, class_names):
    # print "%s size in traing set: %d" % (name, len(y_train[y_train == n]))
    info_dict['size'].append(len(y[y == n]))
    info_dict['proportion'].append(len(y[y == n])/float(len(y)))
  print pd.DataFrame(info_dict, index=class_names)

def run_parameter_search(X,y, clf, cv, param_dict=None, n_search=20, method='random', scoring='f1'):

  if method == 'random':
    if param_dict == None:
      param_dict = {"max_depth": [3, 5, None],
              "max_features": ['auto',],
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    search = RandomizedSearchCV(clf, scoring=scoring, param_distributions=param_dict, n_iter=n_search, cv=cv, n_jobs=-1, iid=True, verbose=False)

  elif method == 'grid':
    if param_dict == None:
      param_dict = [{"max_depth": [3, 5, None],
                    "max_features": range(1,11),
                    "min_samples_split": range(1,11),
                    "min_samples_leaf": range(1,11),
                    "bootstrap": [True, False],
                    "criterion": ["gini", "entropy"]}]
    search = GridSearchCV(clf, scoring=scoring, param_grid=param_dict, cv=cv, n_jobs=-1, iid=True, verbose=False)

  search.fit(X, y)
  best_setting = search.best_params_

  best_score, std_score = report_search_results(search.grid_scores_)
  return best_setting, best_score, std_score

# Utility function to report best scores
def report_search_results(grid_scores, n_top=10):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        if i == 0:
          best_score = score.mean_validation_score
          std_score = np.std(score.cv_validation_scores)
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
    return best_score, std_score

def build_feature_rank_df(feature_rank_arr, data_df):
  return pd.DataFrame(feature_rank_arr, index=data_df.columns)

def puff_f1_score(y_true, y_pred):
  labels = range(0,10)
  score_arr = metrics.f1_score(y_true, y_pred, labels=labels, average=None)
  puff_score = score_arr[-1]
  return puff_score

def nonproto_puff_accuracy(y_true, y_pred, prototypical_labels):
  nonproto_puff_true = y_true[np.logical_and(prototypical_labels==0, y_true==9)]
  nonproto_puff_pred = y_pred[np.logical_and(prototypical_labels==0, y_true==9)]
  proto_puff_true = y_true[np.logical_and(prototypical_labels==1, y_true==9)]
  proto_puff_pred = y_pred[np.logical_and(prototypical_labels==1, y_true==9)]
  result1 = len(nonproto_puff_pred[nonproto_puff_pred==9])/float(len(nonproto_puff_true))
  result2 = len(proto_puff_pred[proto_puff_pred==9])/float(len(proto_puff_true))
  return [result1, result2]

def nonproto_puff_confusion(y_true, y_pred, prototypical_labels, class_names, class_nums):
  nonproto_puff_pred_arr = y_pred[prototypical_labels == 0]
  nonproto_puff_true_arr = y_true[prototypical_labels == 0]
  non_confusion_df = get_metric(nonproto_puff_true_arr, nonproto_puff_pred_arr, class_names, class_nums, method='confusion_matrix')
  proto_puff_pred_arr = y_pred[prototypical_labels == 1]
  proto_puff_true_arr = y_true[prototypical_labels == 1]
  proto_confusion_df = get_metric(proto_puff_true_arr, proto_puff_pred_arr, class_names, class_nums, method='confusion_matrix')
  return proto_confusion_df, non_confusion_df


puff_f1_scorer = metrics.make_scorer(puff_f1_score, greater_is_better=True)

def convert_num_to_name(y, class_nums, class_names):
  result = np.empty_like(y, dtype='S100')
  for num, name in zip(class_nums, class_names):
    result[y==num] = name
  return result
#===============================================================================
#===============================================================================

def test_load_dataset():
  session = 'all'
  sensor = 'BFW'
  data_df, class_df = load_dataset(session, sensor, window_size=160)
  print "=======test load dataset results================"
  print "sessions: %s" % class_df[s_info.session_col].unique()
  print "sensors: %s" % class_df[s_info.sensor_col].unique()
  print data_df.head().T
  print data_df
  print class_df
  print class_df.head()

def test_get_class_num_and_name():
  session = 3
  sensor = 'DW'
  data_df, class_df = load_dataset(session, sensor)
  class_nums, class_names = get_class_num_and_names(class_df)
 
def test_shuffle_dataset():
  session = 1
  sensor = 'DW'
  data_df, class_df = load_dataset(session, sensor)
  print "=======test shuffle dataset results================"
  print "sessions: %s" % class_df[s_info.session_col].unique()
  print "sensors: %s" % class_df[s_info.sensor_col].unique()
  print class_df
  print "=======before shuffle================================"
  print class_df.head()
  print class_df.index
  print len(class_df)
  data_df, class_df = shuffle_dataset(data_df, class_df, random_state=100)
  print "=======after shuffle================================"
  print class_df.head()
  print class_df.index
  print len(class_df)

def test_balance_dataset():
  session = 'all'
  sensor = 'BW'
  data_df, class_df = load_dataset(session, sensor)
  data_df, class_df = shuffle_dataset(data_df, class_df, random_state=100)
  X = data_df.values
  y = class_df.target.values
  class_nums, class_names = get_class_num_and_names(class_df)
  print "=======test balance dataset results================"
  print "=======before resampling=========================="
  print np.sort(y)
  resampled_X, resampled_y = balance_dataset(X, y, class_nums, class_names, random_state=100)
  print np.sort(resampled_y)
  print len(data_df.columns)
  print len(pd.DataFrame(resampled_X).columns)

def test_scale_dataset():
  session = 4
  sensor = 'DW'
  data_df, class_df = load_dataset(session, sensor)
  data_df, class_df = shuffle_dataset(data_df, class_df, random_state=100)
  X = data_df.values
  print "=======test scale dataset results================"
  print "=======before scaling=========================="
  print np.max(X,axis=0)
  print np.min(X,axis=0)
  X, scaler = scale_dataset(X)
  print "=======after scaling=========================="
  print np.max(X,axis=0)
  print np.min(X,axis=0)

def test_set_classifier():
  print "========test set classifier results====================="
  method = 'svc'
  params = {'C': 1, 'gamma':0.1}
  set_classifier(method, params)
  method = 'rf'
  params = {'n_estimators': 20}
  set_classifier(method, params)

def test_set_validation():
  session = 'all'
  sensor = 'BW'
  data_df, class_df = load_dataset(session, sensor)
  data_df, class_df = shuffle_dataset(data_df, class_df, random_state=100)
  X = data_df.values
  y = class_df.target.values
  session_labels = class_df[s_info.session_col].values
  print "=======test set validation results================"
  cv1 = set_validation(X, y, method='kfold', params={'n_folds':8, 'random_state': 100})
  for train_id, test_id in cv1:
    print train_id
    print test_id
  cv2 = set_validation(X, y, method='skfold', params={'n_folds':8})
  for train_id, test_id in cv2:
    print train_id
    print test_id
  cv3 = set_validation(X, y, method='loso', params={'labels':session_labels})
  for train_id, test_id in cv3:
    print np.unique(session_labels[train_id])
    print np.unique(session_labels[test_id])

def test_print_dataset_info():
  session = 4
  sensor = 'BW'
  data_df, class_df = load_dataset(session, sensor)
  data_df, class_df = shuffle_dataset(data_df, class_df, random_state=100)
  class_nums, class_names = get_class_num_and_names(class_df)
  X = data_df.values
  y = class_df.target.values
  print_dataset_information(X, y, class_nums, class_names)

def test_run_classification():
  session = 'all'
  sensor = 'DW'
  data_df, class_df = load_dataset(session, sensor, window_size=200)
  data_df, class_df = shuffle_dataset(data_df, class_df, random_state=100)
  class_nums, class_names = get_class_num_and_names(class_df)
  X = data_df.values
  y = class_df.target.values  
  X_train, X_test, y_train,  y_true = train_test_split(X, y, test_size=0.2, random_state=None)
  X_train, y_train = balance_dataset(X_train, y_train, class_nums, class_names, random_state=None)
  method = 'rf'
  params = {'n_estimators': 20}
  clf = set_classifier(method, params)

  y_pred, y_log_prob, y_prob, feature_rank = run_classification(X_train, y_train, X_test, clf)
  print get_metric(y_true, y_pred, class_names=class_names, class_nums=class_nums, method='classification_report')
  build_feature_rank_df(feature_rank, data_df)

def test_get_metrics():
  session = 4
  sensor = 'BFW'
  window_size = 40
  data_df, class_df = load_dataset(session, sensor, window_size=window_size)
  data_df, class_df = shuffle_dataset(data_df, class_df, random_state=100)
  class_nums, class_names = get_class_num_and_names(class_df)
  X = data_df.values
  y = class_df.target.values  
  X_train, X_test, y_train,  y_true = train_test_split(X, y, test_size=0.2, random_state=None)

  method = 'rf'
  params = {'n_estimators': 20}
  clf = set_classifier(method, params)

  y_pred, y_log_prob, y_prob, feature_rank = run_classification(X_train, y_train, X_test, clf)
  print get_metric(y_true, y_pred, class_names=class_names, class_nums=class_nums, method='classification_report')
  print get_metric(y_true, y_pred, class_names=class_names, class_nums=class_nums, method='confusion_matrix')
  print get_metric(y_true, y_pred, class_names=class_names, class_nums=class_nums, method='f1_score')
  print get_metric(y_true, y_pred, class_names=class_names, class_nums=class_nums, method='accuracy_score')
  print get_metric(y_true, y_pred, class_names=class_names, class_nums=class_nums, method='puff_f1_score')
def test_run_cross_validation():
  session = 'all'
  sensor = 'BFW'
  window_size=40
  data_df, class_df = load_dataset(session, sensor, window_size=window_size)
  data_df, class_df = shuffle_dataset(data_df, class_df, random_state=100)
  class_nums, class_names = get_class_num_and_names(class_df)
  session_labels = class_df[s_info.session_col].values
  X = data_df.values
  y = class_df.target.values
  cv1 = set_validation(X, y, method='kfold', params={'n_folds': 5, 'random_state': 1})
  cv3 = set_validation(X, y, method='loso', params={'labels':session_labels})
  clf = set_classifier(model='rf', params={'n_estimators':200, 'bootstrap': False, 'min_samples_leaf': 2, 'min_samples_split': 6, 'criterion': 'entropy', 'max_features': 'auto', 'max_depth': None})
  y_pred, y_true, y_log_prob, y_prob, feature_ranks = run_cross_validation(X, y, cv1, class_names, class_nums, clf, balance=False)
  print get_metric(y_true, y_pred, class_names, class_nums, method='classification_report')
  print get_metric(y_true, y_pred, class_names, class_nums, method='confusion_matrix')
  print get_metric(y_true, y_pred, class_names, class_nums, method='accuracy_score')
  print get_metric(y_true, y_pred, class_names, class_nums, method='puff_f1_score')

def test_run_parameter_search():
  session = 1
  sensor = 'NDW'
  window_size = 40
  data_df, class_df = load_dataset(session, sensor, window_size=window_size)
  data_df, class_df = shuffle_dataset(data_df, class_df, random_state=100)
  class_nums, class_names = get_class_num_and_names(class_df)
  session_labels = class_df[s_info.session_col].values
  X = data_df.values
  y = class_df.target.values
  cv1 = set_validation(X, y, method='kfold', params={'n_folds': 5, 'random_state': 1})
  cv2 = set_validation(X, y, method='skfold', params={'n_folds':5})
  cv3 = set_validation(X, y, method='loso', params={'labels':session_labels})
  clf = set_classifier(model='rf', params={'n_estimators':200})
  # clf = set_classifier(model='svc')
  # method = 'svc'
  # params_dict = {'kernel': ['rbf'], 'gamma': [0.1, 1, 10],
  #                    'C': [0.1, 1, 10, 100]}
  # clf = set_classifier(method)
  best_setting, best_score = run_parameter_search(X, y, clf=clf, cv=cv1, n_search=5, method='random', scoring=puff_f1_scorer)

#===============================================================================

def exp_best_setting():
  sessions = [1,3,4,5,6,7]
  sensors = ['BFW']
  results = []
  window_sizes = range(360,30,-40)
  # sessions = [4,]
  # sensors = ['DW',]
  window_sizes = [1000,]
  param_dict = {"max_depth": [None,],
                  "max_features": ['auto',],
                  "min_samples_split": [1,3,5,7,9,11],
                  "min_samples_leaf": [1,3,5,7,9,11],
                  "bootstrap": [True, False,],
                  "criterion": ["gini", "entropy"],
                  'n_estimators':[20,]}
  param_dict = {"max_depth": [None, ],
              "max_features": ['auto',],
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              'n_estimators':[20,]}
  n_search = 20
  method = 'random'
  scoring = puff_f1_scorer
  c = 0
  f = open('../../experiment_results/best_settings.txt','a')
  f.write('\n\n')
  f.write('iteration: %d\n\n' % n_search)
  f.write('results:\n\n')
  f.close()
  for session in sessions:
    for window_size in window_sizes:
      for sensor in sensors:
        f = open('../../experiment_results/best_settings.txt','a')
        data_df, class_df = load_dataset(session, sensor, window_size=window_size)
        data_df, class_df = shuffle_dataset(data_df, class_df, random_state=100)
        class_nums, class_names = get_class_num_and_names(class_df)
        session_labels = class_df[s_info.session_col].values
        X = data_df.values
        y = class_df.target.values
        clf = set_classifier(model='rf')
        cv = set_validation(X, y, method='kfold', params={'n_folds': 5, 'random_state': None})
        cv3 = set_validation(X, y, method='loso', params={'labels':session_labels})
        # run parameter search
        best_setting, best_score, std_score = run_parameter_search(X, y, clf=clf, cv=cv, param_dict=param_dict, n_search=n_search, method=method, scoring=scoring)
        c += 1
        print "best setting:"
        print best_setting
        print "best score:"
        print best_score
        temp = pd.DataFrame(best_setting, index=[c,])
        temp['score mean'] = best_score
        temp['score std'] = std_score
        temp['window size'] = window_size
        temp['sensor'] = sensor
        temp['session'] = session
        temp['dataset size'] = len(data_df)
        temp.to_string(f)
        f.write("\n\n")
        f.close()
        results.append(temp)
    f = open('../../experiment_results/best_settings.txt','a')
    f.write("\n====================================================================\n")
    f.close()
  results = pd.concat(results)
  f = open('../../experiment_results/best_settings.txt','a')
  f.write("consolidate results: \n\n")
  results.to_string(f)
  f.write("\n====================================================================\n")
  f.close()

def exp_all_metrics():
  session = 'all'
  sensors = ['BFW',]
  window_size = 1000
  import json
  param_str = open('../../experiment_results/exp_params.json').read()
  experiment_dict = json.loads(param_str)
  for sensor in sensors:
    session_dict = experiment_dict[session][sensor]
    f = open('../../experiment_results/experiments_metrics.txt','a')
    f.write('\n\n')
    f.write('session: %s\n\n' % session)
    f.write('sensor: %s\n\n' % sensor)
    f.write('window size: %d\n\n' % window_size)
    f.write('validation: %s\n\n' % session_dict['validation'])
    data_df, class_df = load_dataset(session, sensor, window_size=window_size)
    data_df, class_df = shuffle_dataset(data_df, class_df, random_state=100)
    class_nums, class_names = get_class_num_and_names(class_df)
    session_labels = class_df[s_info.session_col].values
    prototypical_labels = class_df[s_info.prototypcal_col].values
    X = data_df.values
    y = class_df.target.values
    clf = set_classifier(model='rf', params=session_dict['clf_params'])
    f.write('clf params: %s\n\n' % clf)
    if session_dict['validation'] == 'kfold':
      cv = set_validation(X, y, method='kfold', params={'n_folds': 5, 'random_state': None})
    elif session_dict['validation'] == 'loso':
      cv = set_validation(X, y, method='loso', params={'labels':session_labels})
    y_pred, y_true, y_log_prob, y_prob, feature_rank = run_cross_validation(X, y, cv, class_names, class_nums, clf, balance=False)
    report_df = get_metric(y_true, y_pred, class_names, class_nums, method='classification_report')
    confusion_df = get_metric(y_true, y_pred, class_names, class_nums, method='confusion_matrix')
    nonproto_df = get_metric(y_true, y_pred, class_names, class_nums, prototypical_labels, method='nonproto_puff_accuracy')
    puff_confusion_dfs = get_metric(y_true, y_pred, class_names, class_nums, prototypical_labels, method='nonproto_puff_confusion')
    f.write(report_df)
    f.write('\n\n')
    confusion_df.to_string(f)
    f.write('\n\n')
    f.write(str(nonproto_df))
    f.write('\n\n')
    f.write('==========prototypical puff confusion matrix\n')
    puff_confusion_dfs[0].to_string(f)
    f.write('\n\n')
    f.write('==========non-prototypical puff confusion matrix\n')
    puff_confusion_dfs[1].to_string(f)
    f.write('\n\n')
    f.write('===================================================================')
    f.close()

def exp_prototypical_model():
  session = 'all'
  sensor = 'BFW'
  window_size = 1000
  import json
  param_str = open('../../experiment_results/exp_params.json').read()
  experiment_dict = json.loads(param_str)
  session_dict = experiment_dict[session][sensor]
  data_df, class_df = load_dataset(session, sensor, window_size=window_size)
  data_df, class_df = shuffle_dataset(data_df, class_df, random_state=100)
  class_nums, class_names = get_class_num_and_names(class_df)
  prototypical_labels = class_df[s_info.prototypcal_col].values
  X = data_df.values
  y = class_df.target.values
  clf = set_classifier(model='rf', params=session_dict['clf_params'])
  if session_dict['validation'] == 'kfold':
    cv = set_validation(X, y, method='kfold', params={'n_folds': 5, 'random_state': None})
  elif session_dict['validation'] == 'loso':
    cv = set_validation(X, y, method='loso', params={'labels':session_labels})
  y_pred, y_true, y_log_prob, y_prob, feature_rank = run_cross_validation(X, y, cv, class_names, class_nums, clf, balance=False, proto=True, prototypical_labels=prototypical_labels)
  report_df = get_metric(y_true, y_pred, class_names, class_nums, method='classification_report')
  confusion_df = get_metric(y_true, y_pred, class_names, class_nums, method='confusion_matrix')
  nonproto_df = get_metric(y_true, y_pred, class_names, class_nums, prototypical_labels, method='nonproto_puff_accuracy')
  print report_df
  print confusion_df
  print nonproto_df

def exp_feature_rank():
  session = 'all'
  sensor = 'BFW'
  window_size = 1000
  import json
  param_str = open('../../experiment_results/exp_params.json').read()
  experiment_dict = json.loads(param_str)
  session_dict = experiment_dict[session][sensor]
  data_df, class_df = load_dataset(session, sensor, window_size=window_size)
  data_df, class_df = shuffle_dataset(data_df, class_df, random_state=100)
  class_nums, class_names = get_class_num_and_names(class_df)
  X = data_df.values
  y = class_df.target.values
  clf = set_classifier(model='rf', params=session_dict['clf_params'])
  if session_dict['validation'] == 'kfold':
    cv = set_validation(X, y, method='kfold', params={'n_folds': 5, 'random_state': None})
  elif session_dict['validation'] == 'loso':
    cv = set_validation(X, y, method='loso', params={'labels':session_labels})
  y_pred, y_true, y_log_prob, y_prob, average_feature_rank = run_cross_validation(X, y, cv=cv, class_names=class_names, class_nums=class_nums, clf=clf)
  feature_rank_df = build_feature_rank_df(average_feature_rank, data_df)
  print feature_rank_df.sort(columns=[0,], ascending=False).head(10)

def exp_output_prediction():
  session = 'all'
  sensor = 'BFW'
  window_sizes = [ 40, ]
  import json
  param_str = open('../../experiment_results/prediction_params.json').read()
  experiment_dict = json.loads(param_str)
  for window_size in window_sizes:
    session_dict = experiment_dict[str(window_size)]
    data_df, class_df = load_dataset(session, sensor, window_size=window_size)
    data_df, class_df = shuffle_dataset(data_df, class_df, random_state=100)
    class_nums, class_names = get_class_num_and_names(class_df)
    session_labels = class_df[s_info.session_col].values
    X = data_df.values
    y = class_df.target.values
    clf = set_classifier(model='rf', params=session_dict['clf_params'])
    if session_dict['validation'] == 'kfold':
      cv = set_validation(X, y, method='kfold', params={'n_folds': 5, 'random_state': None})
    elif session_dict['validation'] == 'loso':
      cv = set_validation(X, y, method='loso', params={'labels':session_labels})
    print class_df.index
    y_pred, y_true, y_log_prob, y_prob, average_feature_rank = run_cross_validation(X, y, cv=cv, class_names=class_names, class_nums=class_nums, clf=clf)
    class_df[s_info.predictionnum_col] = y_pred
    class_df[s_info.predictionname_col] = convert_num_to_name(y_pred, class_nums, class_names)
    class_df[s_info.predictionprob_col] = y_prob[-1,:]
    class_df.sort(inplace=True)
    class_df.to_csv(s_info.prediction_dataset_folder + '/session' + '_all_' + sensor + '.' + str(window_size) + '.' + session_dict['validation'] + '.prediction.csv')

if __name__ == "__main__":
  # s_info.test_func(test_load_dataset, profile=False)
  # s_info.test_func(test_get_class_num_and_name, profile=False)
  # s_info.test_func(test_shuffle_dataset, profile=False)
  # s_info.test_func(test_balance_dataset, profile=False)
  # s_info.test_func(test_scale_dataset, profile=False)
  # s_info.test_func(test_set_classifier, profile=False)
  # s_info.test_func(test_set_validation, profile=False)
  # s_info.test_func(test_run_classification, profile=False)
  # s_info.test_func(test_print_dataset_info, profile=False)
  # s_info.test_func(test_get_metrics, profile=False)
  # s_info.test_func(test_run_cross_validation, profile=False)
  # s_info.test_func(test_run_parameter_search, profile=False)
  # exp_resampling_table()
  # exp_best_setting()
  exp_all_metrics()
  # exp_prototypical_model()
  # exp_feature_rank()
  # exp_output_prediction()