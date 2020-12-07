""" Module to generate single puff window sequence and make prediction of smoking with
a smoking probability model
"""
import sys
sys.path.append("../")
import math
import numpy as np
import pandas as pd
import scipy.stats as sp_stats
import smdt.info as s_info
import smdt.loader as s_loader
import smdt.experiment as s_experiment
import sklearn.metrics as metrics
# import matplotlib.pyplot as pyplot

scale_weights = [0.3, 0.3, 0.4]
scale_lengths = [8*60, 4*60, 1*60] # in seconds
mu_n_puffs = 20
average_puff_duration_sec = 5.4
average_interpuff_duration_sec = 32.2
puff_threshold = {1000: 0.3, 360:0.2, 40:0.2}
smoke_threshold = 0.2

def get_single_puff_df_from_prediction(session="all", window_size=1000, sensor='BFW', validation='kfold'):
  print "================generating single puff sequence========================"
  average_puff_duration = math.floor(average_puff_duration_sec * s_info.raw_sample_rate)
  average_interpuff_duration = math.floor(average_interpuff_duration_sec * s_info.raw_sample_rate)
  print "average puff duration (instances): " + str(average_puff_duration)
  print "average average_interpuff_duration (instances): " + str(average_interpuff_duration)
  print "puff threshold: " + str(puff_threshold)
  # single puff window length
  N_single_puff = int((average_interpuff_duration)/window_size/s_info.overlap_rate)
  T_single_puff = average_puff_duration / average_interpuff_duration
  print "Number of instances in a single puff: %d" % N_single_puff 
  print "threshold of single puff: %.2f" % T_single_puff 
  #read in prediction df
  prediction_df = s_loader.load_prediction_df(session=session, window_size=window_size, sensor=sensor, validation=validation)
  # group by session
  grouped_pred_df = prediction_df.groupby(s_info.session_col)
  single_puff_dfs = []
  for group_name, session_pred_df in grouped_pred_df:
    index = range(0,len(session_pred_df)/N_single_puff)
    single_puff_df = pd.DataFrame(columns=prediction_df.columns, index=index)
    c = 0
    for i in range(0, len(index)*N_single_puff, N_single_puff):
      single_puff = session_pred_df.iloc[i:i+N_single_puff]
      # print single_puff.T
      N_pred_puffs = len(np.where(single_puff[s_info.predictionname_col] == 'puff')[0])
      prob_single_puff = np.percentile(single_puff[s_info.predictionprob_col],100 - T_single_puff*100)
      N_true_puffs = len(np.where(single_puff[s_info.classname_col] == 'puff')[0])
      pred_puff_proportion = N_pred_puffs / float(N_single_puff)
      true_puff_proportion = N_true_puffs / float(N_single_puff)

      single_puff_df[s_info.segment_col][c] = single_puff.iloc[0][s_info.segment_col]
      single_puff_df[s_info.st_col][c] = single_puff.iloc[0][s_info.st_col]
      single_puff_df[s_info.et_col][c] = single_puff.iloc[-1][s_info.et_col]
      single_puff_df[s_info.segduration_col][c] = (N_single_puff*s_info.overlap_rate + s_info.overlap_rate)*single_puff.iloc[0][s_info.segduration_col]
      single_puff_df[s_info.session_col][c] = single_puff.iloc[0][s_info.session_col]
      single_puff_df[s_info.sensor_col][c] = single_puff.iloc[0][s_info.sensor_col]
      single_puff_df[s_info.puff_with_segment_duration_col][c] = pred_puff_proportion
      single_puff_df[s_info.smokeproportion_col][c] = single_puff[s_info.smokeproportion_col].fillna(0).mean()
      single_puff_df[s_info.carproportion_col][c] = single_puff[s_info.carproportion_col].fillna(0).mean()
      single_puff_df[s_info.talkproportion_col][c] = single_puff[s_info.talkproportion_col].fillna(0).mean()
      single_puff_df[s_info.unknownproportion_col][c] = single_puff[s_info.unknownproportion_col].fillna(0).mean()
      single_puff_df[s_info.drinkproportion_col][c] = single_puff[s_info.drinkproportion_col].fillna(0).mean()
      single_puff_df[s_info.readproportion_col][c] = single_puff[s_info.readproportion_col].fillna(0).mean()
      single_puff_df[s_info.eatproportion_col][c] = single_puff[s_info.eatproportion_col].fillna(0).mean()
      single_puff_df[s_info.computerproportion_col][c] = single_puff[s_info.computerproportion_col].fillna(0).mean()
      single_puff_df[s_info.phoneproportion_col][c] = single_puff[s_info.phoneproportion_col].fillna(0).mean()
      single_puff_df[s_info.walkproportion_col][c] = single_puff[s_info.walkproportion_col].fillna(0).mean()
      single_puff_df[s_info.spproportion_col][c] = single_puff[s_info.spproportion_col].fillna(0).mean()

      puff_flag = ~np.isnan(single_puff[s_info.prototypcal_col])
      if any(puff_flag):
        single_puff_df[s_info.prototypcal_col][c] = single_puff[puff_flag].iloc[0][s_info.prototypcal_col]
        single_puff_df[s_info.puffindex_col][c] = single_puff[puff_flag].iloc[0][s_info.puffindex_col]
        single_puff_df[s_info.puffside_col][c] = single_puff[puff_flag].iloc[0][s_info.puffside_col]
      single_puff_df[s_info.predictionprob_col][c] = prob_single_puff

      # if pred_puff_proportion >= T_single_puff:
      #   single_puff_df[s_info.predictionname_col][c] = 'puff'
      #   single_puff_df[s_info.predictionnum_col][c] = 1
      # else:
      #   single_puff_df[s_info.predictionname_col][c] = 'no-puff'
      #   single_puff_df[s_info.predictionnum_col][c] = 0

      if true_puff_proportion >= T_single_puff:
        single_puff_df[s_info.classname_col][c] = 'puff'
        single_puff_df[s_info.classnum_col][c] = 9
      else:
        single_puff_df[s_info.classname_col][c] = 'no-puff'
        single_puff_df[s_info.classnum_col][c] = 0
      c +=1
    # do roll mean on prediction probability 
    # print single_puff_df[s_info.predictionprob_col]
    single_puff_df[s_info.predictionprob_col] = pd.rolling_mean(single_puff_df[s_info.predictionprob_col],window=3,min_periods=1)
    # print single_puff_df[s_info.predictionprob_col]
    # refine puff prediction according to prediction probability
    single_puff_df[s_info.predictionname_col] = 'no-puff'
    single_puff_df[s_info.predictionname_col][single_puff_df[s_info.predictionprob_col] > puff_threshold[window_size]] = 'puff'
    single_puff_df[s_info.predictionnum_col] = 0
    single_puff_df[s_info.predictionnum_col][single_puff_df[s_info.predictionprob_col] > puff_threshold[window_size]] = 9
    single_puff_dfs.append(single_puff_df)
  # print prediction_df.head(10).T
  # print pd.concat(single_puff_dfs).head().T
  return pd.concat(single_puff_dfs)

def predict_smoking_state_offline(single_puff_df):
  print "===============predict smoking state==================================="
  print "mu of number of puffs distribution: " + str(mu_n_puffs)
  print "scales (seconds): %s" % scale_lengths
  print "scale weights: %s" % scale_weights
  print "instances in scale: %s" % np.divide(scale_lengths,single_puff_df.iloc[0][s_info.segduration_col])
  grouped_single_puff_df = single_puff_df.groupby(s_info.session_col)
  smoke_predicts = []
  for session_index, session_df in grouped_single_puff_df:
    smoke_predict_dict = {s_info.smokepredprob_col:[]}
    for i in range(0, len(session_df)):
      try:
        smoke_prob = predict_smoking_state_online(session_df.iloc[:i+1])
      except IndexError:
        smoke_prob = predict_smoking_state_online(session_df.iloc[:i])
      smoke_predict_dict[s_info.smokepredprob_col].append(smoke_prob)
    smoke_predicts.append(pd.DataFrame(smoke_predict_dict))
  smoke_df = pd.concat(smoke_predicts)
  # print smoke_df
  # print single_puff_df
  smoke_df = pd.concat([single_puff_df, smoke_df],axis=1)
  return smoke_df

def predict_smoking_state_online(seq_df):
  smoke_prob = 0
  for scale, weight in zip(scale_lengths, scale_weights):
    N_instances = int(scale/seq_df.iloc[0][s_info.segduration_col])
    try:
      scale_df = seq_df.iloc[-1-N_instances:-1]
    except IndexError:
      scale_df = seq_df.iloc[:-1]
    scale_puffs = scale_df[scale_df[s_info.predictionname_col] == 'puff']
    num_of_puffs_inscale = int(len(scale_puffs)/float(scale)*1000)
    num_of_puff_prob = distr_num_of_puffs(num_of_puffs_inscale)
    # print len(scale_puffs)
    # print scale
    # print num_of_puffs_inscale
    # print num_of_puff_prob
    # raw_input('press...')
    smoke_prob += num_of_puff_prob * weight
  return smoke_prob

def export_smoke_df(smoke_df, window_size, sensor, session, validation):
  if session == 'all':
    filename = s_info.smoke_dataset_folder + '/' + 'session' + '_all_' + sensor + '.' + str(window_size) + '.' + validation + '.smoke.csv'
  else:
    filename = s_info.smoke_dataset_folder + '/' + 'session' + str(session) + '.' + str(window_size) + '.' + validation + '.smoke.csv'
  smoke_df.to_csv(filename)
  print filename + ' exported'

def distr_num_of_puffs(num_of_puffs):
  result = sp_stats.poisson.pmf(num_of_puffs, mu_n_puffs)
  return result

#===============================================================================

def test_get_single_puff_df_from_prediction():
  window_size = 1000
  session = 'all'
  sensor = 'BFW'
  validation = 'kfold'
  single_puff_df = get_single_puff_df_from_prediction(window_size=window_size, session=session, sensor=sensor, validation=validation)
  print single_puff_df.head().T
  print single_puff_df.tail().T

def test_predict_smoking_state_offline():
  window_size = 1000
  session = 'all'
  sensor = 'BFW'
  validation = 'kfold'
  single_puff_df = get_single_puff_df_from_prediction(window_size=window_size, session=session, sensor=sensor, validation=validation)
  smoke_df = predict_smoking_state_offline(single_puff_df)
  print smoke_df.head().T
  print smoke_df.tail().T
  # smoke_df.groupby(s_info.session_col)['smoking_pred_prob'].plot(ylim=(0,0.5))
  # pyplot.show()

def test_export_smoke_df():
  window_size = 1000
  session = 'all'
  sensor = 'BFW'
  validation = 'kfold'
  single_puff_df = get_single_puff_df_from_prediction(window_size=window_size, session=session, sensor=sensor, validation=validation)
  smoke_df = predict_smoking_state_offline(single_puff_df)
  export_smoke_df(smoke_df, window_size, sensor, session, validation)

#===============================================================================
def compare_puff_accuracy():
  window_size = 360
  session = 'all'
  sensor = 'BFW'
  validation = 'loso'
  single_puff_df = get_single_puff_df_from_prediction(window_size=window_size, session=session, sensor=sensor, validation=validation)
  prediction_df = s_loader.load_prediction_df(session=session, window_size=window_size, sensor=sensor, validation=validation)

  # compute the puff accuracy of prediction df
  y_true = prediction_df[s_info.classnum_col].values
  y_pred = prediction_df[s_info.predictionnum_col].values
  prototypical_labels = prediction_df[s_info.prototypcal_col].values
  non_proto_accuracy, proto_accuracy = s_experiment.nonproto_puff_accuracy(y_true, y_pred, prototypical_labels)
  y_true[y_true != 9] = 0
  y_pred[y_pred != 9] = 0
  print "Puff Classification level: "
  print "window size: %d, session: %s, sensor: %s, validation: %s" % (window_size, session, sensor, validation)
  print "prototypical puff accuracy: %.2f" % proto_accuracy
  print "non-prototypical puff accuracy: %.2f" % non_proto_accuracy
  print "overall accuracy: %.2f" % (len(y_true[y_true == y_pred])/float(len(y_true)))

  # compute the puff accuracy of single puff df
  y_true = single_puff_df[s_info.classnum_col].values
  y_pred = single_puff_df[s_info.predictionnum_col].values
  prototypical_labels = single_puff_df[s_info.prototypcal_col].values
  # print y_true
  # print y_pred
  non_proto_accuracy, proto_accuracy = s_experiment.nonproto_puff_accuracy(y_true, y_pred, prototypical_labels)
  print "Smoking detection level: "
  print "window size: %d, session: %s, sensor: %s, validation: %s" % (window_size, session, sensor, validation)
  print "prototypical puff accuracy: %.2f" % proto_accuracy
  print "non-prototypical puff accuracy: %.2f" % non_proto_accuracy 
  print "overall accuracy: %.2f" % (len(y_true[y_true == y_pred])/float(len(y_true)))

def report_smoking_prediction():
  window_size = 1000
  session = 'all'
  sensor = 'BFW'
  validation = 'kfold'
  single_puff_df = get_single_puff_df_from_prediction(window_size=window_size, session=session, sensor=sensor, validation=validation)
  smoke_df = predict_smoking_state_offline(single_puff_df)

  # compute ground truth
  N_gt_smoke = len(smoke_df[smoke_df[s_info.smokeproportion_col] >= 0.3])
  print "Ground Truth: number of windows of all smoking: %d" % N_gt_smoke
  N_gt_smoke_c1 = len(smoke_df[(smoke_df[s_info.smokeproportion_col] >= 0.3) & ((smoke_df[s_info.drinkproportion_col] >= 0.25) | (smoke_df[s_info.eatproportion_col] >= 0.25))])
  print "Ground Truth: number of windows of smoking while eating/drinking: %d" % N_gt_smoke_c1
  N_gt_smoke_c2 = len(smoke_df[(smoke_df[s_info.smokeproportion_col] >= 0.3) & (smoke_df[s_info.computerproportion_col] >= 0.3)])
  print "Ground Truth: number of windows of smoking while using computer: %d" % N_gt_smoke_c2
  N_gt_smoke_c3 = len(smoke_df[(smoke_df[s_info.smokeproportion_col] >= 0.3) & (smoke_df[s_info.phoneproportion_col] >= 0.3)])
  print "Ground Truth: number of windows of smoking while using phone: %d" % N_gt_smoke_c3
  N_gt_smoke_c4 = len(smoke_df[(smoke_df[s_info.smokeproportion_col] >= 0.3) & (smoke_df[s_info.talkproportion_col] >= 0.3)])
  print "Ground Truth: number of windows of smoking while using phone: %d" % N_gt_smoke_c4
  N_gt_smoke_c5 = len(smoke_df[(smoke_df[s_info.smokeproportion_col] >= 0.3) & (smoke_df[s_info.spproportion_col] == 0)])
  print "Ground Truth: number of windows of smoking without superposition: %d" % N_gt_smoke_c5

  # compute f1 scores
  y_pred = smoke_df[s_info.smokepredprob_col] >= smoke_threshold / 10
  y_true = smoke_df[s_info.smokeproportion_col] >= smoke_threshold
  print metrics.classification_report(y_true, y_pred, labels=[0,1], target_names=['not smoking', 'smoking'])

  c1_smoke_df = smoke_df[(smoke_df[s_info.drinkproportion_col] >= 0.25) | (smoke_df[s_info.eatproportion_col] >= 0.25)]
  y_pred = c1_smoke_df[s_info.smokepredprob_col] >= smoke_threshold / 10
  y_true = c1_smoke_df[s_info.smokeproportion_col] >= smoke_threshold
  print metrics.classification_report(y_true, y_pred, labels=[0,1], target_names=['not smoking', 'smoking'])

  c2_smoke_df = smoke_df[smoke_df[s_info.computerproportion_col] >= 0.3]
  y_pred = c2_smoke_df[s_info.smokepredprob_col] >= smoke_threshold / 10
  y_true = c2_smoke_df[s_info.smokeproportion_col] >= smoke_threshold
  print metrics.classification_report(y_true, y_pred, labels=[0,1], target_names=['not smoking', 'smoking'])

  c3_smoke_df = smoke_df[smoke_df[s_info.phoneproportion_col] >= 0.3]
  y_pred = c3_smoke_df[s_info.smokepredprob_col] >= smoke_threshold / 10
  y_true = c3_smoke_df[s_info.smokeproportion_col] >= smoke_threshold
  print metrics.classification_report(y_true, y_pred, labels=[0,1], target_names=['not smoking', 'smoking'])

  c4_smoke_df = smoke_df[smoke_df[s_info.talkproportion_col] >= 0.3]
  y_pred = c4_smoke_df[s_info.smokepredprob_col] >= smoke_threshold / 10
  y_true = c4_smoke_df[s_info.smokeproportion_col] >= smoke_threshold
  print metrics.classification_report(y_true, y_pred, labels=[0,1], target_names=['not smoking', 'smoking'])

  c5_smoke_df = smoke_df[smoke_df[s_info.spproportion_col] == 0]
  y_pred = c5_smoke_df[s_info.smokepredprob_col] >= smoke_threshold / 10
  y_true = c5_smoke_df[s_info.smokeproportion_col] >= smoke_threshold
  print metrics.classification_report(y_true, y_pred, labels=[0,1], target_names=['not smoking', 'smoking'])

if __name__ == "__main__":
  # s_info.test_func(test_get_single_puff_df_from_prediction, profile=False)
  # s_info.test_func(test_predict_smoking_state_offline, profile=False)
  # s_info.test_func(test_export_smoke_df, profile=False)
  compare_puff_accuracy()
  # report_smoking_prediction()