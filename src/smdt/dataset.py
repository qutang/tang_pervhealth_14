"""
  Module to connect all dataset from sessions and sensors together

  It will provide three functions:
  The first is to load serialized data_df, class_df for each session and sensor and
  combine them together and serialize them

  The second is to load raw csv files and raw annotation files provided sessions and
  sensors and combine them together, it will actually refresh pkl files and use the first
  function

  The thrid is to combine both left and right wrist sensor dataset into one
"""

import sys
sys.path.append("../")
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import smdt.info as s_info
import smdt.segment as s_segment
import smdt.loader as s_loader
import smdt.feature as s_feature
import smdt.prepare as s_prepare

def combine_both_side_wrist_dataset(sessions='all', window_size=200):
  """ Load pkl from left and right wrists and combine them

  Args:
    sessions: default is all, otherwise a number array for session numbers
  Return:
    both_data_pkl_names, both_class_pkl_names
  """
  print "============start merge left and right side dataset=================="
  print "sessions: %s" % sessions
  both_data_pkl_names = []
  both_class_pkl_names = []
  if type(sessions) == str and sessions == 'all':
    data_dfs = []
    class_dfs = []
    for sensor in ['DW', 'NDW']:
      data_pkl = s_info.pkl_folder + '/session_all_' + sensor + "." + str(window_size) + '.data.pkl'
      data_dfs.append(joblib.load(data_pkl))
      class_pkl = s_info.pkl_folder + '/session_all_' + sensor + "." + str(window_size) + '.class.pkl'
      class_dfs.append(joblib.load(class_pkl))
    # concat
    both_data_df = pd.concat(data_dfs)
    both_class_df = pd.concat(class_dfs)
    # reset index
    both_data_df = both_data_df.reset_index(drop=True)
    both_class_df = both_class_df.reset_index(drop=False)
    both_class_df = both_class_df.rename(columns={'index':'single session and sensor index'})
    # serialize
    both_data_pkl_name = s_info.pkl_folder + '/session_all_BW' + "." + str(window_size) + '.data.pkl'
    both_class_pkl_name = s_info.pkl_folder + '/session_all_BW' + "." + str(window_size) + '.class.pkl'
    both_data_pkl_names.append(both_data_pkl_name)
    both_class_pkl_names.append(both_class_pkl_name)
    joblib.dump(both_data_df, both_data_pkl_name)
    joblib.dump(both_class_df, both_class_pkl_name)
    print "Generated pkls: " + both_data_pkl_name + ", " + both_class_pkl_name
  else:
    for session in sessions:
      data_dfs = []
      class_dfs = []
      for sensor in ['DW', 'NDW']:
        data_pkl = s_info.pkl_folder + '/session' + str(session) + '_' + sensor + "." + str(window_size) + '.data.pkl'
        data_dfs.append(joblib.load(data_pkl))
        class_pkl = s_info.pkl_folder + '/session' + str(session) + '_' + sensor + "." + str(window_size) + '.class.pkl'
        class_dfs.append(joblib.load(class_pkl))
      # concat
      both_data_df = pd.concat(data_dfs)
      both_class_df = pd.concat(class_dfs)
      # reset index
      both_data_df = both_data_df.reset_index(drop=True)
      both_class_df = both_class_df.reset_index(drop=False)
      both_class_df = both_class_df.rename(columns={'index':'single session and sensor index'})
      # serialize
      both_data_pkl_name = s_info.pkl_folder + '/session' + str(session) + '_BW' + "." + str(window_size) + '.data.pkl'
      both_class_pkl_name = s_info.pkl_folder + '/session' + str(session) + '_BW' + "." + str(window_size) + '.class.pkl'
      both_data_pkl_names.append(both_data_pkl_name)
      both_class_pkl_names.append(both_class_pkl_name)
      joblib.dump(both_data_df, both_data_pkl_name)
      joblib.dump(both_class_df, both_class_pkl_name)
      print "Generated pkls: " + both_data_pkl_name + ", " + both_class_pkl_name
  return both_data_pkl_names, both_class_pkl_names

def merge_both_side_wrist_dataset_as_features(sessions='all', window_size=200):
  """ Load pkl from left and right wrists and merge them by columns instead of appending as instances

  Args:
    sessions: default is all, otherwise a number array for session numbers
  Return:
    both_data_pkl_names, both_class_pkl_names
  """
  print "============start merge left and right side dataset=================="
  print "sessions: %s" % sessions
  both_data_pkl_names = []
  both_class_pkl_names = []
  if type(sessions) == str and sessions == 'all':
    data_dfs = []
    class_dfs = []
    for sensor in ['DW', 'NDW']:
      data_pkl = s_info.pkl_folder + '/session_all_' + sensor + "." + str(window_size) + '.data.pkl'
      data_dfs.append(joblib.load(data_pkl))
      class_pkl = s_info.pkl_folder + '/session_all_' + sensor + "." + str(window_size) + '.class.pkl'
      class_dfs.append(joblib.load(class_pkl))
    # concat
    both_data_df = pd.concat(data_dfs, axis=1, keys=['left','right'])
    both_class_df = class_dfs[0].copy(deep=True)
    # for class keep things unchange but merge puff related columns
    # assign "DW" to sensor column
    both_class_df[s_info.sensor_col] = 'BFW'
    # merge puff proportion
    left_side_flag = ~np.isnan(class_dfs[1][s_info.puffindex_col])
    both_class_df[s_info.puff_with_puff_duration_col][left_side_flag] = class_dfs[1][s_info.puff_with_puff_duration_col][left_side_flag]
    both_class_df[s_info.puff_with_segment_duration_col][left_side_flag] = class_dfs[1][s_info.puff_with_segment_duration_col][left_side_flag]
    both_class_df[s_info.prototypcal_col][left_side_flag] = class_dfs[1][s_info.prototypcal_col][left_side_flag]    
    both_class_df[s_info.puffindex_col][left_side_flag] = class_dfs[1][s_info.puffindex_col][left_side_flag]
    both_class_df[s_info.puffside_col][left_side_flag] = class_dfs[1][s_info.puffside_col][left_side_flag]

    # merge class name and target
    class_assign_flag = class_dfs[1][s_info.classnum_col] > both_class_df[s_info.classnum_col]
    both_class_df[s_info.classnum_col][class_assign_flag] = class_dfs[1][s_info.classnum_col][class_assign_flag]
    both_class_df[s_info.classname_col][class_assign_flag] = class_dfs[1][s_info.classname_col][class_assign_flag]

    # print len(class_dfs[0][class_dfs[0][s_info.classnum_col] == 10])
    # print len(class_dfs[1][class_dfs[1][s_info.classnum_col] == 10])
    # print len(both_class_df[both_class_df[s_info.classnum_col] == 10])
    # sys.exit(1)
    # reset index
    both_data_df = both_data_df.reset_index(drop=True)
    both_class_df = both_class_df.reset_index(drop=False)
    both_class_df = both_class_df.rename(columns={'index':'single session and sensor index'})
    # serialize
    both_data_pkl_name = s_info.pkl_folder + '/session_all_BFW' + "." + str(window_size) + '.data.pkl'
    both_class_pkl_name = s_info.pkl_folder + '/session_all_BFW' + "." + str(window_size) + '.class.pkl'
    both_data_pkl_names.append(both_data_pkl_name)
    both_class_pkl_names.append(both_class_pkl_name)
    joblib.dump(both_data_df, both_data_pkl_name)
    joblib.dump(both_class_df, both_class_pkl_name)
    print "Generated pkls: " + both_data_pkl_name + ", " + both_class_pkl_name
  else:
    for session in sessions:
      data_dfs = []
      class_dfs = []
      for sensor in ['DW', 'NDW']:
        data_pkl = s_info.pkl_folder + '/session' + str(session) + '_' + sensor + "." + str(window_size) + '.data.pkl'
        data_dfs.append(joblib.load(data_pkl))
        class_pkl = s_info.pkl_folder + '/session' + str(session) + '_' + sensor + "." + str(window_size) + '.class.pkl'
        class_dfs.append(joblib.load(class_pkl))
      # concat
      both_data_df = pd.concat(data_dfs, axis=1, keys=['left','right'])
      both_class_df = class_dfs[0].copy(deep=True)
      # for class keep things unchange but merge puff related columns
      # assign "DW" to sensor column
      both_class_df[s_info.sensor_col] = 'BFW'
      # merge puff proportion
      left_side_flag = ~np.isnan(class_dfs[1][s_info.puffindex_col])
      both_class_df[s_info.puff_with_puff_duration_col][left_side_flag] = class_dfs[1][s_info.puff_with_puff_duration_col][left_side_flag]
      both_class_df[s_info.puff_with_segment_duration_col][left_side_flag] = class_dfs[1][s_info.puff_with_segment_duration_col][left_side_flag]
      both_class_df[s_info.prototypcal_col][left_side_flag] = class_dfs[1][s_info.prototypcal_col][left_side_flag]    
      both_class_df[s_info.puffindex_col][left_side_flag] = class_dfs[1][s_info.puffindex_col][left_side_flag]
      try:
        both_class_df[s_info.puffside_col][left_side_flag] = class_dfs[1][s_info.puffside_col][left_side_flag]
      except:
        both_class_df[s_info.puffside_col] = class_dfs[1][s_info.puffside_col]
      # merge class name and target
      class_assign_flag = class_dfs[1][s_info.classnum_col] > both_class_df[s_info.classnum_col]
      both_class_df[s_info.classnum_col][class_assign_flag] = class_dfs[1][s_info.classnum_col][class_assign_flag]
      both_class_df[s_info.classname_col][class_assign_flag] = class_dfs[1][s_info.classname_col][class_assign_flag]
      # reset index
      both_data_df = both_data_df.reset_index(drop=True)
      both_class_df = both_class_df.reset_index(drop=False)
      both_class_df = both_class_df.rename(columns={'index':'single session and sensor index'})
      # serialize
      both_data_pkl_name = s_info.pkl_folder + '/session' + str(session) + '_BFW' + "." + str(window_size) + '.data.pkl'
      both_class_pkl_name = s_info.pkl_folder + '/session' + str(session) + '_BFW' "." + str(window_size) + '.class.pkl'
      both_data_pkl_names.append(both_data_pkl_name)
      both_class_pkl_names.append(both_class_pkl_name)
      joblib.dump(both_data_df, both_data_pkl_name)
      joblib.dump(both_class_df, both_class_pkl_name)
      print "Generated pkls: " + both_data_pkl_name + ", " + both_class_pkl_name
  return both_data_pkl_names, both_class_pkl_names  

def build_whole_dataset_each_sensor_from_pkl(sessions=[], sensors=[], window_size=200):
  """
    load and connect pkl files into a larger data_df and class_df and serialize them

  Args:
    sessions: sessions to be loaded
    sensors: sensors to be used
  Return:
    data_pkl_names, class_pkl_names
  """
  print "===============build all session dataset for each sensor from pkls=========="
  data_pkl_names = {}
  class_pkl_names = {}
  for sensor in sensors:
    data_pkl_names[sensor] = []
    class_pkl_names[sensor] = []
  for sensor in sensors:
    data_dfs = []
    class_dfs = []
    for session in sessions:
      data_pkl_name = s_info.pkl_folder + '/session' + str(session) + '_' + sensor + "." + str(window_size) + '.data.pkl'
      class_pkl_name = s_info.pkl_folder + '/session' + str(session) + '_' + sensor + "." + str(window_size) + '.class.pkl'
      data_dfs.append(joblib.load(data_pkl_name))
      class_dfs.append(joblib.load(class_pkl_name))
    # concat them
    combined_data_df = pd.concat(data_dfs)
    combined_class_df = pd.concat(class_dfs)
    # reset index
    combined_data_df = combined_data_df.reset_index(drop=True)
    combined_class_df = combined_class_df.reset_index(drop=False)
    combined_class_df = combined_class_df.rename(columns={'index':'single session and sensor index'})
    # serialize them
    combined_data_pkl = s_info.pkl_folder + "/session_all_" + sensor + "." + str(window_size) + ".data.pkl"
    combined_class_pkl = s_info.pkl_folder + "/session_all_" + sensor + "." + str(window_size) + ".class.pkl"
    joblib.dump(combined_data_df, combined_data_pkl)
    joblib.dump(combined_class_df, combined_class_pkl)
    data_pkl_names[sensor].append(combined_data_pkl)
    class_pkl_names[sensor].append(combined_class_pkl)
    print "generated pkls: " + combined_data_pkl + ', ' + combined_class_pkl
  return data_pkl_names, class_pkl_names

def build_whole_dataset_each_sensor_from_csv(sessions=[], sensors=[], raw_type="corrected", paras={'window_size':320, 'overlap_rate':0.5}):
  """ It will load raw data and annotation from csv files, compute features and merge different sessions
  them into a large dataset

  Load:
    1. Read in raw and annotation dataset
    2. Do segmentation
    3. Do feature construction and information computation
    4. Do class assignment and pre-exclusion
    5. Serialization

  Connect:
    1. Load

  Args:
    sessions: sessions in array to be used
    sensors: sensor code in array to be used
    raw_type: "raw", "clean" and "corrected"
    paras: hyperparameters: window_size and overlap_rate
  Return:
    data_pkl_names, class_pkl_names
  """
  print "================refresh pkls=============================================================="
  print "sessions: %s, sensors: %s, raw_type: %s, paras: %s" % (sessions, sensors, raw_type, paras)
  if len(sessions) == 0 or len(sensors) == 0:
    return None
  data_pkl_names = {}
  class_pkl_names = {}
  for sensor in sensors:
    data_pkl_names[sensor] = []
    class_pkl_names[sensor] = []
  # load from csv, get data and 
  for session in sessions:
    raw_df, annotation_df = s_loader.load_smoking_df(session=session, sensors=sensors, kind=raw_type)
    # groupby sensors
    grouped_raw = raw_df.groupby(s_info.sensor_col)
    grouped_annotation = annotation_df.groupby(s_info.sensor_col)
    for sensor, group_raw in grouped_raw:
      group_annotation = grouped_annotation.get_group(sensor)
      # do segmentation on raw
      seg_raw_df = s_segment.do_segmentation_on_raw(group_raw, method='window',paras=paras)
      # do preprocessing on raw
      seg_raw_df = s_feature.do_preprocess_on_segment_raw(seg_raw_df)
      # do segmentation on annotation
      seg_annotation_df = s_segment.set_segmentation_on_annotation(group_annotation, seg_raw_df)
      # do info construction on annotation
      feature_annotation_df = s_feature.get_info_from_segment_annotation(seg_annotation_df, s_info.feature_info_dict)
      # do class assignment on annotation
      class_df = s_prepare.assign_class_from_feature_annotation(feature_annotation_df)
      # do pre-exclusion on class dataframe
      class_df = s_prepare.pre_exclude_rest_instances(seg_raw_df, class_df=class_df)
      # do feature construction on segment raw
      feature_func_dict = s_info.feature_dataset_folder + "/feature_dict_" + sensor + '.json'
      feature_raw_df = s_feature.get_features_from_segment_raw(seg_raw_df, feature_func_dict)
      data_df = s_prepare.get_data_df_from_feature_raw_df(feature_raw_df)
      # serialize data and class df
      data_pkl, class_pkl = s_prepare.serialize_data_and_class_df(data_df, class_df)
      data_pkl_names[sensor].append(data_pkl)
      class_pkl_names[sensor].append(class_pkl)
    print ""
  # connect together
  if len(sessions) > 1:
    data_pkl_names, class_pkl_names = build_whole_dataset_each_sensor_from_pkl(sessions, sensors, paras['window_size'])
  return data_pkl_names, class_pkl_names

#===============================================================================

def test_build_whole_dataset_each_sensor_from_csv():
  sessions = [7,]
  session = 7
  sensors = ['DW',]
  raw_type = "corrected"
  window_size = 240
  sensor = 'DW'
  data_pkl_names, class_pkl_names = build_whole_dataset_each_sensor_from_csv(sessions, sensors, raw_type, paras={'window_size':window_size, 'overlap_rate':0.5})
  print "==============test result of build_whole_dataset_each_sensor_from_csv=============="
  print data_pkl_names
  print class_pkl_names
  print "==============dataset export================="
  data_df = joblib.load(s_info.pkl_folder + "/session" + str(session) + "_" + sensor + "." + str(window_size) + ".data.pkl")
  class_df = joblib.load(s_info.pkl_folder + "/session" + str(session) + "_" + sensor + "." + str(window_size) + ".class.pkl")
  data_name = s_info.feature_dataset_folder + "/session" + str(session) + "_" + sensor + "." + str(window_size) + ".data.csv"
  data_df.to_csv(data_name)
  class_name = s_info.feature_dataset_folder + "/session" + str(session) + "_" + sensor + "." + str(window_size) + ".class.csv"
  class_df.to_csv(class_name)
  print data_name + " exported"
  print class_name + " exported"
  # for sensor in sensors:
  #   data_df = joblib.load(data_pkl_names[sensor][0])
  #   class_df = joblib.load(class_pkl_names[sensor][0])
  #   excel_name = s_info.feature_dataset_folder + "/session_all_" + sensor + ".feature.xlsx"
  #   s_prepare.export_data_and_class_df_to_excel(data_df, class_df, excel_name)
  #   print excel_name + " exported"

def test_build_whole_dataset_each_sensor_from_pkl():
  sessions = [1,3,]
  sensors = ['DW', 'NDW']
  data_pkl_names, class_pkl_names = build_whole_dataset_each_sensor_from_pkl(sessions, sensors, window_size=200)
  print "==============test result of build_whole_dataset_each_sensor_from_pkl=============="
  print data_pkl_names
  print class_pkl_names
  print "==============dataset export================="
  # for sensor in sensors:
  #   data_df = joblib.load(data_pkl_names[sensor][0])
  #   class_df = joblib.load(class_pkl_names[sensor][0])
  #   excel_name = s_info.feature_dataset_folder + "/session_all_" + sensor + ".feature.xlsx"
  #   s_prepare.export_data_and_class_df_to_excel(data_df, class_df, excel_name)

def test_combine_both_side_wrist_datasets():
  sessions = 'all'
  data_pkl_names, class_pkl_names = combine_both_side_wrist_dataset(sessions, window_size=200)
  print "==============dataset export================="
  # for session in sessions:
  #   data_df = joblib.load(data_pkl_names[sessions.index(session)])
  #   class_df = joblib.load(class_pkl_names[sessions.index(session)])
  #   excel_name = s_info.feature_dataset_folder + "/session" + str(session) + "_BW.feature.xlsx"
  #   s_prepare.export_data_and_class_df_to_excel(data_df, class_df, excel_name)

def test_merge_both_side_wrist_dataset_as_features():
  sessions = [3,]
  data_pkl_names, class_pkl_names = merge_both_side_wrist_dataset_as_features(sessions, window_size=200)
#===============================================================================

def update_all_datasets(paras={'window_size':200, 'overlap_rate':0.5}):
  build_whole_dataset_each_sensor_from_csv(sessions=[1,3,4,5,6,7], sensors=['DW','NDW'], raw_type='corrected',  paras=paras)
  combine_both_side_wrist_dataset(sessions='all', window_size=paras['window_size'])
  combine_both_side_wrist_dataset(sessions=[1,3,4,5,6,7], window_size=paras['window_size'])
  merge_both_side_wrist_dataset_as_features(sessions='all', window_size=paras['window_size'])
  merge_both_side_wrist_dataset_as_features(sessions=[1,3,4,5,6,7], window_size=paras['window_size'])

def export_all_datasets_to_csv(window_size=200):
  sessions = [1,3,4,5,6,7,'_all']
  for session in sessions:
    sensors = ['DW', 'NDW', 'BW', 'BFW']
    for sensor in sensors:
      data_df = joblib.load(s_info.pkl_folder + "/session" + str(session) + "_" + sensor + "." + str(window_size) + ".data.pkl")
      class_df = joblib.load(s_info.pkl_folder + "/session" + str(session) + "_" + sensor + "." + str(window_size) + ".class.pkl")
      data_name = s_info.feature_dataset_folder + "/session" + str(session) + "_" + sensor + "." + str(window_size) + ".data.csv"
      data_df.to_csv(data_name)
      class_name = s_info.feature_dataset_folder + "/session" + str(session) + "_" + sensor + "." + str(window_size) + ".class.csv"
      class_df.to_csv(class_name)
      print data_name + " exported"
      print class_name + " exported"


if __name__ == "__main__":
  # s_info.test_func(test_build_whole_dataset_each_sensor_from_csv, profile=False)
  # s_info.test_func(test_build_whole_dataset_each_sensor_from_pkl, profile=False)
  # s_info.test_func(test_combine_both_side_wrist_datasets, profile=False)
  # s_info.test_func(test_merge_both_side_wrist_dataset_as_features, profile=False)
  for i in [1400,]:
    paras={'window_size':i, 'overlap_rate':0.5}
    update_all_datasets(paras=paras)
    export_all_datasets_to_csv(window_size=paras['window_size'])
