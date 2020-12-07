#! ../venv/bin/python
"""
  Module to compute feature set of the smoking segment dataframe (wockets raw dataframe and smoking annotation dataframe)

  It will first group the segment dataframe by segment index, and compute features upon them
  It will then compute necessary information from segment annotation dataframe and add them to the feature set(dataframe)
"""

import sys
import pandas as pd
import numpy as np
import scipy.stats as sp_stats
import scipy.signal as sp_signal
from scipy.ndimage.filters import gaussian_filter1d
sys.path.append("../")
import smdt.loader as s_loader
import smdt.info as s_info
import smdt.features.features as s_feature
import smdt.segment as s_segment

def do_preprocess_on_segment_raw(seg_raw_df):
  """Do general filtering and normalization before get features from segment raw dataframe

  For smoking data, the necessary general proprocessing is:
    1. Median filtering to eliminate spikes
    2. Z-score(standardization) to transform sequence into same scale
    3. Gaussian smoothing in order to eliminate random noises

  Args:
    seg_raw_df: input segment raw dataframe
  Return:
    result: preprocessed segment raw dataframe

  Notes:
    1. The larger sigma is, the smoothier the sequence would be, the cost is lost of some details
    2. The mean and std used in step 2 is from the whole dataset instead of current segment
  """
  sigma = 2
  median_kernel_size = 5
  print "=======================start preprocessing segment raw dataframe================="
  print "parameters: " + "gaussian filter sigma: %.2f, median kernel size: %.2f" % (sigma, median_kernel_size)
  pp_df = seg_raw_df.copy(deep=True)
  df_mean = pp_df[s_info.raw_value_names].mean()
  df_std = pp_df[s_info.raw_value_names].std()
  pp_df[s_info.raw_value_names] = pp_df.groupby(s_info.segment_col)[s_info.raw_value_names].transform(sp_signal.medfilt, median_kernel_size)
  pp_df[s_info.raw_value_names] = (pp_df[s_info.raw_value_names] - df_mean)/df_std
  pp_df[s_info.raw_value_names] = pp_df.groupby(s_info.segment_col)[s_info.raw_value_names].transform(gaussian_filter1d, sigma=sigma, axis=0, order=0, mode='reflect')
  return pp_df

def get_features_from_segment_raw(seg_raw_df, feature_func_dict):
  """Construct feature raw dataframe from the segment raw dataframe

  It will intensively use groupby and aggregate method to compute features, currently
  it only support multiple sensors but single session

  Args:
    seg_raw_df: input segment raw dataframe with segment index column, it can contain
    multiple sensors
    feature_func_dict: feature computation function handlers dict conrresponding to each feature, json format
    It can also be a json external file, there will be 4 json dicts for each sensor
                        Example:
                          {
                            "mean": {
                                     "hander": {"mean":"mean"}, 
                                     "apply": ["rawx", "rawy", "rawz"],
                                     "paras": {}
                                     }
                            },
                            "peakrate": {
                                         "handler": {"peakrate":"f_peakrate"},
                                         "apply": ['rawx', 'rawy', 'rawz'],
                                         "paras": {"sigma":5}
                            },
                          }
  Returns:
    feature_raw_df: concanated feature raw dataframe, index is equal to segment index
  """
  # parse input
  if type(feature_func_dict) == str: # it's a json filename
    import json
    feature_func_str = open(feature_func_dict).read()
    feature_func_dict = json.loads(feature_func_str)
  print "===========start computing features================="
  print "===========feature function dictionary=============="
  print feature_func_dict
  grouped = seg_raw_df.groupby(s_info.segment_col)
  # parse feature function dictionary
  result = {}
  for feature_name in feature_func_dict:
    print "==========compute " + feature_name + "================"
    feature = feature_func_dict[feature_name]
    if len(feature['paras']) == 0: # no parameter need to be set, easiest case
      # find out the function
      func_name = feature['handler']
      if hasattr(np, func_name):
        func = getattr(np, func_name)
      elif hasattr(sp_stats, func_name):
        func = getattr(sp_stats, func_name)
      elif hasattr(s_feature, func_name):
        func = getattr(s_feature, func_name)
      else:
        func = func_name
      # prepare columns
      temp = grouped[feature['apply']].aggregate(func)
      result[feature_name] = temp
    else: # has parameters, will compute column one by one
      paras = feature['paras']
      print paras
      # find out the function
      func_name = feature['handler']
      if hasattr(s_feature, func_name):
        func = getattr(s_feature, func_name)
      elif hasattr(np, func_name):
        func = getattr(np, func_name)
      else:
        print func_name + " can't be found, ignore this feature"
        continue
      # iterate over columns
      temp = {}
      c = 0
      for col in feature['apply']:
        if paras.has_key('with'): # need another column
          paras['another'] = grouped[paras['with'][c]].copy(True)
        temp[col] = grouped[col].aggregate(func, paras)
        c += 1
      # construct DataFrame
      result[feature_name] = pd.DataFrame(temp)
    print "Inf values: %s" % np.any(np.isinf(result[feature_name]))
    print "NaN values: %s" % np.any(np.isnan(result[feature_name]))
  feature_raw_df = pd.concat(result, axis=1)
  # feature_raw_df = feature_raw_df.reset_index(drop=True)
  return feature_raw_df

def get_info_from_segment_annotation(seg_annotation_df, feature_info_dict):
  seg_annotation_df = seg_annotation_df.copy(True)
  """construct feature anntoation dataframe from the segment annotation dataframe

  It will intensively use groupby and aggregate method to compute useful information
  including start and end time, duration, proportion of puffs and so on, currently
  it only support single sensor and single session

  Args:
    seg_annotation_df: segment annotation df
  """
  columns_order = []

  print "========================start retrieving info from annotation segment df================="
  grouped = seg_annotation_df.groupby(s_info.segment_col)
  result = {}
  print "================get start time==========================="
  columns_order.append(s_info.st_col)
  result[s_info.st_col] = grouped[s_info.st_col].min()
  print "================get end time============================="
  columns_order.append(s_info.et_col)
  result[s_info.et_col] = grouped[s_info.et_col].max()
  print "================get segment duration====================="
  columns_order.append(s_info.segduration_col)
  result[s_info.segduration_col] = (result[s_info.et_col] - result[s_info.st_col]).div(np.timedelta64(1,'s'))
  # session and sensors
  print "================get session number====================="
  columns_order.append(s_info.session_col)
  result[s_info.session_col] = grouped[s_info.session_col].first()
  print "================get sensor code====================="
  columns_order.append(s_info.sensor_col)
  result[s_info.sensor_col] = grouped[s_info.sensor_col].first()

  # about puff
  sensor = seg_annotation_df[s_info.sensor_col][0]
  if sensor == 'DW':
    puff_st = seg_annotation_df[s_info.puff_col] == 'right-puff'
  elif sensor == 'NDW':
    puff_st = seg_annotation_df[s_info.puff_col] == 'left-puff'
  else:
    print "This is not a wrist sensor, puff related annotation info will be skipped"
    puff_st = []
  puff_df = seg_annotation_df[puff_st]
  grouped_puff_df = puff_df.groupby(s_info.segment_col)
  puff_duration_df = grouped_puff_df[s_info.puffduration_col].first()
  seg_annotation_df['temp'] = seg_annotation_df[s_info.et_col] - seg_annotation_df[s_info.st_col]
  puff_inside_df = seg_annotation_df.ix[puff_st].groupby(s_info.segment_col)['temp'].sum()
  if len(puff_inside_df) == 0: # no puff found
    puff_inside_df = pd.Series()
  else:
    puff_inside_df = puff_inside_df.div(np.timedelta64(1,'s'))
  print "================get inside puff / puff duration====================="
  columns_order.append(s_info.puff_with_puff_duration_col)
  result[s_info.puff_with_puff_duration_col] = puff_inside_df / puff_duration_df
  print "================get inside puff / segment duration====================="
  columns_order.append(s_info.puff_with_segment_duration_col)
  result[s_info.puff_with_segment_duration_col] = puff_inside_df / result[s_info.segduration_col]

  print "================get prototypical flag of puffs====================="
  columns_order.append(s_info.prototypcal_col)
  result[s_info.prototypcal_col] = grouped_puff_df[s_info.prototypcal_col].first()
  print "================get puff index====================="
  columns_order.append(s_info.puffindex_col)
  result[s_info.puffindex_col] = grouped_puff_df[s_info.puffindex_col].first()
  print "================get puff side======================"
  columns_order.append(s_info.puffside_col)
  result[s_info.puffside_col] = grouped_puff_df[s_info.puff_col].last()

  # single activity proportions
  for info_name in feature_info_dict:
    info_entry = feature_info_dict[info_name]
    print "===================get " + info_entry[1] + " proportion================"
    columns_order.append(info_name)
    info_st = seg_annotation_df[info_entry[0]] == info_entry[1]
    info_inside_df = seg_annotation_df[info_st].groupby(s_info.segment_col)['temp'].sum()
    if len(info_inside_df) == 0: # no this kind of activity found
      result[info_name] = pd.Series()
    else:
      info_inside_df = info_inside_df.div(np.timedelta64(1,'s'))
      result[info_name] = info_inside_df / result[s_info.segduration_col]


  # superposition proportions
  print "===================get superposition proportion================"
  info_st = (seg_annotation_df[s_info.smoke_col] == 'smoking') & ((seg_annotation_df[s_info.activity_col] != 'no-activity') | (seg_annotation_df[s_info.post_col] == 'walking'))
  info_inside_df = seg_annotation_df[info_st].groupby(s_info.segment_col)['temp'].sum()
  info_inside_df = info_inside_df.div(np.timedelta64(1,'s'))
  columns_order.append(s_info.spproportion_col)
  result[s_info.spproportion_col] = info_inside_df / result[s_info.segduration_col]

  # concat result
  print "================concate information====================="
  result = pd.concat(result, axis=1)
  result = pd.DataFrame(result, columns=columns_order)
  return result

#===============================================================================

def test_get_features_from_segment_raw():
  sensor = 'DW'
  raw_df, annotation_df = s_loader.load_smoking_df(session=1, sensors=[sensor,], kind='corrected')
  seg_raw_df = s_segment.do_segmentation_on_raw(raw_df, method='window',paras={'window_size':320, 'overlap_rate':0.5})
  feature_func_dict = s_info.feature_dataset_folder + "/feature_dict_" + sensor + '.json'
  feature_raw_df = get_features_from_segment_raw(seg_raw_df, feature_func_dict)
  print "===================test feature construction result=================="
  print seg_raw_df
  print feature_raw_df
  print "===================feature dataframe head==========================="
  print feature_raw_df.head().T
  print "===================feature dataframe tail==========================="
  print feature_raw_df.tail().T

def test_get_info_from_segment_annotation():
  sensor = 'DW'
  raw_df, annotation_df = s_loader.load_smoking_df(session=1, sensors=[sensor,], kind='corrected')
  seg_raw_df = s_segment.do_segmentation_on_raw(raw_df, method='window',paras={'window_size':320, 'overlap_rate':0.5})
  seg_annotation_df = s_segment.set_segmentation_on_annotation(annotation_df, seg_raw_df)
  feature_annotation_df = get_info_from_segment_annotation(seg_annotation_df, s_info.feature_info_dict)
  print "===================test feature construction result=================="
  print feature_annotation_df
  print "===================feature dataframe head==========================="
  print feature_annotation_df.head().T
  print "===================feature dataframe tail==========================="
  print feature_annotation_df.tail().T

def test_do_preprocess_on_segment_raw():
  sensor = 'DW'
  raw_df, annotation_df = s_loader.load_smoking_df(session=1, sensors=[sensor,], kind='corrected')
  seg_raw_df = s_segment.do_segmentation_on_raw(raw_df, method='window',paras={'window_size':320, 'overlap_rate':0.5})
  pp_seg_raw_df = do_preprocess_on_segment_raw(seg_raw_df)
  print "===============test results of preprocessing on segment raw=================="
  print seg_raw_df.head()
  print pp_seg_raw_df.head()

if __name__ == "__main__":
  s_info.test_func(test_get_features_from_segment_raw, profile=True)
  # s_info.test_func(test_get_info_from_segment_annotation, profile=True)
  # s_info.test_func(test_do_preprocess_on_segment_raw, profile=True)