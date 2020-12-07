"""
  Module to prepare dataset for scikit learning for single session and single sensor

  Include two algorithms:

  1. Class assignment and encoding
  2. Pre-exclusion for "Rest" instances
  3. Serialize data_df and class_df

  The input dataframe should be feature raw and annotation dataframe
  Note that any standardization, scaling or normalization on data frame will be done during training/testing
"""
import sys
sys.path.append("../")
import smdt.info as s_info
import smdt.segment as s_segment
import smdt.loader as s_loader
import smdt.feature as s_feature
import numpy as np

def assign_class_from_feature_annotation(feature_annotation_df):
  """Decide and encode class names, add other necessary information for evaluation

  Args:
    feature_annotation_df: input feature annotation dataframe
  Return:
    class_df: output class dataframe
  """
  print "============Start assigning and encoding classes================="
  print "================thresholds====================="
  walkthreshold = 0.3
  ambithreshold = 0.25
  phonethreshold = 0.3
  computerthreshold = 0.3
  activity_thresholds = [0.3,0.25,0.3,0.3,0.3,0.3,0.3,0.25]
  activity_names = ['walking','eating','phone','computer','talking', 'reading','in car', 'drinking']
  class_codes = [1,2,3,4,5,6,7,8]
  activity_cols = [s_info.walkproportion_col,
                   s_info.eatproportion_col,
                   s_info.phoneproportion_col,
                   s_info.computerproportion_col,
                   s_info.talkproportion_col,
                   s_info.readproportion_col,
                   s_info.carproportion_col,
                   s_info.drinkproportion_col]
  puff_with_puff_duration_threshold = 0.3
  puff_with_segment_duration_threshold = 0.3
  print "walking: " + str(walkthreshold)
  print "drinking/eating: " + str(ambithreshold)
  print "phone: " + str(phonethreshold)
  print "computer: " + str(computerthreshold)
  print "puff: " + str(puff_with_puff_duration_threshold) + ", " + str(puff_with_segment_duration_threshold)
  class_df = feature_annotation_df.copy(deep=True)
  
  # class assignment rules
  class_df[s_info.classname_col] = 'others'
  class_df[s_info.classnum_col] = 0

  for code, name, threshold, col in zip(class_codes, activity_names, activity_thresholds, activity_cols):
    flag = class_df[col] >= threshold
    class_df[s_info.classname_col][flag] = name
    class_df[s_info.classnum_col][flag] = code
  # for short window size
  puff_flag1 = class_df[s_info.puff_with_segment_duration_col] >= puff_with_segment_duration_threshold
  class_df[s_info.classname_col][puff_flag1] = 'puff'
  class_df[s_info.classnum_col][puff_flag1] = len(class_codes) + 1
  # for short puffs
  puff_flag2 = class_df[s_info.puff_with_puff_duration_col] >= puff_with_puff_duration_threshold
  class_df[s_info.classname_col][puff_flag2] = 'puff'
  class_df[s_info.classnum_col][puff_flag2] = len(class_codes) + 1
  return class_df

def get_data_df_from_feature_raw_df(feature_raw_df):
  print "=======get data dataframe from feature raw dataframe==============="
  data_df = feature_raw_df.copy(deep=True)
  return data_df

def pre_exclude_rest_instances(seg_raw_df, class_df):
  """Pre exclude "Rest" instance according to std and slope of the magnitude of a sequence

  It will add a rest marker(classname: rest, target: -1)

  Args:
    seg_raw_df: used to find out the segments to be excluded (should be better to use preprocessed seg raw df)
    class_df: class dataframe for training/testing, from assign_class_from_feature_annotation
  Returns:
    class_df: with rest encoded as -1 for class number
  """
  import smdt.features.features as features

  print "============start pre exclusion=================================="
  

  seg_mag_df = seg_raw_df.copy(deep=True)
  temp = [seg_mag_df[name]**2 for name in s_info.raw_value_names]
  temp = np.sum(temp, axis=0)
  seg_mag_df['mag'] = np.sqrt(temp)

  grouped = seg_mag_df.groupby(s_info.segment_col)

  c1 = grouped['mag'].std()
  c2 = grouped['mag'].aggregate(features.f_slope).abs()
  c3 = grouped['mag'].aggregate(features.f_pppeakamplitude, paras={"q":10})

  # Used for visualization testing
  # import matplotlib.pyplot as pyplot
  # c1.hist()
  # pyplot.figure()
  # c2.hist(bins=100)
  # pyplot.figure()
  # c3.hist()

  # pyplot.show()
  # sys.exit(1)
  t1 = 0.13
  t2 = 0.0004
  t3 = 0.5
  print "===================preexclusion criterions===================="
  print "std: <= %f, slope: <= %f, peak-peak amplitude: < %f" % (t1, t2, t3) 
  excluded = (c1 <= t1) & (c2 <= t2) & (c3 <= t3)
  class_df[s_info.classname_col][excluded] = 'rest'
  class_df[s_info.classnum_col][excluded] = -1

  c_rest = len(class_df[excluded])
  c_keep = len(class_df[~excluded])
  c_total = len(class_df)
  print "Exclusion result: excluded/keep/total: %.1f, %.1f, %.1f    exclusion rate: %.2f" % (c_rest, c_keep, c_total, c_rest/float(c_total))
  return class_df

def serialize_data_and_class_df(data_df, class_df):
  """Use joblib the serialize, cache the data_df and class_df for each session

  The reason of doing so is we don't need to persist one session's data when loading another session, 
  or sometimes it's more convenient to save them external and resume later
  Args:
    data_df: data dataframe
    class_df: class dataframe
  Returns:
    data_pkl_filename, class_pkl_filename
  """

  from sklearn.externals import joblib
  print "==========start serializing data and class dataframe================"
  # construct filename
  session = class_df.ix[0,s_info.session_col]
  sensor = class_df.ix[0, s_info.sensor_col]
  window_size = (class_df[s_info.segduration_col][0] + 1/float(s_info.raw_sample_rate))*s_info.raw_sample_rate
  window_size = str(int(window_size))
  print "session: %d, sensor: %s, window size: %s" % (session, sensor, window_size)
  data_pkl_filename = s_info.pkl_folder + "/session" + str(session) + "_" + sensor + "." + window_size + ".data.pkl"
  class_pkl_filename = s_info.pkl_folder + "/session" + str(session) + "_" + sensor + "." + window_size + ".class.pkl"
  joblib.dump(data_df, data_pkl_filename)
  joblib.dump(class_df, class_pkl_filename)
  print data_pkl_filename + " serialized"
  print class_pkl_filename + " serialized"
  return data_pkl_filename, class_pkl_filename

def export_data_and_class_df_to_excel(data_df, class_df, excel_filename=None):
  """Export data and class df to excel in two different sheets

  The output excel is mainly used for format or visual check
  Args:
    data_df: data dataframe
    class_df: class dataframe
  Returns:
    excel_filename
  """
  from pandas import ExcelWriter
  print "==========start exporting data and class dataframe to excel================"
  if excel_filename == None:
    session = class_df.ix[0,s_info.session_col]
    sensor = class_df.ix[0, s_info.sensor_col]
    print "session: %d, sensor: %s" % (session, sensor)
    excel_filename = s_info.feature_dataset_folder + "/session" + str(session) + "_" + sensor + ".feature.xlsx"
  writer = ExcelWriter(excel_filename)
  data_df.to_excel(writer, sheet_name="data(features)")
  class_df.to_excel(writer, sheet_name="class(other information)")
  writer.save()
  print excel_filename + " exported"
  return excel_filename

#===============================================================================

def test_assign_class_from_feature_annotation():
  sensor = 'DW'
  raw_df, annotation_df = s_loader.load_smoking_df(session=1, sensors=[sensor,], kind='corrected')
  seg_raw_df = s_segment.do_segmentation_on_raw(raw_df, method='window',paras={'window_size':40, 'overlap_rate':0.5})
  seg_annotation_df = s_segment.set_segmentation_on_annotation(annotation_df, seg_raw_df)
  feature_annotation_df = s_feature.get_info_from_segment_annotation(seg_annotation_df, s_info.feature_info_dict)
  class_df = assign_class_from_feature_annotation(feature_annotation_df)
  print "============Test result of assigning class================="
  print class_df
  print "============class dataframe head================="
  print class_df.head().T
  print "============class dataframe tail================="
  print class_df.tail().T

def test_get_data_df_from_feature_raw():
  sensor = 'DW'
  raw_df, annotation_df = s_loader.load_smoking_df(session=6, sensors=[sensor,], kind='corrected')
  seg_raw_df = s_segment.do_segmentation_on_raw(raw_df, method='window',paras={'window_size':320, 'overlap_rate':0.5})
  feature_func_dict = s_info.feature_dataset_folder + "/feature_dict_" + sensor + '.json'
  feature_raw_df = s_feature.get_features_from_segment_raw(seg_raw_df, feature_func_dict)
  data_df = get_data_df_from_feature_raw_df(feature_raw_df)
  print "============Test result of getting data dataframe================="
  print data_df
  print "============class dataframe head================="
  print data_df.head().T
  print "============class dataframe tail================="
  print data_df.tail().T

def test_pre_exclude_rest_instances():
  sensor = 'DW'
  raw_df, annotation_df = s_loader.load_smoking_df(session=6, sensors=[sensor,], kind='corrected')
  seg_raw_df = s_segment.do_segmentation_on_raw(raw_df, method='window',paras={'window_size':320, 'overlap_rate':0.5})
  seg_raw_df = s_feature.do_preprocess_on_segment_raw(seg_raw_df)
  seg_annotation_df = s_segment.set_segmentation_on_annotation(annotation_df, seg_raw_df)
  feature_annotation_df = s_feature.get_info_from_segment_annotation(seg_annotation_df, s_info.feature_info_dict)
  class_df = assign_class_from_feature_annotation(feature_annotation_df)

  # test
  class_df = pre_exclude_rest_instances(seg_raw_df, class_df=class_df)
  print "============Test result of pre-exclusion================="
  print "============class dataframe comparation================"
  print class_df
  print "=========rest segments==============="
  print class_df[class_df[s_info.classnum_col] == -1]
  print "=========not rest segments==========="
  print class_df[class_df[s_info.classnum_col] != -1]
  # selected_segment = class_df_rest[class_df_rest[s_info.puffindex_col] > 0].index
  # for s in selected_segment:
  #   s = seg_raw_df.groupby(s_info.segment_col).get_group(s)
  #   s.plot(ylim=(-3,3))
  # import matplotlib.pyplot as pyplot
  # pyplot.show()

def test_serialize_data_and_class_df():
  sensor = 'DW'
  raw_df, annotation_df = s_loader.load_smoking_df(session=6, sensors=[sensor,], kind='corrected')
  seg_raw_df = s_segment.do_segmentation_on_raw(raw_df, method='window',paras={'window_size':200, 'overlap_rate':0.5})
  seg_raw_df = s_feature.do_preprocess_on_segment_raw(seg_raw_df)
  seg_annotation_df = s_segment.set_segmentation_on_annotation(annotation_df, seg_raw_df)
  feature_annotation_df = s_feature.get_info_from_segment_annotation(seg_annotation_df, s_info.feature_info_dict)
  class_df = assign_class_from_feature_annotation(feature_annotation_df)
  feature_func_dict = s_info.feature_dataset_folder + "/feature_dict_" + sensor + '.json'
  feature_raw_df = s_feature.get_features_from_segment_raw(seg_raw_df, feature_func_dict)
  data_df = get_data_df_from_feature_raw_df(feature_raw_df)
  data_pkl_name, class_pkl_name = serialize_data_and_class_df(data_df, class_df)

def test_export_data_and_class_df_to_excel():
  sensor = 'DW'
  raw_df, annotation_df = s_loader.load_smoking_df(session=6, sensors=[sensor,], kind='corrected')
  seg_raw_df = s_segment.do_segmentation_on_raw(raw_df, method='window',paras={'window_size':320, 'overlap_rate':0.5})
  seg_raw_df = s_feature.do_preprocess_on_segment_raw(seg_raw_df)
  seg_annotation_df = s_segment.set_segmentation_on_annotation(annotation_df, seg_raw_df)
  feature_annotation_df = s_feature.get_info_from_segment_annotation(seg_annotation_df, s_info.feature_info_dict)
  class_df = assign_class_from_feature_annotation(feature_annotation_df)
  feature_func_dict = s_info.feature_dataset_folder + "/feature_dict_" + sensor + '.json'
  feature_raw_df = s_feature.get_features_from_segment_raw(seg_raw_df, feature_func_dict)
  data_df = get_data_df_from_feature_raw_df(feature_raw_df)
  excel_filename = export_data_and_class_df_to_excel(data_df, class_df)

if __name__ == "__main__":
  s_info.test_func(test_assign_class_from_feature_annotation, profile=False)
  # s_info.test_func(test_get_data_df_from_feature_raw, profile=True)
  # s_info.test_func(test_pre_exclude_rest_instances, profile=False)
  # s_info.test_func(test_serialize_data_and_class_df, profile=True)
  # s_info.test_func(test_export_data_and_class_df_to_excel, profile=True)