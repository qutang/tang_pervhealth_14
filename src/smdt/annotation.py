#! /usr/bin/python

"""
  Module for reading and processing smdt annotation files
"""


import sys
sys.path.append('../')
import random as random
from datetime import timedelta
import numpy as np
import pandas as pd
import smdt.info as s_info


header_index = 0 # annotation csv header row index
use_cols = ['STARTTIME','ENDTIME','posture','activity','smoking','puffing']
st_col = 'STARTTIME'
et_col = 'ENDTIME'
puff_col = 'puffing'
activity_col = 'activity'
post_col = 'posture'
smoke_col = 'smoking'
category_names = ['posture','activity','smoking','puffing']
label_dict = {'posture':['sitting','standing','walking','lying/lounging','unknown-posture'],
                          'activity':['eating-a-meal','drinking-beverage',
                                      'using-computer','using-phone',
                                      'reading-paper','in-car',
                                      'talking','walking-down-stairs',
                                      'unknown-activity'],
                          'smoking':['smoking','not-smoking'],
                          'puffing':['left-puff','right-puff','no-puff']}
tstr_format = '%Y-%m-%d %H:%M:%S'

def get_categories_from_labels(labels):
  """Get label category provided labels
  """
  cats = []
  for cat in label_dict:
    for label in labels: 
      if label in label_dict[cat]:
        cats.append(cat)
  return cats

def generate_random_bounds(csv_data, duration=timedelta(seconds=30)):
  """Generate upper and lower bounds from annotation dataset

  Generate lower bound by randomly select one row in annotation dataset and add
  duration to get upper bound

  Args:
    csv_data: annotation dataset
    duration: timedelta of duration of bounds, default is 30s

  Returns:
    (lower_bound, upper_bound): tuple of lower and upper bound
  """
  time1 = random.choice(csv_data[st_col])
  time2 = time1 + duration
  return (time1, time2)

def generate_bounds_by_labels(csv_data, duration=timedelta(seconds=30), labels=['sitting',], seed=None):
  """Generate random bounds that given labels are within the bounds

  Filter annotation dataset according to the provided labels and randomly choose
  one row from the filtered annotation set. Use the middle time point of this row
  as the center time point of generated duration. The duration length is an input argument.

  Args:
    csv_data: annotation dataset to be used for generating bounds
    duration: timedelta, specify the generated bounds duration, default is 30s
    labels: list of labels that we want the generated bounds to involve, default is 'sitting'
    seed: seed for random generator

  Returns:
    (lower_bound, upper_bound): generated lower and upper bound
  """

  use_cats = get_categories_from_labels(labels)
  action_data = csv_data.filter(use_cats)
  bool_data = action_data.applymap(lambda x: labels[0] == x)
  for label in labels:
    bool_data = np.logical_or(bool_data,action_data.applymap(lambda x: label == x))
  bool_index = bool_data.apply(lambda x: np.all(x), axis=1)
  selected_index = bool_index[bool_index == True]
  random.seed(seed)
  selected_row = random.choice(selected_index.index) # randomly choose one row that contains these labels
  selected_st = csv_data.ix[selected_row, st_col]
  selected_et = csv_data.ix[selected_row, et_col]
  selected_md = selected_st + (selected_et - selected_st)/2 # use the middle time point of this row as the center of generated bounds
  lbound = selected_md - duration/2
  lbound.replace(microsecond=0)
  rbound = selected_md + duration/2
  rbound.replace(microsecond=0)
  return lbound,rbound

def select_annotation_by_ts(csv_data, lbound=None, rbound=None, by=None):
  """Select subset of annotation dataset by provided bounds

  Select subset of annotation dataset by provided lower and upper bounds, this function
  support group by a column first then select data

  Args:
    csv_data: annotation dataset to be used
    lbound: lower(left) bound to be used
    rbound: upper(right) bound to be used
    by: specify the column to be used to group dataset

  Return:
    selected subset of annotation dataset or None if the subset is empty
  """
  if by==None:
    if not lbound:
      lbound = csv_data[st_col].iloc[0] # iloc is faster than head() or tail()
    if not rbound:
      rbound = csv_data[et_col].iloc[-1]
    # start_flags = np.array(csv_data[et_col].apply(lambda x: x>lbound)) ## Note it's too slow
    flags = (csv_data[et_col] > lbound) & (csv_data[st_col] < rbound)
    # end_flags = np.array(csv_data[st_col].apply(lambda x:x<rbound)) ## Note it's too slow
    subset_annotation_data = csv_data[flags]
    # subset_annotation_data = subset_annotation_data.reset_index(drop=True) ## Don't reset index
    subset_annotation_data[st_col].iloc[0] = max(lbound,subset_annotation_data[st_col].iloc[0])
    subset_annotation_data[et_col].iloc[-1] = min(rbound,subset_annotation_data[et_col].iloc[-1])
  else:
    groupby_annotation = csv_data.groupby(by)
    subset_group_datas = []
    for group_name, group_data in groupby_annotation:
      if lbound == None:
        lbound = group_data[st_col].iloc[0]
      if rbound == None:
        rbound = group_data[et_col].iloc[-1]
      # start_flags = np.array(group_data[et_col].apply(lambda x: x>lbound)) ## Note it's too slow
      start_flags = group_data[et_col] > lbound
      # end_flags = np.array(group_data[st_col].apply(lambda x:x<rbound)) ## Note it's too slow
      end_flags = group_data[st_col] < rbound
      subset_group_data = group_data[np.logical_and(start_flags,end_flags)]
      subset_group_data[st_col].iloc[0] = max(lbound,subset_group_data[st_col].iloc[0])
      subset_group_data[et_col].iloc[-1] = min(rbound,subset_group_data[et_col].iloc[-1])
      # subset_group_data = subset_group_data.reset_index(drop=True) ## Don't reset index
      subset_group_datas.append(subset_group_data)
    subset_annotation_data = annotation_data_consolidator(subset_group_datas)
  return subset_annotation_data

def select_annotation_by_random(csv_data, duration=timedelta(seconds=30), by=None):
  lbound, rbound = generate_random_bounds(csv_data, duration=duration)
  random_annotation_data = select_annotation_by_ts(csv_data, lbound=lbound, rbound=rbound, by=by)
  return random_annotation_data

def select_annotation_by_labels(csv_data, duration=timedelta(seconds=30), labels=['sitting',], by=None):
  lbound, rbound = generate_bounds_by_labels(csv_data, duration=duration, labels=labels)
  subset_annotation_data = select_annotation_by_ts(csv_data, lbound, rbound, by)
  return subset_annotation_data

def annotation_csv_importer(filename):

  # don't use annotation_converters, slow and not necessary
  # annotation_converters = {}
  # annotation_converters[st_col] = lambda s: w_utils.convert_fromstring(s, tstr_format)
  # annotation_converters[et_col] = lambda s: w_utils.convert_fromstring(s, tstr_format)
  
  # import options:
  # header: used to specify header row index
  # usecols: used to specify column indexes to be imported
  # converters: a dict used to specify the format converter used for specific columns
  # dtype: a dict to specify the data type for each column
  csv_data = pd.read_csv(filename, header=header_index, parse_dates=[st_col, et_col], verbose=False)

  # for test
  # print csv_data.tail(5)
  # print csv_data.STARTTIME[0]
  # print csv_data.describe()
  return csv_data

def fix_annotation(csv_data, time_offset = 0):
  """ Do some fixing to the annotation dataset

  Fix the annotation dataset by 
    1. filter out invalid rows which have the same start and end time or have NaN values in start or end time
    2. fill in blank puffing cells with "no-puff", fill in blank activity cells with "no-activity", fill in blank posture cells with backfill
    3. use backfill for 'no-activity' cells whose length is less than 3s
    4. change isolated "smoking" cells into "not-smoking"
    Then apply time offset to all the timestamps to realign annotation dataset. 
    Filtered dataset will be reindexed

  Args:
    csv_data: annotation dataset to be fixed
    time_offset: time_offset in seconds to be added to all the timestamps, default is 0

  Returns:
    Fixed annotation dataset
  """
  # step 1: eliminate rows with same starttime and endtime
  csv_data = csv_data[csv_data.STARTTIME != csv_data.ENDTIME]

  # step 2: elminate nan in starttime and endtime
  csv_data = csv_data.dropna(axis=0,subset=[st_col,et_col])

  # step 3: fill "blank" cells
  csv_data = csv_data.reset_index(drop=True)
  csv_data[puff_col] = csv_data[puff_col].fillna(value='no-puff')
  csv_data[activity_col] = csv_data[activity_col].fillna(value='no-activity')
  csv_data[post_col] = csv_data[post_col].fillna(method='backfill')
  csv_data[post_col] = csv_data[post_col].fillna(method='ffill')
  csv_data[smoke_col] = csv_data[smoke_col].fillna(value='not-smoking')
  
  # step 4: fill 'no-activity' cells whose length is less than 3s with backfill
  csv_data = csv_data.reset_index(drop=True)
  filt = csv_data.apply(lambda x: x[et_col] - x[st_col] <= timedelta(seconds=2) and x[activity_col] == 'no-activity', axis=1)
  csv_data.ix[csv_data[filt].index, activity_col] = csv_data.ix[csv_data[filt].index+1, activity_col].values
  csv_data[activity_col] = csv_data[activity_col].fillna(value='no-activity')
  # step 5: change isolated single "smoking" cells into proper label
  bshift_smoke = csv_data[smoke_col].shift(1).fillna(method='backfill')
  fshift_smoke = csv_data[smoke_col].shift(-1).fillna(method='ffill')
  filt = np.logical_and(csv_data[smoke_col] != bshift_smoke, csv_data[smoke_col] != fshift_smoke)
  # print csv_data[filt]
  # ind = csv_data[filt].index
  filt1 = np.logical_and(filt, csv_data[smoke_col] == 'smoking')
  csv_data.ix[filt1, smoke_col] = 'not-smoking'
  filt = np.logical_and(csv_data[smoke_col] != bshift_smoke, csv_data[smoke_col] != fshift_smoke)
  filt2 = np.logical_and(np.logical_and(filt, csv_data[smoke_col] == 'not-smoking'), csv_data.apply(lambda x: x[et_col] - x[st_col] < timedelta(minutes=1),axis=1))
  csv_data.ix[filt2, smoke_col] = 'smoking'
  # print csv_data.iloc[ind]

  # step 6: turn smoking sequence without puffs into "not smoking"
  st_filt = np.logical_and(csv_data[smoke_col] != csv_data[smoke_col].shift(1), csv_data[smoke_col] == 'smoking')
  et_filt = np.logical_and(csv_data[smoke_col] != csv_data[smoke_col].shift(-1), csv_data[smoke_col] == 'smoking')
  cig_st = csv_data[st_filt]
  cig_et = csv_data[et_filt]
  for i in range(0,len(cig_st.index)):
    puff_flag = csv_data[cig_st.index[i]:cig_et.index[i]+1][puff_col] == 'no-puff'
    if puff_flag.all():
      csv_data[cig_st.index[i]:cig_et.index[i]+1][smoke_col] = 'not-smoking'

  # step 7: add offset to starttime and endtime
  # print csv_data.head()
  csv_data[et_col] = csv_data[et_col] + timedelta(seconds=time_offset)
  csv_data[st_col] = csv_data[st_col] + timedelta(seconds=time_offset)
  # print csv_data.head()

  # step 8: reindex from 0
  csv_data = csv_data.reset_index(drop=True)
  return csv_data

def correct_puffs_and_add_prototypical_marks(csv_data, correction_list=None):
  # add extra columns
  csv_data['puff index'] = -1
  csv_data['prototypical?'] = -1
  csv_data['potential error?'] = 0
  csv_data['note'] = ''
  csv_data['link'] = ''
  csv_data[s_info.puffduration_col] = 0
  if correction_list is None:
    return csv_data
  else:
    for idx, (st, et, link, loffset, roffset, prototypical, error, note) in correction_list.iterrows():
      puff_flag = np.logical_and(csv_data[et_col] > st, csv_data[st_col] < et)
      print csv_data[puff_flag]
      # apply puff correction
      corrected_st = st + timedelta(seconds=loffset-1.5) # systematically shift the beginning to the left 1.5s
      corrected_et = et + timedelta(seconds=roffset)
      # clear puff first
      side = csv_data[puff_col][puff_flag].iloc[0]
      csv_data[puff_col][puff_flag] = 'no-puff'
      print csv_data[puff_flag]
      # find out the new puff range
      new_puff_flag = np.logical_and(csv_data[et_col] > corrected_st, csv_data[st_col] < corrected_et)
      # split the first and last row if necessary
      print corrected_st, corrected_et
      new_st_idx = csv_data[new_puff_flag].index[0]
      print csv_data[new_puff_flag]
      if csv_data.ix[new_st_idx,st_col] != corrected_st:
        left_split = csv_data.ix[:new_st_idx]
        right_split = csv_data.ix[new_st_idx+1:]
        insert_row = csv_data.ix[new_st_idx].copy()
        left_split[et_col].iloc[-1] = corrected_st
        insert_row[st_col] = corrected_st
        csv_data = pd.concat([left_split, pd.DataFrame(insert_row).T, right_split]).reset_index(drop=True)
      new_puff_flag = np.logical_and(csv_data[et_col] >= corrected_st, csv_data[st_col] < corrected_et)
      new_et_idx = csv_data[new_puff_flag].index[-1]
      if csv_data.ix[new_et_idx,et_col] != corrected_et:
        left_split = csv_data.ix[:new_et_idx-1]
        right_split = csv_data.ix[new_et_idx:]
        insert_row = csv_data.ix[new_et_idx].copy()
        right_split[st_col].iloc[0] = corrected_et
        insert_row[et_col] = corrected_et
        csv_data = pd.concat([left_split, pd.DataFrame(insert_row).T, right_split]).reset_index(drop=True)
      correct_puff_flag = np.logical_and(csv_data[st_col] >= corrected_st, csv_data[et_col] <= corrected_et)
      # assign extra columns
      csv_data['puff index'][correct_puff_flag] = idx
      csv_data['prototypical?'][correct_puff_flag] = prototypical
      csv_data['potential error?'][correct_puff_flag] = error
      csv_data['note'][correct_puff_flag] = note
      csv_data['link'][correct_puff_flag] = link
      csv_data[puff_col][correct_puff_flag] = side
      # compute duration and assign
      duration = csv_data[et_col][correct_puff_flag] - csv_data[st_col][correct_puff_flag]
      duration = duration.apply(lambda x: x / np.timedelta64(1,'s'))
      duration = duration.sum()
      csv_data[s_info.puffduration_col][correct_puff_flag] = duration
      print csv_data.ix[correct_puff_flag]
      # raw_input('pause')
  return csv_data

def annotation_data_consolidator(annotat_datas,  sessions=[], subjects=[], sensors=[]):
  single_sess_annotats = []
  c = 0
  for single_sess_annotat in annotat_datas:
    if np.iterable(sessions) and len(sessions) == len(annotat_datas):
      single_sess_index = [sessions[c],]*single_sess_annotat.shape[0]
      single_sess_annotat['session'] = single_sess_index
    if np.iterable(subjects) and len(subjects) == len(annotat_datas):
      single_subj_index = [subjects[c],]*single_sess_annotat.shape[0]
      single_sess_annotat['subject'] = single_subj_index
    if np.iterable(sensors) and len(sensors) == len(annotat_datas):
      single_sensor_index = [sensors[c],]*single_sess_annotat.shape[0]
      single_sess_annotat['sensor'] = single_sensor_index
    single_sess_annotats.append(single_sess_annotat)
    c += 1
  consolidate_annotation_data = pd.concat(single_sess_annotats)
  # consolidate_annotation_data = consolidate_annotation_data.reset_index(drop=False)
  # consolidate_annotation_data = consolidate_annotation_data.rename(columns={"index":"index per sensor"})
  return consolidate_annotation_data

def annotation_csv_consolidator(filenames, sessions=[], subjects=[], sensors=[]):
  """combine several annotation files into one big annotation dataset

  Read in several annotation files and combine them into one big annotation dataset,
  session number, subject id and sensor id can be specified for each file and willb
  be added in the big dataset

  Args:
    filenames: list of annotation csv filpath strings to be read
    time_offsets: list of time offset for each annotation file
    sessions: list of session id for each annotation file
    subjects: list of subject id for each annotation file
    sensors: list of sensor id for each annotation file

  Return:
    Consolidated annotation dataset
  """
  
  single_sess_annotats = []
  c = 0
  for filename in filenames:
    single_sess_annotat = annotation_csv_importer(filename)
    if np.iterable(sessions) and len(sessions) == len(filenames):
      single_sess_index = [sessions[c],]*single_sess_annotat.shape[0]
      single_sess_annotat['session'] = single_sess_index
    if np.iterable(subjects) and len(subjects) == len(filenames):
      single_subj_index = [subjects[c],]*single_sess_annotat.shape[0]
      single_sess_annotat['subject'] = single_subj_index
    if np.iterable(sensors) and len(sensors) == len(filenames):
      single_sensor_index = [sensors[c],]*single_sess_annotat.shape[0]
      single_sess_annotat['sensor'] = single_sensor_index
    c += 1
    single_sess_annotats.append(single_sess_annotat)
  consolidate_annotation_data = pd.concat(single_sess_annotats)
    # consolidate_annotation_data = consolidate_annotation_data.reset_index(drop=False)
    # consolidate_annotation_data = consolidate_annotation_data.rename(columns={"index":"index per sensor"})
  return consolidate_annotation_data

def unit_tests(testfile):
  return

if __name__ == "__main__":
  import cProfile
  pr = cProfile.Profile()
  pr.enable()
  testfile = "../../test.annotation.csv"
  unit_tests(testfile)
  pr.disable()
  pr.create_stats()
  pr.print_stats(sort=2)