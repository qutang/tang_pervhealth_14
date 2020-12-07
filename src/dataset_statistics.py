#! /usr/bin/python

"""
  Script to get dataset statistics:
    1. left/right/total/prototypical puffs for each cigarette and each subject
    2. puff durations for each cigarette and each subject
    3. interpuff intervals for each cigarette and each subject
    4. smoking duration for each cigarette and each subject
    5. hand swaps for each cigarette and each subject
"""

import numpy as np
import pandas as pd
import smdt.annotation as s_annotation
import smdt.info as s_info
import sys

def get_cigarette_bounds(alist):
  # split into cigarettes
  smoke_st = np.logical_and(alist[s_annotation.smoke_col] != alist[s_annotation.smoke_col].shift(1), alist[s_annotation.smoke_col] == 'smoking')
  smoke_et = np.logical_and(alist[s_annotation.smoke_col] != alist[s_annotation.smoke_col].shift(-1), alist[s_annotation.smoke_col] == 'smoking')
  cig_sts = alist[smoke_st]
  cig_ets = alist[smoke_et]
  return cig_sts, cig_ets

def get_num_of_puffs_statistics(alist, compute='num-of-puffs'):
  """ For one subject
  """
  results = {'left-puffs':[],'right-puffs':[], 'num-of-puffs':[], 
              'hand-swap-count':[], 'prototypical-left':[], 'prototypical-right':[],
              'prototypical-total':[]}
  cig_sts, cig_ets = get_cigarette_bounds(alist)
  for st, et in zip(cig_sts.index, cig_ets.index): # each cigarette
    cig_list = alist[st:et+1]
    if compute == 'left-puffs':
      # get number of left puffs
      lpuff_filt = (cig_list[s_annotation.puff_col] == 'left-puff') & (cig_list[s_annotation.puff_col] != cig_list[s_annotation.puff_col].shift(1))
      results['left-puffs'].append(cig_list[lpuff_filt].shape[0])
    elif compute == 'right-puffs':
      # get number of right puffs
      rpuff_filt = (cig_list[s_annotation.puff_col] == 'right-puff') & (cig_list[s_annotation.puff_col] != cig_list[s_annotation.puff_col].shift(1))
      results['right-puffs'].append(cig_list[rpuff_filt].shape[0])
    elif compute == 'num-of-puffs':
      # get number of total puffs
      tpuff_filt = (cig_list[s_annotation.puff_col] != 'no-puff') & (cig_list[s_annotation.puff_col] != cig_list[s_annotation.puff_col].shift(1))
      results['num-of-puffs'].append(cig_list[tpuff_filt].shape[0])
    elif compute == 'prototypical-left' and 'prototypical?' in cig_list.columns:
      # get number of prototypical left puffs
      plpuff_filt = (cig_list[s_annotation.puff_col] == 'left-puff') & (cig_list[s_annotation.puff_col] != cig_list[s_annotation.puff_col].shift(1)) & (cig_list[s_info.prototypcal_col] == 1)
      results['prototypical-left'].append(cig_list[plpuff_filt].shape[0])
    elif compute == 'prototypical-right' and 'prototypical?' in cig_list.columns:
      # get number of prototypical right puffs
      prpuff_filt = (cig_list[s_annotation.puff_col] == 'right-puff') & (cig_list[s_annotation.puff_col] != cig_list[s_annotation.puff_col].shift(1)) & (cig_list[s_info.prototypcal_col] == 1)
      results['prototypical-right'].append(cig_list[prpuff_filt].shape[0])
    elif compute == 'prototypical-total' and 'prototypical?' in cig_list.columns:
      # get number of total prototypical puffs
      ptpuff_filt = (cig_list[s_annotation.puff_col] != 'no-puff') & (cig_list[s_annotation.puff_col] != cig_list[s_annotation.puff_col].shift(1)) & (cig_list[s_info.prototypcal_col] == 1)
      results['prototypical-total'].append(cig_list[ptpuff_filt].shape[0])
    elif compute == 'hand-swap-count':
      # get hand swapping counts
      tpuff_filt = (cig_list[s_annotation.puff_col] != 'no-puff') & (cig_list[s_annotation.puff_col] != cig_list[s_annotation.puff_col].shift(1))
      puff_list = cig_list[tpuff_filt]
      hand_swaps = puff_list[puff_list[s_annotation.puff_col] != puff_list[s_annotation.puff_col].shift(1).fillna(method='bfill')]
      results['hand-swap-count'].append(len(hand_swaps))
  result = results[compute]
  result_set = pd.DataFrame(result, columns=['count',])
  result_set['index'] = range(1,len(cig_sts)+1)
  result_set = result_set.set_index('index')
  temp = result_set
  result_set = result_set.append(pd.DataFrame([result_set['count'].sum(),], index=['sum',], columns=['count',]))
  result_set = result_set.append(temp.describe())
  return result_set

def get_interpuff_intervals_statistics(alist):
  """ For one subject
  """
  interval_results = []
  interval_statistics = []
  cig_sts, cig_ets = get_cigarette_bounds(alist)
  for st, et in zip(cig_sts.index, cig_ets.index):
    cig_list = alist[st:et+1]
    interval_st = (cig_list[s_annotation.puff_col] == 'no-puff') & (cig_list[s_annotation.puff_col] != cig_list[s_annotation.puff_col].shift(1))
    interval_et = (cig_list[s_annotation.puff_col] == 'no-puff') & (cig_list[s_annotation.puff_col] != cig_list[s_annotation.puff_col].shift(-1))
    interval_st_index = cig_list[interval_st].index
    interval_et_index = cig_list[interval_et].index
    first_p_index = cig_list[cig_list[s_annotation.puff_col] != 'no-puff'].index[0]
    last_p_index = cig_list[cig_list[s_annotation.puff_col] != 'no-puff'].index[-1]
    interval_st_index = interval_st_index[interval_st_index >= first_p_index]
    interval_et_index = interval_et_index[interval_et_index >= first_p_index]
    interval_st_index = interval_st_index[interval_st_index <= last_p_index]
    interval_et_index = interval_et_index[interval_et_index <= last_p_index]
    retrieved_intervals = alist.iloc[interval_st_index]
    retrieved_intervals[s_annotation.et_col] = alist.ix[interval_et_index,s_annotation.et_col].values # change end time
    
    # add unique activities/postures to each interval
    for ist, iet in zip(interval_st_index, interval_et_index):
      retrieved_intervals.ix[ist,s_annotation.activity_col]=' '.join(alist[ist:iet+1][s_annotation.activity_col].unique()).strip()
    
    retrieved_intervals = retrieved_intervals.drop([s_annotation.smoke_col,s_annotation.puff_col], axis=1)
    # compute interval duration
    retrieved_intervals['duration'] = retrieved_intervals[s_annotation.et_col] - retrieved_intervals[s_annotation.st_col]
    retrieved_intervals['duration'] = retrieved_intervals['duration'].apply(lambda x: x / np.timedelta64(1, 's'))
    interval_results.append(retrieved_intervals)

    # compute statistics for this cigarette
    temp = retrieved_intervals['duration'].describe().append(pd.Series(retrieved_intervals['duration'].sum(),['sum',]))
    interval_statistics.append(temp)
  stats_set = pd.concat(interval_statistics, keys=range(1,len(cig_sts)+1))
  result_set = pd.concat(interval_results, keys=range(1, len(cig_sts)+1))
  return result_set, stats_set


def get_puff_duration_statistics(alist):
  """ For one subject
  """
  puff_results = []
  duration_statistics = []
  cig_sts, cig_ets = get_cigarette_bounds(alist)
  for st, et in zip(cig_sts.index, cig_ets.index):
    cig_list = alist[st:et+1]
    puff_st = (cig_list[s_annotation.puff_col] != 'no-puff') & (cig_list[s_annotation.puff_col] != cig_list[s_annotation.puff_col].shift(1))
    puff_et = (cig_list[s_annotation.puff_col] != 'no-puff') & (cig_list[s_annotation.puff_col] != cig_list[s_annotation.puff_col].shift(-1))
    puff_st_index = cig_list[puff_st].index
    puff_et_index = cig_list[puff_et].index
    retrieved_puffs = alist.iloc[puff_st_index]
    retrieved_puffs[s_annotation.et_col] = alist.ix[puff_et_index,s_annotation.et_col].values # change end time
    
    # add unique activities/postures to each puff
    for ist, iet in zip(puff_st_index, puff_et_index):
      retrieved_puffs.ix[ist,s_annotation.activity_col]=' '.join(alist[ist:iet+1][s_annotation.activity_col].unique()).strip()

    retrieved_puffs = retrieved_puffs.drop([s_annotation.smoke_col,], axis=1)
    # compute puff duration for each puff
    retrieved_puffs['duration'] = retrieved_puffs[s_annotation.et_col] - retrieved_puffs[s_annotation.st_col]
    retrieved_puffs['duration'] = retrieved_puffs['duration'].apply(lambda x: x / np.timedelta64(1, 's'))
    puff_results.append(retrieved_puffs)
    # compute statistics for this cigarette
    temp = retrieved_puffs['duration'].describe().append(pd.Series(retrieved_puffs['duration'].sum(),['sum',]))
    duration_statistics.append(temp)
  stats_set = pd.concat(duration_statistics, keys=range(1, len(cig_sts)+1))
  result_set = pd.concat(puff_results, keys=range(1, len(cig_sts)+1))
  return result_set, stats_set

def get_smoke_duration_statistics(alist):
  """ Get smoking duration statistics
  """
  cig_sts, cig_ets = get_cigarette_bounds(alist)
  sduration = cig_ets[s_annotation.et_col] - cig_sts[s_annotation.st_col].values
  sduration = pd.DataFrame([sduration.values,range(1,len(cig_sts)+1)],['duration','index']).transpose().set_index('index')
  result_set = sduration
  result_set['duration'] = result_set['duration'].apply(lambda x: x / np.timedelta64(1, 's'))
  temp = pd.Series(result_set['duration'].sum(),['sum',]).append(result_set['duration'].describe())
  result_set = result_set.append(pd.DataFrame(temp,columns=['duration',]))
  return result_set

def get_derived_statistics(alist, compute='hand-swap-rate'):
  result = []
  cig_sts, cig_ets = get_cigarette_bounds(alist)
  for st, et in zip(cig_sts.index, cig_ets.index):
    # puff numbers related
    cig_list = alist[st:et+1]
    # get necessary infos
    puff_st = (cig_list[s_annotation.puff_col] != 'no-puff') & (cig_list[s_annotation.puff_col] != cig_list[s_annotation.puff_col].shift(1))
    puff_st_index = cig_list[puff_st].index
    retrieved_puffs = alist.iloc[puff_st_index]
    total_puffs = len(retrieved_puffs)
    smoke_duration = (cig_list[s_annotation.et_col][et] - cig_list[s_annotation.st_col][st]).total_seconds()
    if compute == 'puff-speed':
      result.append(total_puffs/float(smoke_duration)*60) # in minutes
    elif compute == 'hand-swap-rate':
      hand_swaps = retrieved_puffs[retrieved_puffs[s_annotation.puff_col] != retrieved_puffs[s_annotation.puff_col].shift(1).fillna(method='bfill')]
      result.append(len(hand_swaps)/float(total_puffs)) # in percentage
    elif compute == 'prototypical-left-percentage' and 'prototypical?' in cig_list.columns:
      lefts = retrieved_puffs[retrieved_puffs[s_annotation.puff_col] == 'left-puff']
      prototypical_lefts = lefts[lefts[s_info.prototypcal_col] == 1]
      if len(lefts) == 0:
        result.append(np.NaN)
      else:
        result.append(len(prototypical_lefts)/float(len(lefts)))
    elif compute == 'prototypical-right-percentage' and 'prototypical?' in cig_list.columns:
      rights = retrieved_puffs[retrieved_puffs[s_annotation.puff_col] == 'right-puff']
      prototypical_rights = rights[rights[s_info.prototypcal_col] == 1]
      if len(rights) == 0:
        result.append(np.NaN)
      else:
        result.append(len(prototypical_rights)/float(len(rights)))
    elif compute == 'prototypical-total-percentage' and 'prototypical?' in cig_list.columns:
      prototypicals = retrieved_puffs[retrieved_puffs[s_info.prototypcal_col] == 1]
      result.append(len(prototypicals)/float(total_puffs))
  if compute == 'puff-speed':
    result = pd.DataFrame(result,index=range(1,len(cig_sts)+1),columns=['counts/minutes',])
  else:
    result = pd.DataFrame(result,index=range(1,len(cig_sts)+1),columns=['percentage',])
  result = result.append(result.describe())
  return result

def get_complexity_statistics(alist, compute='superposition'):
  """ Compute the proportion of activity superposition and ambiguity when smoking
  """
  result = []
  cig_sts, cig_ets = get_cigarette_bounds(alist)
  for st, et in zip(cig_sts.index, cig_ets.index):
    cig_list = alist[st:et+1]
    if compute == 'superposition':
      selected_index = (cig_list[s_annotation.activity_col] != 'no-activity') | (cig_list[s_annotation.post_col] == 'walking')
    elif compute == 'ambiguity':
      selected_index = (cig_list[s_annotation.activity_col] == 'drinking-beverage') | (cig_list[s_annotation.activity_col] == 'eating-a-meal')
    selected_duration = cig_list[selected_index][s_annotation.et_col] - cig_list[selected_index][s_annotation.st_col].values
    selected_seconds = selected_duration.apply(lambda x: x / np.timedelta64(1, 's'))
    selected_total_sec = selected_seconds.sum()
    smoking_total_sec = (cig_list[s_annotation.et_col][et] - cig_list[s_annotation.st_col][st]).total_seconds()
    selected_percentage = selected_total_sec/float(smoking_total_sec)
    result.append(selected_percentage)
  if compute == 'superposition':
    result = pd.DataFrame(result, index=range(1,len(cig_sts)+1),columns=['activity superposition(percentage)',])
  elif compute == 'ambiguity':
    result = pd.DataFrame(result, index=range(1,len(cig_sts)+1),columns=['activity ambiguity(percentage)',])
  result = result.append(result.describe())
  return result

def main():
  npuff_names = ['left-puffs','prototypical-left','right-puffs','prototypical-right','num-of-puffs','prototypical-total','hand-swap-count']
  npuff_statistics = {'left-puffs':[],'right-puffs':[],'num-of-puffs':[],'hand-swap-count':[],
                      'prototypical-left':[],'prototypical-right':[],'prototypical-total':[]}
  derived_names = ['hand-swap-rate','puff-speed','prototypical-left-percentage','prototypical-right-percentage','prototypical-total-percentage']
  derived_statistics = {'hand-swap-rate':[],'puff-speed':[],'prototypical-left-percentage':[],'prototypical-right-percentage':[],'prototypical-total-percentage':[]}
  complexity_names = ['superposition','ambiguity']
  complexity_statistics = {'superposition':[],'ambiguity':[]}
  dsmoke_statistics = []
  dpuff_statistics = []
  ipuff_statistics = []
  interpuff_stats = []
  interpuff_lists = []
  puffduration_lists = []
  puffduration_stats = []
  for i in s_info.session_arr:
    annotation_file = 'session' + str(i) + '.annotation.csv'
    # alist = s_annotation.annotation_csv_importer(s_info.clean_dataset_folder + annotation_file)
    # use puff corrected dataset
    alist = s_annotation.annotation_csv_importer(s_info.puff_corrected_folder + annotation_file)
    for name in npuff_names:
      npuff_statistics[name].append(get_num_of_puffs_statistics(alist, compute=name))
    dsmoke_statistics.append(get_smoke_duration_statistics(alist))
    interpuff_list, interpuff_stat = get_interpuff_intervals_statistics(alist)
    interpuff_stats.append(interpuff_stat)
    interpuff_lists.append(interpuff_list)
    puffduration_list, puffduration_stat = get_puff_duration_statistics(alist)
    puffduration_stats.append(puffduration_stat)
    puffduration_lists.append(puffduration_list)
    for name in derived_names:
      derived_statistics[name].append(get_derived_statistics(alist,compute=name))
    for name in complexity_names:
      complexity_statistics[name].append(get_complexity_statistics(alist, compute=name))


  dsmoke_statistics = pd.concat(dsmoke_statistics, keys=s_info.session_arr, axis=1)
  for name in npuff_names:
    npuff_statistics[name] = pd.concat(npuff_statistics[name],keys=s_info.session_arr,axis=1)
  npuff_statistics = pd.concat(npuff_statistics, keys=npuff_names)
  dpuff_statistics = pd.concat(puffduration_stats, keys=s_info.session_arr, axis=1)
  ipuff_statistics = pd.concat(interpuff_stats, keys=s_info.session_arr, axis=1)
  for name in derived_names:
    derived_statistics[name] = pd.concat(derived_statistics[name], keys=s_info.session_arr, axis=1)
  for name in complexity_names:
    complexity_statistics[name] = pd.concat(complexity_statistics[name], keys=s_info.session_arr, axis=1)
  # save
  writer = pd.ExcelWriter(s_info.stat_dataset_folder + 'stat.xlsx')
  dsmoke_statistics.to_excel(writer, sheet_name='smoking duration')
  npuff_statistics.to_excel(writer, sheet_name='puff counts')
  dpuff_statistics.to_excel(writer, sheet_name='puff-duration')
  ipuff_statistics.to_excel(writer, sheet_name='interpuff-interval')
  for name in derived_names:
    derived_statistics[name].to_excel(writer, sheet_name=name)
  for name in complexity_names:
    complexity_statistics[name].to_excel(writer, sheet_name='activity-' + name)
  c = 1
  for interpuff_li, puffduration_li in zip(interpuff_lists, puffduration_lists):
    interpuff_li.to_excel(writer, sheet_name='interpuff-session'+str(c))
    puffduration_li.to_excel(writer, sheet_name='puffduration-session'+str(c))
    c += 1
  writer.save()
  return


if __name__ == "__main__":
  main()