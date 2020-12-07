#! /usr/bin/python

"""
Module for reading wockets data and other utility functions 
"""

import logging
import random as random
from datetime import datetime, timedelta
import calendar
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
import smdt.info as s_info


raw_col_names = ['ts','rawx','rawy','rawz']
raw_header_index = None
raw_use_cols = raw_col_names
raw_ts_col = 0
raw_ts_name = 'ts'
raw_value_names = ['rawx','rawy','rawz']
raw_ubound = 1023
raw_lbound = 0
raw_sample_rate = 40 # 40samples/s

def convert_fromtimestamp(unix_timestamp):
  # print unix_timestamp
  return datetime.utcfromtimestamp(int(unix_timestamp)/1000.0)     # use UTC time, unix_timestamp is 13 digits

def convert_totimestamp(dtime):
  # print "%f" % (calendar.timegm(dtime.utctimetuple())*1000 + dtime.microsecond/1000)
  return calendar.timegm(dtime.utctimetuple())*1000 + dtime.microsecond/1000         # use UTC time

def convert_fromstring(tsstr, format):
  try:
    return datetime.strptime(tsstr, format)     # use UTC time
  except ValueError:
    return np.nan

def generate_random_bounds(raw_data, duration=timedelta(seconds=30)):
  """ Generate upper and lower timestamp bounds randomly

  Generate upper and lower timestamp bounds by randomly selecting a timestamp
  from raw data as lower bound and add the specified duration to get upper bound

  Args:
    raw_data: raw dataset used for timestamp selection
    duration: timedelta object to specify range duration, default is 30s.

  Returns:
    tuple of (lower bound, upper bound)
  """

  time1 = random.choice(raw_data[raw_ts_name])
  time1 = time1.replace(microsecond=0)
  time2 = time1 + duration
  return (time1, time2)

def raw_csv_importer(filename):

  # read_csv with options:
  # names: column names
  # header: row header index
  # usecols: column index selected to be imported
  # converters: dict to map column to a format converter function  
  csv_data = pd.read_csv(filename, names=raw_col_names, header=raw_header_index, 
                         usecols=raw_use_cols)
  csv_data[raw_ts_name] = csv_data[raw_ts_name].apply(convert_fromtimestamp)
  logging.info("import %s raw data file", filename)
  return csv_data

def raw_csv_exporter(raw_data, filename):
  raw_data[raw_ts_name] = raw_data[raw_ts_name].apply(convert_totimestamp)
  raw_data.to_csv(filename, index=False, header=False)

def raw_data_consolidator(raw_datas, sessions=[], subjects=[], sensors=[]):
  """combine list of raw datasets

  combine list of raw datasets into one by raw dataset, index will be reset

  Args:
    raw_datas: list of raw dataset

  Returns: 
    consolidated raw dataset
  """
  c=0
  single_sess_datas = []
  for single_sess_data in raw_datas:
    if np.iterable(sessions) and len(sessions) == len(raw_datas):
      single_sess_index = [sessions[c],]*single_sess_data.shape[0]
      single_sess_data[s_info.session_col] = single_sess_index
    if np.iterable(subjects) and len(subjects) == len(raw_datas):
      single_subj_index = [subjects[c],]*single_sess_data.shape[0]
      single_sess_data['subject'] = single_subj_index
    if np.iterable(sensors) and len(sensors) == len(raw_datas):
      single_sensor_index = [sensors[c],]*single_sess_data.shape[0]
      single_sess_data[s_info.sensor_col] = single_sensor_index
    c+=1
    single_sess_datas.append(single_sess_data)
  consolidate_csv_data = pd.concat(single_sess_datas)
  # consolidate_csv_data = consolidate_csv_data.reset_index(drop=False)
  # consolidate_csv_data = consolidate_csv_data.rename(columns={'index':'index per sensor'})
  logging.info('consolidate %d raw datasets into one', len(raw_datas))
  return consolidate_csv_data
  
def raw_csv_consolidator(filenames, sessions=[], subjects=[], sensors=[]):
  """Import list of raw data files and combine them into one big raw dataset

  Import list of raw data files, add session, subject and sensor information of
  each raw dataset into the big raw dataset

  Args:
    filenames: list of raw data filepath strings
    sessions: list of session numbers
    subjects: list of subject strings
    sensors: list of sensor strings/numbers

  Returns:
    consolidated raw dataset
  """

  single_sess_datas = []
  c = 0
  for filename in filenames:
    single_sess_data = raw_csv_importer(filename)
    if np.iterable(sessions) and len(sessions) == len(filenames):
      single_sess_index = sessions[c]
      single_sess_data[s_info.session_col] = single_sess_index
    if np.iterable(subjects) and len(subjects) == len(filenames):
      single_subj_index = subjects[c]
      single_sess_data['subject'] = single_subj_index
    if np.iterable(sensors) and len(sensors) == len(filenames):
      single_sensor_index = sensors[c]
      single_sess_data[s_info.sensor_col] = single_sensor_index
    single_sess_datas.append(single_sess_data)
    c += 1
  consolidate_csv_data = pd.concat(single_sess_datas)
  # consolidate_csv_data = consolidate_csv_data.reset_index(drop=False)
  # consolidate_csv_data = consolidate_csv_data.rename(columns={'index':'index per sensor'})
  logging.info('consolidate %d raw data files into one raw dataset', len(filenames))
  return consolidate_csv_data

def unit_tests(testfile):

  # test raw_csv_importer
  t_data1 = raw_csv_importer(testfile)
  print "test raw_csv_importer"
  print "length passed:", len(t_data1) == 280321
  print "column passed:", np.all(t_data1.columns == [u'ts', u'rawx', u'rawy', u'rawz'])
  print "first row passed:", np.all(t_data1.values[0] == [convert_fromtimestamp(1336043562000),0.7265625,0.0625,0.0625])
  print "last row passed:", np.all(t_data1.values[-1] == [convert_fromtimestamp(1336050570000),0.1484375,-0.6328125,-0.1796875])
  print "======================"

  # test raw_data_consolidator
  test_datasets = []
  test_datasets.append(pd.DataFrame({'one' : [1., 2., 3., 4.],'two' : [0., 0., 0., 0.]}))
  test_datasets.append(pd.DataFrame({'one' : [5., 6., 7., 8.],'two' : [0., 0., 0., 0.]}))
  t_data2 = raw_data_consolidator(test_datasets)
  print "test raw_data_consolidator"
  print "length passed:", len(t_data2) == 4*2
  print "column passed:", np.all(t_data2.columns == [u'one', u'two'])
  print "first row passed:", np.all(t_data2.values[0] == [1., 0.])
  print "last row passed:", np.all(t_data2.values[-1] == [8., 0.])
  print "======================"

  # test raw_csv_consolidator
  testfiles = [testfile, testfile]
  sessions = [1,2]
  subjects = [3,4]
  sensors = ['left wrist','right wrist']
  t_data3 = raw_csv_consolidator(testfiles, sessions, subjects, sensors)
  print "test raw_csv_consolidator"
  print "length passed:", len(t_data3) == 280321*2
  print "column passed:", np.all(t_data3.columns == [u'ts', u'rawx', u'rawy', u'rawz', u'session', u'subject', u'sensor'])
  print "first row passed:", np.all(t_data3.values[0] == [convert_fromtimestamp(1336043562000),0.7265625,0.0625,0.0625, 1, 3, 'left wrist'])
  print "last row passed:", np.all(t_data3.values[-1] == [convert_fromtimestamp(1336050570000),0.1484375,-0.6328125,-0.1796875, 2, 4, 'right wrist'])
  print "======================"

  # test convert_fromtimestamp
  test_timestamp = 1381417810500
  t_timestamp = convert_fromtimestamp(test_timestamp)
  print "test convert_fromtimestamp"
  print "conversion passed:", str(t_timestamp) == "2013-10-10 15:10:10.500000"
  print "======================"

if __name__ == "__main__":
  logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s', level=logging.WARNING)
  testfile = "../../test_DW.raw.csv"
  unit_tests(testfile)
