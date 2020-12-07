"""
  Function to Load wockets raw dataframe and smoking annotation dataframe given 
  various options: session(only support single session), sensors, type (raw, clean or puff corrected)  
"""

import sys
sys.path.append('../')
import pandas as pd
import smdt.info as s_info
import wockets.utils as w_utils
import smdt.annotation as s_annotation

def load_smoking_df(session=1, sensors=['DW',], kind='corrected'):
  """
  Load wockets raw dataframe and smoking annotation dataframe given 
  various options

  Args:
    session: default is session 1, session number to be loaded
    sensors: default is 'DW', sensor code array to specify sensors to be loaded
    type: default is 'corrected', what type of dataset to be loaded, currently support
          "raw", "clean" and "corrected"
  Return:
    raw_df, annotation_df
  """
  if kind == 'raw':
    folder = s_info.raw_dataset_folder
  elif kind == 'clean':
    folder = s_info.clean_dataset_folder
  elif kind == 'corrected':
    folder = s_info.puff_corrected_folder
  else:
    raise RuntimeError
  raw_filenames = [folder +'/session' + str(session) + '_' + sensor + '.raw.csv' for sensor in sensors]
  annotation_filenames = [folder + '/session' + str(session) + '.annotation.csv' for sensor in sensors]
  print "======start loading wockets raw csv files: ========"
  print '\n'.join(raw_filenames)
  raw_df = w_utils.raw_csv_consolidator(raw_filenames, sessions=[session,]*len(sensors), sensors=sensors)
  print "======start loading smoking annotation csv files: ========"
  print '\n'.join(annotation_filenames)
  annotation_df = s_annotation.annotation_csv_consolidator(annotation_filenames, sessions=[session,]*len(sensors), sensors=sensors)
  print "=====================end loading files======================"
  return raw_df, annotation_df

def load_prediction_df(session='all',window_size=1000,sensor='BFW', validation='kfold'):
  if session == 'all':
    filename = s_info.prediction_dataset_folder + '/session_' + session + '_' + sensor + '.' + str(window_size) + '.' + validation + '.prediction.csv'
  else:
    filename = s_info.prediction_dataset_folder + '/session' + str(session) + '_' + sensor + '.' + str(window_size) + '.' + validation + '.prediction.csv'
  prediction_df = pd.read_csv(filename)
  return prediction_df

def load_smokeprediction_df(session='all',window_size=1000,sensor='BFW', validation='kfold'):
  if session == 'all':
    filename = s_info.smoke_dataset_folder + '/session_' + session + '_' + sensor + '.' + str(window_size) + '.' + validation + '.smoke.csv'
  else:
    filename = s_info.smoke_dataset_folder + '/session' + str(session) + '_' + sensor + '.' + str(window_size) + '.' + validation + '.smoke.csv'
  smoke_df = pd.read_csv(filename)
  return smoke_df

def test_load_smoking_df():
  print "input parameters:"
  print "session: " + str(1)
  print "sensors: DW, NDW"
  print "kind: corrected"
  raw_df, annotation_df = load_smoking_df(session=1, sensors=['DW','NDW'], kind='corrected')
  print "===========test loading results============================="
  print raw_df.groupby('sensor').head()
  print annotation_df.groupby('sensor').head()

def test_load_prediction_df():
  session = 'all'
  sensor = 'BFW'
  window_size = 1000
  validation = 'kfold'
  print "input parameters:"
  print "session: " + session
  print "sensors: " + sensor
  prediction_df = load_prediction_df(session=session, window_size=window_size, sensor=sensor, validation=validation)
  print "===========test loading results============================="
  print prediction_df.groupby(s_info.session_col).head(1).T

if __name__ == '__main__':
  # s_info.test_func(test_load_smoking_df, profile=False)
  s_info.test_func(test_load_prediction_df, profile=False)