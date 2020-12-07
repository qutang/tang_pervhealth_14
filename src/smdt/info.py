#! /usr/bin/python

"""
  Store project related static information

  To run the experiment, first set the root path to be the current project directory
"""
import os

# global constants
root_path = os.path.abspath('/home/james/PythonProjects/smdt_python')

# folders
raw_dataset_folder = root_path + '/raw_dataset'
clean_dataset_folder = root_path + '/clean_dataset'
feature_dataset_folder = root_path + '/feature_dataset'
prediction_dataset_folder = root_path + '/prediction_dataset'
smoke_dataset_folder = root_path + '/smoke_dataset'
stat_dataset_folder = root_path + '/statistics_dataset'
puff_figure_folder = root_path + '/puff_figures'
post_figure_folder = root_path + '/posture_figures'
puff_corrected_folder = root_path + '/puff_corrected_dataset'
pkl_folder = root_path + '/pkl'

# sessions and sensors
session_arr = [1,3,4,5,6,7] # session 2 is put aside, very bad
sensor_codes = ['DAR','DW','NDW','DAK']
sensor_names = ['Dominant Arm', 'Dominant Wrist', 'Non-Dominant Wrist', 'Dominant Ankle']
sys_offsets = [0.0,0.0,20.0,0.0,4.0,-3.0] #session2: -22.0 # session4: flip axis # session5 NDW: axis flip
session_dates = ['2011-10-27', '2011-12-14', '2012-02-08', '2012-05-03', '2012-08-16','2012-11-06']

# columns
prototypcal_col = 'prototypical?'
puffduration_col = 'puff duration'
segduration_col = 'seg duration'
puff_with_segment_duration_col = 'inside puff/segment duration'
puff_with_puff_duration_col = 'inside puff/puff duration'
puffside_col = 'puff side'
smokeproportion_col = 'smoking percentage'
walkproportion_col = 'walking percentage'
drinkproportion_col = 'drinking percentage'
eatproportion_col = 'eating percentage'
phoneproportion_col = 'phone percentage'
computerproportion_col = 'computer percentage'
talkproportion_col = 'talk percentage'
readproportion_col = 'reading percentage'
carproportion_col = 'car percentage'
unknownproportion_col = 'unknown percentage'
spproportion_col = 'superposition percentage'
puffindex_col = 'puff index'
segment_col = 'segment'
sensor_col = 'sensor'
session_col = 'session'
classname_col = 'name'
classnum_col = 'target'
predictionnum_col = 'prediction num'
predictionname_col = 'prediction name'
predictionprob_col = 'prediction prob'
smokepredprob_col = 'smoking_pred_prob'
raw_col_names = ['ts','rawx','rawy','rawz']
raw_value_names = ['rawx','rawy','rawz']
raw_ts_name = 'ts'
raw_ts_col = 0
header_index = 0 # annotation csv header row index
raw_header_index = None
st_col = 'STARTTIME'
et_col = 'ENDTIME'
puff_col = 'puffing'
activity_col = 'activity'
post_col = 'posture'
smoke_col = 'smoking'
use_cols = ['STARTTIME','ENDTIME','posture','activity','smoking','puffing']
category_names = ['posture','activity','smoking','puffing']
label_dict = {'posture':['sitting','standing','walking','lying/lounging','unknown-posture'],
                          'activity':['eating-a-meal','drinking-beverage',
                                      'using-computer','using-phone',
                                      'reading-paper','in-car',
                                      'talking','walking-down-stairs',
                                      'unknown-activity'],
                          'smoking':['smoking','not-smoking'],
                          'puffing':['left-puff','right-puff','no-puff']}

raw_use_cols = raw_col_names
class_nums = [0,1,2,3,4,5,6,7,8,9]
raw_ubound = 1023
raw_lbound = 0
raw_sample_rate = 40 # 40samples/s
overlap_rate = 0.5
scale_bounds = (4, -4) # raw data scale range from -4g to 4g
tstr_format = '%Y-%m-%d %H:%M:%S'

feature_info_dict = {
  smokeproportion_col:[smoke_col, 'smoking'], 
  walkproportion_col:[post_col, 'walking'], 
  drinkproportion_col:[activity_col, 'drinking-beverage'],
  eatproportion_col:[activity_col, 'eating-a-meal'],
  phoneproportion_col:[activity_col, 'using-phone'],
  computerproportion_col:[activity_col, 'using-computer'],
  talkproportion_col:[activity_col, 'talking'],
  readproportion_col:[activity_col, 'reading-paper'],
  carproportion_col:[activity_col,'in-car'],
  unknownproportion_col:[activity_col,'unknown-activity']
}

def test_func(func, profile=True, *args):
  if profile:
    import cProfile
    # from pympler import tracker
    # tr = tracker.SummaryTracker()
    pr = cProfile.Profile()
    pr.enable()
    if args:
      func(args)
    else:
      func()
    # tr.print_diff()
    pr.disable()
    pr.create_stats()
    pr.print_stats(sort=2)
  else:
    if args:
      func(args)
    else:
      func()