#! /usr/bin/python

"""
 Module for processing smdt raw data
"""
import sys
from datetime import timedelta
sys.path.append("../")
import numpy as np
import scipy.signal
import scipy.fftpack
import scipy.ndimage.filters
import matplotlib.pyplot as pyplot
import annotation as s_annotation
import wockets.utils as w_utils
import smdt.info as s_info

def scale_raw(raw_data,ubound,lbound):
  """linearly scale raw data into (lbound, ubound)

  Linearly scale every raw data sample into range of (lbound, ubound), wockets
  unscaled raw data is from 0 to 1023, which can be defined in wockets.utils

  Args:
    raw_data: raw dataset to be scaled
    ubound: upper bound of scale range
    lbound: lower bound of scale range

  Return: scaled raw dataset
  """

  scale_factor = (s_info.raw_ubound - s_info.raw_lbound + 1)/float(ubound - lbound)
  raw_data[s_info.raw_value_names] = raw_data[s_info.raw_value_names].apply(lambda x: x/scale_factor+lbound)
  return raw_data

def select_raw_by_ts(raw_data, lbound=None, rbound=None, by=None):
  """select a subset of raw dataset by given timestamp bounds

  Select a subset of raw dataset by given timestamp bounds. The selection can be
  done on raw dataset with multiple sessions, sensors or subjects. The selected
  subset will have index been reset

  Args: 
    raw_data: raw dataset to be selected
    lbound: lower bound timestamp, default is None which indicates the start time
    rbound: upper bound timestamp, default is None which indicates the end time
    by: group raw dataset by a column and then select subset from each group, 
    should be a string which corresponds to one of the columns. Default is None.

  Returns:
    selected subset of raw dataset
  """

  if by == None:
    if lbound == None:
      lbound = raw_data[s_info.raw_ts_name][0]
    if rbound == None:
      rbound = raw_data[s_info.raw_ts_name][-1]
    subset_raw_data = raw_data[raw_data[s_info.raw_ts_name].apply(lambda x:x>=lbound and x<=rbound)]
    subset_raw_data = subset_raw_data.reset_index(drop=True)
  else:
    subset_group_datas = []
    groupby_data = raw_data.groupby(by)
    for group_name, group_data in groupby_data:
      if lbound == None:
        lbound = group_data[s_info.raw_ts_name][0]
      if rbound == None:
        rbound = group_data[s_info.raw_ts_name][-1]
      subset_group_data = group_data[group_data[s_info.raw_ts_name].apply(lambda x:x>=lbound and x<=rbound)]
      subset_group_datas.append(subset_group_data)
    subset_raw_data = w_utils.raw_data_consolidator(subset_group_datas)
  return subset_raw_data

def select_raw_by_random(raw_data, duration=timedelta(seconds=30), by=None):
  lbound, rbound = w_utils.generate_random_bounds(raw_data, duration=duration)
  return select_raw_by_ts(raw_data, lbound=lbound, rbound=rbound, by=None)

def preprocess_raw(raw_data, annotation_data, grace_period=timedelta(minutes=3), by=None):
  """ Preprocess raw dataset

  Preprocess raw dataset by truncate it within the range of actual session, user
  can specify a grace period before and after that, and then scale the raw data
  into -4g to 4g range

  Args:
    raw_data: raw dataset to be preprocessed
    annotation_data: annotation dataset
    grace_period: timedelta used in truncation, default is 3min
    by: mainly used in truncation which can be thought as subset selection, this
    is used to specify a column name used to group raw dataset and then do truncation

  Returns:
    preprocessed raw dataset
  """

  if by != None:
    group_raw_datas = []
    grouped_raw = raw_data.groupby(by)
    grouped_annotation = annotation_data.groupby(by)
    for group_name, group_data in grouped_raw:
      # step 1: truncate raw data according to annotation start and end time
      lbound = annotation_data[s_annotation.st_col][grouped_annotation.groups[group_name][0]] - grace_period
      rbound = annotation_data[s_annotation.et_col][grouped_annotation.groups[group_name][-1]] + grace_period
      group_data = select_raw_by_ts(group_data,lbound,rbound)
      
      # step 2: scale raw data into [-4, 4]g from [0, 1023]
      group_data = scale_raw(group_data, s_info.scale_bounds[0], s_info.scale_bounds[1])

      group_raw_datas.append(group_data)
    raw_data = w_utils.raw_data_consolidator(group_raw_datas)
  else:
    # step 1: truncate raw data according to annotation start and end time    
    lbound = annotation_data[s_annotation.st_col][0] - grace_period
    rbound = annotation_data[s_annotation.et_col].iloc[-1] + grace_period
    raw_data = select_raw_by_ts(raw_data, lbound,rbound)
    
    # step 2: scale raw data into [-4, 4]g from [0, 1023]
    raw_data = scale_raw(raw_data, s_info.scale_bounds[0], s_info.scale_bounds[1])
  return raw_data

def filter_raw(raw_data, raw_value_names,filter_type=None, paras=None):
  """Apply filters onto raw dataset

  Apply different filters onto raw dataset, various filter's parameters can be 
  specified, this function doesn't support "group by", so be care when using it
  on multiple sessions, sensors and subjects

  Args:
    raw_data: raw dataset to be filtered
    raw_value_names: list of raw dataset value column names
    type: string of filter type
    paras: dict of mapping of parameter names to its values
           "median": {"kernel_size": 3}
           "lowpass": {"pass_freq": 0.01, "stop_freq": 0.1, "pass_loss": 1, "stop_loss": 80}
           "dcblock": {"p":0.95}
           "gaussian": {"sigma": 1}
  Returns:
    return filtered raw dataset
  """

  new_data = raw_data.copy(deep=True)
  if filter_type == 'median':
    if paras == None:
      paras['kernel_size'] = 5
    for value in raw_value_names:
      new_data[value] = scipy.signal.medfilt(new_data[value], paras['kernel_size'])
  elif filter_type == 'lowpass':
    if paras == None:
      paras['pass_freq'] = 0.01
      paras['stop_freq'] = 0.15
      paras['pass_loss'] = 1
      paras['stop_loss'] = 80
    # pass frequency: pass_freq*Fs/2 = 0.01*40/2 = 0.2Hz
    # stop frequency: stop_freq*Fs/2 = 0.1*40/2 = 2Hz
    # accordingly, walking frequency is above 1.5Hz, shaking frequency is also above 1.5Hz
    pf, sf, pg, sg = (paras['pass_freq'], paras['stop_freq'], paras['pass_loss'], paras['stop_loss'])
    ords, wn = scipy.signal.buttord(pf, sf, pg, sg)
    b,a = scipy.signal.butter(ords, wn, btype='low')
    for value in raw_value_names:
      new_data[value] = scipy.signal.lfilter(b,a,new_data[value])
  elif filter_type == 'dcblock':
    if paras == None:
      paras['p'] = 0.95
    # transfer function: H = 1 - z^-1/1 - p*z^-1
    # b = [1, -1]
    # a = [1, -p]
    # 0 < p < 1
    b = [1, -1]
    a = [1, -paras['p']]
    for value in raw_value_names:
      new_data[value] = scipy.signal.filtfilt(b,a,new_data[value])
  elif filter_type == 'gaussian':
    new_data[raw_value_names] = scipy.ndimage.filters.gaussian_filter1d(new_data[raw_value_names], sigma=paras['sigma'], axis=0, order=0, mode='reflect')
  elif filter_type == 'NLoG':
    new_data[raw_value_names] = paras['sigma']*scipy.ndimage.filters.gaussian_filter1d(new_data[raw_value_names], sigma=paras['sigma'], axis=0, order=2, mode='reflect')    
  # new_data = pd.DataFrame(new_data, columns=raw_data.columns)
  return new_data

def transform_raw(raw_data, transform_type=None, value_names=w_utils.raw_value_names):
  """transform raw dataset into other measurement or space

  Transform raw dataset into other measurement or space like orientation, posture
  distance, frequency domain

  Args:
    raw_data: raw dataset to be transformed
    transform_type: string to specify the transformation operation
    value_names: raw dataset's value column names

  Returns:
    (new_values_names, new_data): transformed dataset's value column names, 
    transformed new dataset
  """

  if transform_type == None:
    return s_info.raw_value_names, raw_data
  elif transform_type == 'orientation':
    # normalize to 1g
    raw_data.rawx = raw_data.rawx/np.sqrt(raw_data.rawx**2 + raw_data.rawy**2 + raw_data.rawz**2)
    raw_data.rawy = raw_data.rawy/np.sqrt(raw_data.rawx**2 + raw_data.rawy**2 + raw_data.rawz**2)
    raw_data.rawz = raw_data.rawz/np.sqrt(raw_data.rawx**2 + raw_data.rawy**2 + raw_data.rawz**2)
    # calculate orientations
    # pitch: yz, roll: xz, theta: xy
    pitch_data = np.arctan2(raw_data.rawx, np.sqrt(raw_data.rawy**2 + raw_data.rawz**2))*180/np.pi
    roll_data = np.arctan2(raw_data.rawy, np.sqrt(raw_data.rawx**2 + raw_data.rawz**2))*180/np.pi
    theta_data = np.arctan2(np.sqrt(raw_data.rawy**2 + raw_data.rawx**2),raw_data.rawz)*180/np.pi
    new_data = raw_data.drop(value_names, axis=1)
    new_data['pitch'] = pitch_data
    new_data['roll'] = roll_data
    new_data['theta'] = theta_data
    new_values_names = ['pitch','roll','theta']
  elif transform_type == 'magnitude':
    mag_data =  np.sqrt(raw_data.rawx**2 + raw_data.rawy**2 + raw_data.rawz**2)
    new_data = raw_data.drop(value_names, axis=1)
    new_data['magnitude'] = mag_data
    new_values_names = ['magnitude']
  elif transform_type == 'post-distance':
    xydistance =  np.abs(raw_data.rawx - raw_data.rawy)
    yzdistance =  np.abs(raw_data.rawy - raw_data.rawz)
    zxdistance =  np.abs(raw_data.rawz - raw_data.rawx)
    new_data = raw_data.drop(value_names, axis=1)
    new_data['xypostdist'] = xydistance
    new_data['yzpostdist'] = yzdistance
    new_data['zxpostdist'] = zxdistance
    new_values_names = ['xypostdist','yzpostdist','zxpostdist']
  elif transform_type == 'fft':
    xfft = np.abs(scipy.fftpack.fft(raw_data.rawx))
    yfft = np.abs(scipy.fftpack.fft(raw_data.rawy))
    zfft = np.abs(scipy.fftpack.fft(raw_data.rawz))
    new_data = raw_data.drop(value_names, axis=1)
    new_data['xfft'] = xfft
    new_data['yfft'] = yfft
    new_data['zfft'] = zfft
    new_values_names = ['xfft','yfft','zfft']
  return (new_values_names, new_data)

def unit_test():
  import viewer as s_viewer
  testfile_raw = "../../test.raw.csv"
  testfile_annotation = "../../test.annotation.csv"
  consol_raw = w_utils.raw_csv_consolidator([testfile_raw,],sessions=[5,])
  consol_annotation = s_annotation.annotation_csv_consolidator([testfile_annotation,],time_offsets=[0,],sessions=[5,])
  raw_data = preprocess_raw(consol_raw, consol_annotation, by='session')
  new_value_names, new_data = transform_raw(raw_data)
  lbound, rbound = w_utils.generate_random_bounds(new_data, duration=timedelta(minutes=5))
  random_raw = select_raw_by_ts(new_data, lbound, rbound)
  random_annotation = s_annotation.select_annotation_by_ts(consol_annotation, lbound, rbound)

  # s_viewer.get_simple_raw_plot(random_raw, labels=random_annotation, subplots=False)
  lowpass_raw = filter_raw(random_raw, new_value_names, filter_type='dcblock')
  # new_value_names, random_raw = transform_raw(random_raw, value_names=new_value_names, transform='fft')

  s_viewer.get_multistep_view_plot([random_raw, lowpass_raw], labels=random_annotation, subplots=True)
  # s_viewer.get_simple_raw_plot(random_raw, labels=random_annotation, subplots=False)
  # pyplot.show()
  # random_raw = scale_raw(random_raw,4,-4)
  # random_raw = random_raw.set_index([utils.raw_ts_name, 'session'])
  # random_raw.plot()
  pyplot.show()
if __name__ == "__main__":
  s_info.test_func(unit_test)