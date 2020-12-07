#! /usr/bin/python

"""
  Script to show randomly selected time range plot
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as pyplot
import scipy.signal as sp_signal

import wockets.utils as w_utils
import smdt.viewer as s_viewer
import smdt.annotation as s_annotation
import smdt.raw as s_raw
import feature.slope_based as f_slope_based
import smdt.segment as s_segment


def test_peak_rate_computation(selected_raw, selected_a, mean_raw, std_raw):
  # playground
  # preprocess
  
  selected_raw = s_viewer._prepare_raw_for_plot(selected_raw)
  temp = s_raw.filter_raw(selected_raw, w_utils.raw_value_names, filter_type='median', paras={'kernel_size': 5})
  temp = (temp - mean_raw)/std_raw
  temp = s_raw.filter_raw(temp, w_utils.raw_value_names, filter_type='gaussian', paras={'sigma': 2})
  # LoG
  temp = s_raw.filter_raw(temp, raw_value_names=w_utils.raw_value_names, filter_type='NLoG', paras={'sigma':5})
  for name,c in zip(['rawx','rawy','rawz'], [1,2,3]):
      peakind = sp_signal.argrelmax(temp[name].values, order=3)
      # peakind = sp_signal.find_peaks_cwt(temp[name].values, np.arange(3,10), min_length=4)
      peakvalue = pd.DataFrame(temp[name].ix[peakind])
      peakvalue = peakvalue.sort([name,],ascending=False)
      peakvalue = peakvalue[peakvalue >= peakvalue.iloc[0]/4]
      peakvalue = peakvalue.dropna()
      peakind = np.vstack(peakvalue.index)[:,0]
      peakrate = len(peakind)/float(32)
      print peakrate
      if c == 1:
        ax = pyplot.subplot(2,3,c)
        sharex = ax
      else:
        ax = pyplot.subplot(2,3,c,sharex=sharex)
      s_viewer.get_singlesensor_raw_plot(selected_raw.pop(name), labels=selected_a, subplots=False, ax=ax, title=name)
      ax = pyplot.subplot(2,3,c+3, sharex=sharex)
      s_viewer.get_singlesensor_raw_plot(temp.pop(name), labels=selected_a, subplots=False, ax=ax, title=name)
      ax.set_ylim((-0.2,0.2))
      pyplot.scatter(peakind, peakvalue, marker="*")
      ax.annotate(str(peakrate), xy=(0,0),xytext=(0, 0.15))


def test_preprocessing(selected_raw, selected_a, mean_raw, std_raw):
  # playground
  # preprocess
  selected_raw = s_raw.filter_raw(selected_raw, w_utils.raw_value_names, filter_type='median', paras={'kernel_size': 5})
  selected_raw = s_viewer._prepare_raw_for_plot(selected_raw)
  selected_raws = [selected_raw,]
  temp = (selected_raw - mean_raw)/std_raw
  temp = s_raw.filter_raw(temp, w_utils.raw_value_names, filter_type='gaussian', paras={'sigma': 2})
  selected_raws.append(temp)
  names = ['original','After']
  #plot
  for raw, c, name in zip(selected_raws,range(1,len(selected_raws)+1), names):
    if c == 1:
      ax = pyplot.subplot(len(selected_raws),2,c)
      preax = ax
    else:
      ax = pyplot.subplot(len(selected_raws),2,c, sharex=preax)
      preax = ax
    s_viewer.get_singlesensor_raw_plot(raw, labels=selected_a, subplots=False, ax=ax,
                              figsize=(20,10), color=['blue','cyan','gray'], title=name)
    # if c > 1:
    #   ax.set_ylim((raw.values.min()-0.2*abs(raw.values.min()), raw.values.max()*1.2))
    
  ax = pyplot.subplot(len(selected_raws),2,c+1)
  s_viewer.get_legends_plot()


def test_linearRegression(selected_raw, selected_a):
  # playground
  # test linear regression
  (seg_set, anchor_arr) = s_segment.get_segments(selected_raw, paras={'window_size':150, 'overlap_rate':0.5})
  (annotation_set, annotation_data) = s_segment.get_seg_annotations(selected_a, seg_set)

  for seg, annotation,d in zip(seg_set[6:8],annotation_data[6:8], [0,3]):

    
   

    slopes = f_slope_based.compute_slope(seg.values[2:])[2:]
    intercepts = f_slope_based.compute_intercept(seg.values[2:])[2:]

    seg = s_viewer._prepare_raw_for_plot(seg)
    print slopes
    for name,c, slope, intercept in zip(['rawx','rawy','rawz'], [1,2,3], slopes, intercepts): 
      ax = pyplot.subplot(2,3,c+d)
      s_viewer.get_singlesensor_raw_plot(seg.pop(name), labels=annotation, subplots=False, ax=ax, title=name)
      x = np.array(range(0, len(seg.index)),dtype='float64')
      pyplot.plot(x, x*slope+intercept)

    # if c > 1:
    #   ax.set_ylim((raw.values.min()-0.2*abs(raw.values.min()), raw.values.max()*1.2))
    
  # ax = pyplot.subplot(2,len(names),c+1)
  # s_viewer.get_legends_plot()

def main():
  testfile_raw = "../puff_corrected_dataset/session7_DW.raw.csv"
  testfile_annotation = "../puff_corrected_dataset/session7.annotation.csv"
  raw_df = w_utils.raw_csv_importer(testfile_raw)
  annotation_df = s_annotation.annotation_csv_importer(testfile_annotation)
  mean_raw = raw_df.mean()
  std_raw = raw_df.std()
  # select by label
  labels = [['right-puff',],['right-puff',],['walking',],['eating-a-meal',],['sitting','not-smoking','no-activity']]
  seeds = [0, 10, 2, 3, 10]
  for label, seed in zip(labels, seeds):
    lbound, rbound = s_annotation.generate_bounds_by_labels(annotation_df, duration=timedelta(seconds=30), labels=label, seed=seed)

    selected_a = s_annotation.select_annotation_by_ts(annotation_df, lbound=lbound, rbound=rbound)
    selected_raw = s_raw.select_raw_by_ts(raw_df, lbound=lbound, rbound=rbound)
    # test_filtering(selected_raw, selected_a)
    # test_linearRegression(selected_raw, selected_a)
    pyplot.figure()
    pyplot.suptitle(label)
    # test_preprocessing(selected_raw, selected_a, mean_raw, std_raw)
    test_peak_rate_computation(selected_raw, selected_a, mean_raw, std_raw)
  pyplot.show()
  # pyplot.close()

if __name__ == '__main__':
  import cProfile
  pr = cProfile.Profile()
  pr.enable()
  main()
  pr.disable()
  pr.create_stats()
  pr.print_stats(sort=1)

