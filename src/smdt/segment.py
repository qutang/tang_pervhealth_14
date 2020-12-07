#! ../venv/bin/python

"""
  Module to segment the smoking dataset (wockets raw dataframe and smoking annotation dataframe)

  It will first call segmentation algorithm in segmentation module to get start and
  end anchors of each segment, and add an extra column called "segment" to the wockets
  raw dataframe and annotation dataframe with a segment index
"""
import sys
from pandas import concat as pd_concat
sys.path.append("../")
import smdt.loader as s_loader
import smdt.info as s_info

def _construct_segment_dataframe(raw_df, start_anchors, end_anchors):
  """Helper function to construct segment dataframe structure

  Args:
    raw_df: input wockets raw dataframe, only support single session and sensor
    start_anchors: start index of each segment
    end_anchors: end indexes of each segment
    drop: column index to be dropped

  Return:
    seg_df: segment dataframe
  """
  print "======segmentation information==============="
  
  # loop over anchors and append new segments
  # NOTE!! concating an array of dataframes is much faster than concat them one by one with append
  seg_arr = []
  indexes = range(0,len(start_anchors))
  print "========connect segments============="
  for start, end, i in zip(start_anchors, end_anchors, indexes):
    if end - start <= 1:
      continue
    new_seg_df = raw_df.iloc[start: end]
    new_seg_df[s_info.segment_col] = i
    seg_arr.append(new_seg_df)
  seg_df = pd_concat(seg_arr)
  print "total segmentations: " + str(seg_df[s_info.segment_col].max())
  # seg_df = raw_data.copy(deep=True) # only used for test
  return seg_df


def do_segmentation_on_raw(raw_data, method='window', paras={'window_size':200, 'overlap_rate':0.5}):
  """ do segmentation on the input raw dataframe, use the start and end anchors to form a new 
      raw dataframe by adding a "segment" column to the dataframe with segment index, it's possible some 
      rows will appear multiple times in different segment which is fine

  Args:
    raw_dataset: raw smoking dataset to be segmented
    method: string to specify which segmentation methods to use, default is 'window'
      'window': windowing segmentation with equal sized window
      'swab': SWAB algorithm
      'bottom_up': bottom up algorithm
      'sliding_window': sliding window algorithm
    paras: dict of parameters for each methods, default is {'window_size': 200, 'overlap_rate': 0.5}
      'window': window_size, overlap_rate
      'swab': max_error, buffer_size, column(column used as y)
      'bottom_up': max_error, column
      'sliding_window': max_error, column

  Return:
    seg_raw_df: raw segment dataframe
  """
  print "=========segmentation starts=============="
  print "method: " + method
  print paras

  if method == 'window':
    import segmentation.windowing as seg_win
    start_anchors, end_anchors = seg_win.standard_windowing_segment(raw_data.values, paras['window_size'], paras['overlap_rate'])
    seg_raw_df = _construct_segment_dataframe(raw_data, start_anchors, end_anchors)
  else:
    if paras['column'] is str:
      column = raw_data.columns.index[paras['column']]
    else:
      column = paras['column']
    raw_data = raw_data.reset_index(drop=False)
    import segmentation.linear_approx as seg_linear
    if method == 'swab':
      start_anchors, end_anchors = seg_linear.SWAB_segment(raw_data.values, max_error=paras['max_error'], buffer_size=paras['buffer_size'], column=column)
    elif method == 'bottom_up':
      start_anchors, end_anchors = seg_linear.bottom_up_merge(raw_data.values, max_error=paras['max_error'], column=column)
    seg_raw_df = _construct_segment_dataframe(raw_data, start_anchors, end_anchors)
  # reset index without drop
  seg_raw_df = seg_raw_df.reset_index(drop=False)
  seg_raw_df = seg_raw_df.rename(columns={"index":"reference index"})
  return seg_raw_df

def set_segmentation_on_annotation(annotation_df, seg_df):
  """ Get segment annotation dataframe by looking into the start and end timestamp
  of each segment, add a "segment" column to the dataframe

  Args:
    annotation_df: annotation dataset to be processed
    seg_df: input segment dataframe (output of function do_segmentation_on_raw)

  Return:
    seg_annotation_df: annotation segment dataframe
  """
  import smdt.annotation as s_annotation
  print "=========setting segment annotation dataset=============="
  # print annotation_df.head()
  seg_annotation_arr = []
  for seg_index, one_seg_df in seg_df.groupby(s_info.segment_col):
    start_time = one_seg_df[s_info.raw_ts_name].iloc[0]
    end_time = one_seg_df[s_info.raw_ts_name].iloc[-1]

    one_annotation_df = s_annotation.select_annotation_by_ts(annotation_df, lbound=start_time, rbound=end_time)
    one_annotation_df[s_info.segment_col] = [seg_index,]*len(one_annotation_df)
    seg_annotation_arr.append(one_annotation_df)
  seg_annotation_df = pd_concat(seg_annotation_arr)
  # reset index but keep the original index as a reference to previous dataframe
  seg_annotation_df = seg_annotation_df.reset_index(drop=False)
  # rename "index" to "reference index"
  seg_annotation_df = seg_annotation_df.rename(columns={"index":"reference index"})

  return seg_annotation_df

def unit_test():
  
  raw_df, annotation_df = s_loader.load_smoking_df(session=5, sensors=['DW',], kind='corrected')
  seg_raw_df = do_segmentation_on_raw(raw_df, method='window',paras={'window_size':320, 'overlap_rate':0.5})
  print "============test do windowing segmentation on raw==============="
  print "=====test head======"
  print seg_raw_df.head()
  print "=====test joint====="
  print seg_raw_df.iloc[315:325]
  print "=====test tail======"
  print seg_raw_df.tail()
  seg_annotation_df = set_segmentation_on_annotation(annotation_df, seg_raw_df)
  print "============test set segmentation on annotation==============="
  print seg_annotation_df
  print "=====test head======"
  print seg_annotation_df.head(10)
  # print "=====test joint====="
  # print seg_annotation_df.iloc[315:325]
  print "=====test tail======"
  print seg_annotation_df.tail(10)
  
  # test bottom up on raw dataset (too slow, run out of time!)
  # seg_df = do_segmentation_on_raw(raw_df, method='bottom_up',paras={'max_error':2, 'column':3})
  # print "============test bottom_up segmentation==============="
  # print "=====test head======"
  # print seg_df.head()
  # print "=====test joint====="
  # print seg_df.iloc[315:325]
  # print "=====test tail======"
  # print seg_df.tail()

if __name__ == "__main__":
  s_info.test_func(unit_test, profile=False)

