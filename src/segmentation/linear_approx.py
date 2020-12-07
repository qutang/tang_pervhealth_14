#! /usr/bin/python

"""
Module of segmentation algorithm used to segmentate time series signal by piecewise
linear approximation

You can find the introduction of SWAB algorithm by searching the technical paper
called "Segmenting Time Series: A Survey and Novel Approach" on Google
"""
import sys
import numpy as np
import matplotlib.pyplot as pyplot

def _calculate_error(T_seq, method='regression', column=1):
  """ Given time series sequence, calculate the linear segmentation approximation
  error, column 0 will be used as x by default

  Args:
    T_seq: numpy array of time series sequence, two columns one for time and one
    for value
    method: string to specify type of approximation approach, either 'regression'
    or 'interpolation'
    column: column of input sequence to be used as y
  Return:
    approximation error: square root error
  """
  
  if method == 'regression':
    A = np.vstack([T_seq[:,0], np.ones(len(T_seq[:,0]))]).T
    # print T_seq[:,0]
    A = np.array(A,dtype='float')
    y = T_seq[:,column]
    slope, intercept = np.linalg.lstsq(A,y)[0]
    error = np.sqrt(np.sum(((A[:,0]*slope+intercept - y)**2)))
  elif method == 'interpolation':
    return 0
  return error

def sliding_window_segment(seq, max_error, step_length=1, method='regression', column=1, online=False):
  """Given time series sequence, apply sliding_window algorithm to segment the
  sequence. Criteria of segmentation is the approximation error of piecewise linear
  regression

  Args:
    seq: input sequence to be segmented
    max_error: max error to decide accept or reject segment, > error reject and 
    generate new segment, < error accept and add to current segment
    step_length: sliding step length of points to be decided
    method: string to decide whether to use 'regression' or 'interpolation' to
    approximate error
    column: column to be used as y to calculate error
    online: if True, return the first segment at the time get it
  Returns:
    start and end anchors
  """

  anchor = 0
  # seg_arr = []
  start_anchors = []
  end_anchors = []
  # anchor_arr = []
  while anchor < np.shape(seq)[0] :
    right_end = 2
    start_anchors.append(anchor)
    start_anchor = anchor
    while anchor+right_end <= np.shape(seq)[0] and _calculate_error(seq[anchor:anchor+right_end,:], method=method, column=column) < max_error:
      right_end += step_length
    end_anchor = anchor + right_end -1
    if online:
      return start_anchor, end_anchor
    else:
      # seg_arr.append(seq[anchor:anchor+right_end-1,:])
      anchor += right_end-1
      end_anchors.append(anchor)
  # anchor_arr = np.array([start_anchors, end_anchors])
  return start_anchors, end_anchors

def bottom_up_merge(seq, max_error, method='regression', column=1):
  """Given a time series sequence, divide them into n/2 pieces and merge them to
  as large segment as possible so that all segments' approximation error smaller
  than max error

  Args:
    seq: input sequence to be segmented
    max_error: max error to decide whether to merge two segments, > error stop
    merging, < error continue merging
    method: string to decide whether to use 'regression' or 'interpolation' to
    approximate error
  Return:
    start and end anchors
  """

  seg_arr = []
  cost_arr = []
  start_anchors = [0]
  end_anchors = []
  for i in range(0,np.shape(seq)[0]-1,2):
    seg_arr.append(seq[i:i+2,:])
  for i in range(0,len(seg_arr)-1):
    cost_arr.append(_calculate_error(np.vstack((seg_arr[i],seg_arr[i+1])), method=method, column=column))
  while len(seg_arr) > 1 and np.amin(cost_arr) < max_error:
    index = cost_arr.index(np.amin(cost_arr))
    seg_arr[index] = np.vstack((seg_arr[index], seg_arr[index+1]))
    del seg_arr[index+1]
    del cost_arr[index]  
    if index + 1 < len(seg_arr):
      cost_arr[index] = _calculate_error(np.vstack((seg_arr[index],seg_arr[index+1])), method=method, column=column)
    if index > 0:
      cost_arr[index-1] = _calculate_error(np.vstack((seg_arr[index-1],seg_arr[index])), method=method, column=column)
  for seg in seg_arr:
    start_anchors.append(start_anchors[-1]+len(seg))
    end_anchors.append(start_anchors[-1])
  start_anchors = start_anchors[:-1]
  # anchor_arr = np.array([start_anchors[:-1], end_anchors])
  return start_anchors, end_anchors

def SWAB_segment(seq, max_error, buffer_size, method='regression', column=1):
  """Given a time series sequence, use SWAB algorithm to segment sequence

  SWAB algorithm: set buffer to be a certain length, use sliding window algorithm
  to read in new segment, and use bottom up algorithm to segment current buffer,
  and add leftmost segment to the segment array

  Args:
    seq: input sequence to be segmented
    max_error: max error to be used in sliding window and bottom up algorithm
    buffer_size: buffer size for each loop
    method: string to decide whether to use 'regression' or 'interpolation' to
    approximate error
    column: column of input seqence used to calculate error
  Return:
    start and end anchors
  """
  buffer_seq = seq[0:buffer_size,:]
  # seg_arr = []
  start_anchors = [0]
  end_anchors = []
  while len(buffer_seq) > 1 and start_anchors[-1] < np.shape(seq)[0]:
    st, et = bottom_up_merge(buffer_seq, max_error=max_error, method=method, column=column)
    # seg_arr.append(buffer_seg_arr[0])
    start_anchors.append(start_anchors[-1] + et[0] - st[0])
    end_anchors.append(start_anchors[-1])
    if len(seq[end_anchors[-1]:,:]) != 0:
      online_st, online_et = sliding_window_segment(seq[end_anchors[-1]:,:], max_error=max_error, column=column, online=True)
      print online_st, online_et
      new_seq = seq[end_anchors[-1]:end_anchors[-1]+online_et,:]
      buffer_seq = new_seq[0:buffer_size,:]
  # anchor_arr = np.array([start_anchors[:-1], end_anchors])
  start_anchors = start_anchors[:-1]
  return start_anchors, end_anchors

def unit_tests():

  # test _calculate
  seq = np.array([range(0,10),range(10,20)]).T
  print seq
  err = _calculate_error(seq, method='regression')
  print "test _calcuate_error"
  print "test passed:", err < 10**-10
  # test sliding_window_segment
  x = np.linspace(0,10,100)
  seq = np.sin(x) + np.random.normal(0,1,np.shape(x))
  input_seq = np.array([x,seq]).T
  pyplot.plot(input_seq[:,0],input_seq[:,1])
  start_anchors, end_anchors = sliding_window_segment(input_seq, max_error=2)
  print "=============Test sliding_window_segmentation ============"
  print "=============Start Anchors======================"
  print start_anchors
  print "=============End Anchors======================"
  print end_anchors
  pyplot.vlines(x[start_anchors], 0, 1, color='red')
  # test bottom up segment
  x = np.linspace(0,10,100)
  seq = np.sin(x) + np.random.normal(0,1,np.shape(x))
  input_seq = np.array([x,seq]).T
  pyplot.figure()
  pyplot.plot(input_seq[:,0],input_seq[:,1])
  start_anchors, end_anchors = bottom_up_merge(input_seq, max_error=2)
  print "=============Test bottom_up_merge ============"
  print "=============Start Anchors======================"
  print start_anchors
  print "=============End Anchors======================"
  print end_anchors
  pyplot.vlines(x[start_anchors], 0, 1, color='red')
  # test SWAB segment algorithm
  x = np.linspace(0,10,100)
  seq = np.sin(x) + np.random.normal(0,1,np.shape(x))
  input_seq = np.array([x,seq]).T
  pyplot.figure()
  pyplot.plot(input_seq[:,0],input_seq[:,1])
  start_anchors, end_anchors = SWAB_segment(input_seq, max_error=2, buffer_size=20)
  print "=============Test SWAB segmentation ============"
  print "=============Start Anchors======================"
  print start_anchors
  print "=============End Anchors======================"
  print end_anchors
  pyplot.vlines(x[start_anchors], 0, 1, color='red')
  pyplot.show()
  # sys.exit(1)
if __name__ == "__main__":
  import cProfile
  pr = cProfile.Profile()
  pr.enable()
  unit_tests()
  pr.disable()
  pr.create_stats()