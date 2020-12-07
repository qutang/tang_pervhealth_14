#! /usr/bin/python

"""
Module of windowing segmentation algorithms or overlapping windowing algorithms
"""

import numpy as np
import matplotlib.pyplot as pyplot

def standard_windowing_segment(seq, window_size = 200, overlap_rate = 0.5):
  """Standard windowing segmentation algorithm, segment the sequence into pieces
  with equal window size and overlap with overlap rate specified

  Args:
    seq: input sequence to be segmented
    window_size: window size for each segment, default is 200 for SMDT
    overlap_rate: 0-1, overlap rate for windows, default is 0.5

  Return:
    start and end anchors
  """

  # seg_arr = []
  # anchor_arr = []
  start_anchors = []
  end_anchors = []
  step_length = round(window_size*(1-overlap_rate))
  step_length = int(step_length)
  for anchor in range(0, np.shape(seq)[0], step_length):
    if anchor+window_size <= np.shape(seq)[0]:
      # seg_arr.append(seq[anchor:anchor+window_size,:])
      end_anchors.append(anchor+window_size)
    else:
      # seg_arr.append(seq[anchor:,:])
      end_anchors.append(np.shape(seq)[0])
    start_anchors.append(anchor)
  # anchor_arr = np.array([start_anchors,end_anchors])
  return start_anchors, end_anchors

def unit_tests():
  x = np.linspace(0,10,100)
  seq = np.sin(x) + np.random.normal(0,1,np.shape(x))
  input_seq = np.array([x,seq]).T
  pyplot.figure()
  pyplot.plot(input_seq[:,0],input_seq[:,1])
  start_anchors, end_anchors = standard_windowing_segment(input_seq, window_size=10, overlap_rate=0.8)
  print start_anchors
  print end_anchors
  pyplot.vlines(x[start_anchors], 0, 0.1, color='red')

if __name__ == "__main__":
  import cProfile
  pr = cProfile.Profile()
  pr.enable()
  unit_tests()
  pyplot.show()
  pr.disable()
  pr.create_stats()