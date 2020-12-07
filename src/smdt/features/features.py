"""
Module contains all feature computation functions, used for smoking dataset
"""

import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.stats
import sys

def get_linear_regression(seq, r=None):
  A = np.vstack([seq.index, np.ones(len(seq))]).T
  A = np.array(A,dtype=np.float64)
  y = seq.values
  result = np.linalg.lstsq(A,y)
  slopes, intercepts = result[0]
  residuals = result[1][0]
  if r == 'slope':
    return slopes
  elif r == 'intercept':
    return intercepts
  elif r == 'residual':
    return residuals
  else:
    return slopes, intercepts, residuals

def zcr(seq):
  temp = np.sign(seq)
  # temp = temp[1:] - temp[:-1]
  temp = np.diff(temp)
  crossings = np.where(temp)[0]
  zcc = len(crossings)
  zcr = float(zcc)/len(seq)
  return zcr

def f_percentile(seq, paras):
  """compute percentil of a seq

  paras contains only one key which is "q", the range should be between 0 to 100
  0 indicates min and 100 indicates max

  Args:
    seq: input array_like sequence
    paras: paras array, in this case should be "q" and "axis"
  """
  seq = np.array(seq, dtype=np.float64)
  q = paras["q"]
  axis = 0
  if paras.has_key("axis"):
    axis = paras["axis"]
  return np.percentile(seq, q=q, axis=axis)

def f_snr(seq):
  """compute signal to noise rate of a seq

  Args:
    seq: input array_like sequence
    paras: paras array, in this case should be "axis"
  """
  seq = np.array(seq, dtype=np.float64)
  result = np.mean(seq)/float(np.std(seq))
  if np.isinf(result):
    print "marker"
    result = 0
  return result

def f_pppeakamplitude(seq, paras):
  """compute peak-peak amplitude of a seq

  paras contains only one key which is "q", the range should be between 0 to 100
  0 indicates min and 100 indicates max

  Args:
    seq: input array_like sequence
    paras: paras array, in this case should be "q" and "axis"
  """
  seq = np.array(seq, dtype=np.float64)
  q = paras["q"]
  axis = 0
  if paras.has_key("axis"):
    axis = paras["axis"]
  result = np.abs(np.percentile(seq, q=q, axis=axis) - np.percentile(seq, q=100-q, axis=axis))
  return result

def f_peakrate(seq, paras):
  """compute peak rate of a seq

  compute positive peak rate value of a sequence(vertical direction if dimension is 2)

  Args:
    seq: input array
    paras: "sigma": the LoG filter parameter, "q": q percentage below of the max peak will be kept
  """
  seq = np.array(seq, dtype=np.float64)
  factor = paras['q']
  sigma = paras['sigma']
  seq = sigma*scipy.ndimage.filters.gaussian_filter1d(seq, axis=0, sigma=sigma, order=2, mode='reflect')
  # pyplot.plot(seq[:,0],seq[:,1])
  if len(np.shape(seq)) == 1:
    # peakind = scipy.signal.find_peaks_cwt(seq, np.arange(3,10), min_length=4.0)
    peakind = scipy.signal.argrelmax(seq, order=3, mode='clip')
    if len(peakind[0]) == 0:
      peakrate = 0
    else:
      peakvalue = seq[peakind]
      peakmax = np.max(peakvalue)
      peakvalue = peakvalue[peakvalue >= peakmax*factor]
      peakrate = len(peakvalue)/float(len(seq))
    return peakrate
  else:
    peakrate_arr = []
    for col in range(0, np.shape(seq)[1]):
      peakind = scipy.signal.find_peaks_cwt(seq[:,col], np.arange(3,10), min_length=4.0)
      if len(peakind) == 0:
        peakrate = 0
      else:
        peakvalue = seq[peakind,col]
        peakmax = np.max(peakvalue)
        peakvalue = peakvalue[peakvalue >= peakmax*factor]
        peakrate = len(peakvalue)/float(len(seq))
      peakrate_arr.append(peakrate)
      return peakrate_arr

def f_correlation(seq, paras):
  """compute correlation of a seq with another

  compute correlation value between seq and another seq

  Args:
    seq: input seq must be a series
    paras: "another": contains another sequence, "type": type of correlation coefficient
  """
  another = paras['another']
  another = another.ix[seq.index]
  method = paras['type']
  result = seq.corr(another, method=method)
  if np.isnan(result):
    return 1
  else:
    return result

def f_rms(seq):
  """compute root mean square of a seq

  compute root mean square

  Args:
    seq: input seq
  """
  seq = np.array(seq, dtype=np.float64)
  return np.sqrt(np.sum(seq**2, axis=0))

def f_slope(seq):
  """compute slope of linear regression of a seq

  Args:
    seq: input seq must be a series
  """
  return get_linear_regression(seq, r='slope')

def f_mse(seq):
  """compute mean square error of linear regression of a seq

  Args:
    seq: input seq must be a series
  """
  return get_linear_regression(seq, r='residual')

def f_rsquared(seq):
  """compute R squared(coefficient of determination) of linear regression of a seq

  Args:
    seq: input seq must be a series
  """
  res = get_linear_regression(seq, r='residual')
  var = seq.var()
  if var == 0 :
    return 1
  else:
    return 1 - res/(len(seq)*var)

def f_axescrossrate(seq, paras):
  """compute the axes crossing rate of a seq

  the axes crossing rate is actually the zero crossing rate of the substraction of two columns

  Args:
    seq: input seq must be a series
    paras: should contain at least "another"
  """
  another = paras['another']
  another = another.ix[seq.index]
  q = paras['q']
  subtract = seq - another
  # filter out small changes
  subtract[np.abs(subtract) <= q*np.max(subtract)] = 0
  # filter out zeros
  subtract = subtract[subtract != 0]
  return zcr(subtract)

def f_std(seq):
  seq = np.array(seq, dtype=np.float64)
  return np.std(seq, axis=0)

def f_mean(seq):
  seq = np.array(seq, dtype=np.float64)
  return np.mean(seq, axis=0)

def f_skew(seq):
  seq = np.array(seq, dtype=np.float64)
  return scipy.stats.skew(seq, axis=0)

def f_kurtosis(seq):
  seq = np.array(seq, dtype=np.float64)
  return scipy.stats.kurtosis(seq, axis=0)