#! /usr/bin/python

"""
Script to process and plot prototype experiment puffs
"""
import sys
import pandas as pd
import smdt.viewer as s_viewer
import smdt.info as s_info
import matplotlib.pyplot as pyplot
from datetime import timedelta, datetime

def main():
  filepath = '../prototype_exp/'
  filenames = ['test normal (2013-07-30)RAW (copy).csv', 'test flipped (2013-07-30)RAW (copy).csv']
  segments = [[4100, 8200],[11000,14500],[16500, 19800], [21800,25100], [27700,31500],[33400,36800],[38500, 42000],[43800, 47200]]
  segments = [[4300, 8000],[16700, 19700], [22000,25000]]
  names = ['sitting puff from below', 'sitting puff from near', 'sitting puff from side', ]
  # names = ['sitting puff from below', 'sitting puff from middle', 'sitting puff from near', 'sitting puff from side', ]
          # 'standing puff from below', 'standing puff from middle', 'standing puff from near', 'standing puff from side']
  sensors = ['flipped','normal']
  dfs = [pd.read_csv(filepath + filename, header=None, names=['rawx','rawy','rawz']) for filename in filenames]
  for df, sensor in zip(dfs, sensors):
    c = 1
    selected_dfs = []
    for segment, name in zip(segments, names):
        temp = df[segment[0]:segment[1]]
        temp[s_info.raw_ts_name] = pd.date_range('2011-01-01', periods=len(temp.index), freq='25L')
        temp = temp.set_index(keys=[s_info.raw_ts_name, ])
        selected_dfs.append(temp)
    s_viewer.get_multistep_view_plot(selected_dfs, labels=[None,]*4, titles=names, sharex=False, figsize=(9.08,2))
      # ax = pyplot.subplot(3, 3, c)
      # .plot(ax=ax, ylim=(-3,3), title=name, colormap='rainbow')
      # pyplot.legend().set_visible(False)
      # c+=1
    # ax = pyplot.subplot(3, 3, 9)
    # dfs[0][:1].plot(ax=ax, colormap='rainbow')
    # pyplot.axis('off')
    # pyplot.savefig(filepath + sensor + '_prototype_puffs.png')
    # pyplot.close(pyplot.gcf())
  pyplot.show()

if __name__ == '__main__':
  main()