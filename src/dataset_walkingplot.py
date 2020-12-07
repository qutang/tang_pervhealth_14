#! /usr/bin/python

"""
  Script to generate walking and standing plots for sensors to verify the systematic offsets
  With only DAK sensor
"""

import sys
import numpy as np
import matplotlib.pyplot as pyplot
from datetime import timedelta
import smdt.annotation as s_annotation
import smdt.info as s_info
import wockets.utils as w_utils
import smdt.raw as s_raw
import smdt.viewer as s_viewer

def get_walking_or_standing_annotation(annotation_list):
  selected_st = annotation_list[(annotation_list[s_annotation.post_col] != 'sitting') & (annotation_list[s_annotation.post_col].shift(1) != 'walking') & (annotation_list[s_annotation.post_col].shift(1) != 'standing')].index
  selected_et = annotation_list[(annotation_list[s_annotation.post_col] != 'sitting') & (annotation_list[s_annotation.post_col].shift(-1) != 'walking') & (annotation_list[s_annotation.post_col].shift(-1) != 'standing')].index
  return selected_st, selected_et

def main():
  for i in s_info.session_arr:
    i=2
    annotation_file = 'session' + str(i) + '.annotation.csv'
    annotation_list = s_annotation.annotation_csv_importer(s_info.clean_dataset_folder + annotation_file)

    raw_file = 'session' + str(i) + '_DAK.raw.csv'
    raw_ankle_data = w_utils.raw_csv_importer(s_info.clean_dataset_folder + raw_file)

    selected_st, selected_et = get_walking_or_standing_annotation(annotation_list)
    n_subplots = len(selected_st) + 1
    #generate subplot grid
    ncols = np.ceil(np.sqrt(n_subplots))
    nrows = np.ceil(n_subplots/float(ncols))
    c = 1
    consolidate_figure = pyplot.figure(figsize=(20,10))
    for st, et in zip(selected_st, selected_et):
      lbound = annotation_list.ix[st, s_annotation.st_col]
      rbound = annotation_list.ix[et, s_annotation.et_col]
      lbound = lbound - timedelta(seconds=5)
      rbound = rbound + timedelta(seconds=5)
      selected_raw = s_raw.select_raw_by_ts(raw_ankle_data, lbound, rbound)
      selected_annotation = s_annotation.select_annotation_by_ts(annotation_list, lbound, rbound)
      selected_raw = s_viewer._prepare_raw_for_plot(selected_raw)
      print selected_annotation
      ax = consolidate_figure.add_subplot(nrows, ncols, c)
      s_viewer.get_singlesensor_raw_plot(selected_raw, selected_annotation, subplots=False, ax=ax, figsize=None)
      single_figure = pyplot.figure(figsize=(20,10))
      s_viewer.get_singlesensor_raw_plot(selected_raw, selected_annotation, ax=single_figure.gca(), subplots=False)
      figfile = 'session' + str(i) + '_walking_episode' + str(c) + '.rawplot.png'
      single_figure.savefig(s_info.post_figure_folder + figfile)
      pyplot.close(single_figure)
      c+=1
    # add legend
    ax = consolidate_figure.add_subplot(nrows, ncols, c)
    s_viewer.get_legends_plot()
    figfile = 'session' + str(i) + '_walking_episodes.rawplot.png'
    consolidate_figure.savefig(s_info.post_figure_folder + figfile)
    pyplot.close(consolidate_figure)
    sys.exit(1)
if __name__ == "__main__":
  main()