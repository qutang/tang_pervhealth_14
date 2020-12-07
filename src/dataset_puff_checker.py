#! /usr/bin/python

"""
Script to output all multi sensor figures for all the segments with duration of 30s,
puff will be put in the center
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as pyplot
import wockets.utils as w_utils
import smdt.raw as s_raw
import smdt.viewer as s_viewer
import smdt.annotation as s_annotation
import smdt.info as s_info

def generate_puff_correction_csv():
  for i in s_info.session_arr:
    i=7
    annotation_file = 'session' + str(i) + '.annotation.csv'
    annotation_data = s_annotation.annotation_csv_importer(s_info.clean_dataset_folder + annotation_file)
    # get all puff annotations
    puff_st_markers = (annotation_data[s_annotation.puff_col] != 'no-puff') & ((annotation_data[s_annotation.puff_col].shift(1) == 'no-puff') | pd.isnull(annotation_data[s_annotation.puff_col].shift(-1)))
    puff_et_markers = (annotation_data[s_annotation.puff_col] != 'no-puff') & ((annotation_data[s_annotation.puff_col].shift(-1) == 'no-puff') | pd.isnull(annotation_data[s_annotation.puff_col].shift(-1)))
    c = 0
    puff_correction_item = {'STARTTIME':[],'ENDTIME':[],'prototypical':[],'offset-left':[],'offset-right':[],'potential error':[],'link':[],'note':[]}
    for st, et in zip(annotation_data[puff_st_markers][s_annotation.st_col], annotation_data[puff_et_markers][s_annotation.et_col]):
      puff_correction_item['STARTTIME'].append(st)
      puff_correction_item['ENDTIME'].append(et)
      puff_correction_item['prototypical'].append(1)
      puff_correction_item['offset-left'].append(0)
      puff_correction_item['offset-right'].append(0)
      puff_correction_item['potential error'].append(0)
      puff_correction_item['note'].append('')
      if annotation_data[annotation_data[s_annotation.st_col] == st][s_annotation.puff_col] == 'left-puff':
        side = 'L'
      else:
        side = 'R'
      fname = s_info.puff_figure_folder + 'session' + str(i) + '_puff' + str(c) + '_' + side + '.rawplot.png'
      puff_correction_item['link'].append(os.path.abspath(fname))
      c += 1
    #save puff correction to csv for each session
    tosave = pd.DataFrame(puff_correction_item, columns=['STARTTIME','ENDTIME','link','offset-left','offset-right','prototypical','potential error','note'])
    csvname = s_info.clean_dataset_folder + 'session' + str(i) + '_puff.correction.csv'
    tosave.to_csv(csvname)
    sys.exit(1)

def main():
  # generate legend plot
  fhandle = s_viewer.get_legends_plot()
  pyplot.show()
  # fname = s_info.puff_figure_folder + 'legend.png'
  # fhandle.savefig(fname)
  # pyplot.close(fhandle)
  sys.exit(1)
  for i in s_info.session_arr:
    # i=7
    # read in raw and annotation
    annotation_file = 'session' + str(i) + '.annotation.csv'
    raw_files = ['session' + str(i) + '_' + code + '.raw.csv' for code in s_info.sensor_codes]
    raw_datas = [w_utils.raw_csv_importer(s_info.clean_dataset_folder + raw_file) for raw_file in raw_files]
    consolidate_raw = w_utils.raw_data_consolidator(raw_datas, sessions=[i,]*len(raw_datas), sensors=s_info.sensor_codes)
    annotation_data = s_annotation.annotation_csv_importer(s_info.puff_corrected_folder + annotation_file) # use puff corrected annotation
    # annotation_data = s_annotation.annotation_csv_importer(s_info.clean_dataset_folder + annotation_file) # use clean annotation without puff correction
    
    # get all puff annotations
    puff_st_markers = (annotation_data[s_annotation.puff_col] != 'no-puff') & ((annotation_data[s_annotation.puff_col].shift(1) == 'no-puff') | pd.isnull(annotation_data[s_annotation.puff_col].shift(-1)))
    puff_et_markers = (annotation_data[s_annotation.puff_col] != 'no-puff') & ((annotation_data[s_annotation.puff_col].shift(-1) == 'no-puff') | pd.isnull(annotation_data[s_annotation.puff_col].shift(-1)))
    c = 0
    for st, et in zip(annotation_data[puff_st_markers][s_annotation.st_col], annotation_data[puff_et_markers][s_annotation.et_col]):
      # get raw and annotation for current puff
      middle_time = st + (et - st)/2
      from datetime import timedelta
      lbound = middle_time - timedelta(seconds=15)
      rbound = middle_time + timedelta(seconds=15)
      selected_puff_raw = s_raw.select_raw_by_ts(consolidate_raw, lbound=lbound, rbound=rbound, by='sensor')
      selected_annotation = s_annotation.select_annotation_by_ts(annotation_data, lbound=lbound, rbound=rbound)

      selected_annotations = [selected_annotation.copy() for s in s_info.sensor_codes]
      consolidate_selected_annotation = s_annotation.annotation_data_consolidator(selected_annotations, sensors=s_info.sensor_codes)
      
      # plot multisensor figure and save them
      if annotation_data[annotation_data[s_annotation.st_col] == st][s_annotation.puff_col] == 'left-puff':
        side = 'L'
      else:
        side = 'R'
      fig = s_viewer.get_multisensor_raw_plot(selected_puff_raw, consolidate_selected_annotation, subplots=False)
      fname = s_info.puff_figure_folder + 'session' + str(i) + '_puff' + str(c) + '_' + side + '.rawplot.png'
      fig.savefig(fname)
      pyplot.close(fig)
      print fname + " written"
      c += 1
    # sys.exit(1)

if __name__ == '__main__':
  main()
  # generate_puff_correction_csv()