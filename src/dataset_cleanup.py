#! /usr/bin/python

"""
  Script for dataset clean up:
  will do:
    fix the annotation file
    not decided:synchronize annotation data
    trim raw data to annotation range
    scale raw data to -4 to 4 range
    save both raw dataset and annotation to csv files in clean dataset folder
"""

from datetime import timedelta
import wockets.utils as w_utils
import smdt.annotation as s_annotation
import smdt.raw as s_raw
import smdt.info as s_info
import sys

def main():
  # batch process for each session
  for i in s_info.session_arr:
    i=7
    # process annotation files
    annotation_file = 'session' + str(i) + '.annotation.csv'
    annotation_list = s_annotation.annotation_csv_importer(s_info.raw_dataset_folder + annotation_file)
    annotation_list = s_annotation.fix_annotation(annotation_list, time_offset=s_info.sys_offsets[i-1])
    annotation_list.to_csv(s_info.clean_dataset_folder + annotation_file, index=False)
    print annotation_file + ' written'
    # process raw data files
    for j in s_info.sensor_codes:
      data_file = 'session' + str(i) + '_' + j + '.raw.csv'
      data_set = w_utils.raw_csv_importer(s_info.raw_dataset_folder + data_file)
      data_set = s_raw.preprocess_raw(data_set, annotation_list, grace_period=timedelta(seconds=0))
      w_utils.raw_csv_exporter(data_set, s_info.clean_dataset_folder + data_file)
      print data_file + ' written'
    sys.exit(1)
  return

if __name__ == "__main__":
  main()