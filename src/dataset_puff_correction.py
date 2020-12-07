#/ usr/bin/python

"""
  Script to correct puffs in clean dataset and save them into new files
"""

import sys
from datetime import timedelta
import pandas as pd
import smdt.annotation as s_annotation
import smdt.info as s_info
import smdt.raw as s_raw
import wockets.utils as w_utils

def main():
  num_session = 7
  # batch process for each session
  for i in range(1,num_session+1):
    # process annotation files
    annotation_file = '/session' + str(i) + '.annotation.csv'
    puff_correction_file = '/session' + str(i) + '_puff.correction.csv'
    annotation_list = s_annotation.annotation_csv_importer(s_info.clean_dataset_folder + annotation_file)
    try:
      correction_list = pd.read_csv(s_info.clean_dataset_folder + puff_correction_file, parse_dates=['STARTTIME', 'ENDTIME'], index_col=0)
    except:
      print puff_correction_file + ' not processed'
      correction_list = None
    annotation_list = s_annotation.correct_puffs_and_add_prototypical_marks(annotation_list, correction_list)
    annotation_list.to_csv(s_info.puff_corrected_folder + annotation_file, index=False)
    print annotation_file + ' written'
    # process raw data files
    for j in s_info.sensor_codes:
      data_file = '/session' + str(i) + '_' + j + '.raw.csv'
      data_set = w_utils.raw_csv_importer(s_info.raw_dataset_folder + data_file)
      data_set = s_raw.preprocess_raw(data_set, annotation_list, grace_period=timedelta(seconds=0))
      w_utils.raw_csv_exporter(data_set, s_info.puff_corrected_folder + data_file)
      print data_file + ' written'
    # sys.exit(1)
  return

if __name__ == '__main__':
  main()