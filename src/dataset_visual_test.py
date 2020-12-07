#! /usr/bin/python

"""
visual random test about smoking events:
1. plot 5 min sequence with and without labels
2. mark each sequence with potential evidence
3. decide whether smoking or not
"""

import wockets.utils as w_utils
import smdt.annotation as s_annotation
import smdt.raw as s_raw
import smdt.viewer as s_viewer
from datetime import timedelta
import matplotlib.pyplot as pyplot
# import sys

num_session = 4

def main(raw_filenames, annotation_filenames, sessions, subjects, sensors, time_offsets):
  """ Main function for visual test on dataset

  Generate 20 random sequences of 5 min with and without labels on multisensor 
  plot, and save them to files

  Args:
    raw_filenames: raw data filepath strings for each sensor of current session 
    annotation_filenames: annotation filepath strings for each sensor of current session
    sessions: session numbers for each sensor
    subjects: subject strings for each sensor of current session
    sensors: sensor numbers of current session
    time_offsets: list of label time offset for each sensor of current session
  """
  raw_data = w_utils.raw_csv_consolidator(raw_filenames, sessions, subjects, sensors)
  annotation_data = s_annotation.annotation_csv_consolidator(annotation_filenames, time_offsets, sessions, subjects, sensors)
  raw_data = s_raw.preprocess_raw(raw_data, annotation_data, by='sensor')
  # index of plot which needs to be regenerated
  regen = [11,19]
  secs = []
  for count in regen:
    try:
      lbound, rbound = w_utils.generate_random_bounds(raw_data, timedelta(minutes=5))
    # count = 4
    # lbound = w_utils.convert_fromstring("2012-05-03 12:43:16", annotation.annotation_tstr_format)
    # rbound = w_utils.convert_fromstring("2012-05-03 12:48:16", annotation.annotation_tstr_format)
      random_raw = s_raw.select_raw_by_ts(raw_data, lbound, rbound, by='sensor')
      random_annotation = s_annotation.select_annotation_by_ts(annotation_data, lbound, rbound, by='sensor')
      s_viewer.get_multisensor_raw_plot(random_raw, labels=random_annotation, subplots=False)
      true_filename = "../visual_test_data/session"+str(num_session)+"/true/true" + str(count) + '.png'
      pyplot.savefig(true_filename)
      s_viewer.get_multisensor_raw_plot(random_raw, subplots=False)
      test_filename = "../visual_test_data/session"+str(num_session)+"/test/test" + str(count) + '.png'
      pyplot.savefig(test_filename)
      # count += 1
      secs.append(lbound)
    except IndexError:
      continue
  # print the list of time ranges that is randomly generated   
  for s in secs:
    print s

if __name__ == "__main__":
  raw_filenames = ["../raw_dataset/session"+ str(num_session)+"_sensor2.raw.csv", "session"+str(num_session)+"_sensor3.raw.csv"]
  annotation_filenames = ["../raw_dataset/session"+str(num_session)+"_sensor2.annotation.csv","session"+str(num_session)+"_sensor3.annotation.csv"]
  sessions = [num_session, num_session]
  subjects = ["23400"+str(num_session),"23400"+str(num_session)]
  sensors = [2, 3]
  time_offsets = [0,-22,0,20,0,4,-4]
  time_offset = [time_offsets[num_session-1], time_offsets[num_session-1]]
  main(raw_filenames, annotation_filenames, sessions, subjects, sensors, time_offset)