#! /usr/bin/python

"""
program used to analyze dataset
"""

from datetime import timedelta
import matplotlib.pyplot as pyplot
# import sys
import pandas as pd
import wockets.utils as w_utils
import smdt.annotation as s_annotation
import smdt.raw as s_raw
import smdt.viewer as s_viewer

timeformat = '%m/%d/%Y %H:%M:%S'

def generate_puffing_example_plots(folder, example_file, time_offsets):
  """ Generate examples of puffing plots

  Generate examples of puffing plots with subplots of different preprocessing
  skill and save them as files. Call pyplot.show() afterwards to show plots in time

  Args:
    folder: path string to save the generated examples
    example_file: csv file path string which contains info of puffing examples
    time_offsets: list of label offset for each session
  """

  examples = pd.read_csv(example_file, header=0)
  for idx, row in examples.iterrows():
    session = row[0]
    if isinstance(row[1], int):
      sensors = [row[1],]
    else:
      sensors = row[1]
    use = row[4]
    case = row[5]
    lbound = w_utils.convert_fromstring(row[2], timeformat)
    rbound = w_utils.convert_fromstring(row[3], timeformat)
    rawfiles = ["../raw_dataset/session"+ str(session)+"_sensor"+ str(sensor) +".raw.csv" for sensor in sensors]
    annotationfiles = ["../raw_dataset/session"+str(session)+"_sensor" + str(sensor) + ".annotation.csv" for sensor in sensors]
    sessions = [str(session) for sensor in sensors]
    time_offset = [time_offsets[session-1] for sensor in sensors]
    raw_data = w_utils.raw_csv_consolidator(rawfiles, sessions=sessions, sensors=sensors)
    annotation_data = s_annotation.annotation_csv_consolidator(annotationfiles,sessions=sessions, time_offsets=time_offset, sensors=sensors)
    raw_data = s_raw.preprocess_raw(raw_data, annotation_data, grace_period = timedelta(seconds=0), by='sensor')
    raw_data = s_raw.select_raw_by_ts(raw_data, lbound, rbound, by='sensor')
    annotation_data = s_annotation.annotation_select_by_ts(annotation_data, lbound, rbound, by='sensor')
    orientation_values_name, orientation_data = s_raw.transform_raw(raw_data, transform_type='orientation') # get orientation measurement
    post_distance_values_name, post_distance_data = s_raw.transform_raw(raw_data, transform_type='post-distance') # get post distance measurement
    datas = [raw_data, orientation_data, post_distance_data]
    annotations = [annotation_data, annotation_data, annotation_data]
    fig_to_save = s_viewer.get_multistep_view_plot(datas, annotations, titles=['raw','orientation','post distance'], subplots=True, sharex=True, appear='gray')
    fig_to_save.set_size_inches(30,10)
    filename = str(idx) + "_session" + str(session) + "_sensor" + str(sensor) + "_" + use + "_" + case + ".png"
    fig_to_save.savefig(folder + filename)
    pyplot.clf()
    # pyplot.show()

def generate_random_plot_given_labels(session, labels, time_offsets):
  """ Generate random raw data plot including given labels

  Generate random raw data plot with given labels on it, multisensor plot used,
  call pyplot.show() to show plot in time. Plot will not be saved automatically

  Args:
    session: session number which is used to choose session data files
    labels: string list contains labels we want to be involved in plot
    time_offsets: list of label time offsets for each session
  """

  sensors = [3]
  rawfiles = ["../raw_dataset/session"+ str(session)+"_sensor"+ str(sensor) +".raw.csv" for sensor in sensors]
  annotationfiles = ["../raw_dataset/session"+str(session)+"_sensor" + str(sensor) + ".annotation.csv" for sensor in sensors]
  sessions = [str(session) for sensor in sensors]
  time_offset = [time_offsets[session-1] for sensor in sensors]
  raw_data = w_utils.raw_csv_consolidator(rawfiles, sessions=sessions, sensors=sensors)
  annotation_data = s_annotation.annotation_csv_consolidator(annotationfiles,sessions=sessions, time_offsets=time_offset, sensors=sensors)
  annotation_single = s_annotation.annotation_csv_consolidator([annotationfiles[0],],sessions=[sessions[0],], time_offsets=[time_offset[0],])
  raw_data = s_raw.preprocess_raw(raw_data, annotation_data, grace_period = timedelta(seconds=0), by='sensor')
  lbound, rbound = s_annotation.generate_bounds_by_labels(annotation_single, duration=timedelta(minutes=1, seconds=30), labels=labels)
  raw_data = s_raw.select_raw_by_ts(raw_data, lbound, rbound, by='sensor')
  annotation_data = s_annotation.select_annotation_by_ts(annotation_data, lbound, rbound, by='sensor')
  # s_viewer.get_multisensor_raw_plot(raw_data, labels=annotation_data, subplots=False)

  # multi steps
  post_value_names, posture_data = s_raw.transform_raw(raw_data, transform_type='post-distance')
  raw_datas = [raw_data, posture_data]
  labels = [annotation_data, annotation_data]
  titles = ['raw','post-distance']
  s_viewer.get_multistep_view_plot(raw_datas, labels, titles, subplots=False)
  pyplot.show()



if __name__ == "__main__":
  folder = '../puff_examples/set1/'
  example_file = '../puff_examples/examples_set1.csv'
  time_offsets = [0,-22,0,20,0,4,-4]
  generate_random_plot_given_labels(1,['walking',], time_offsets)
  # generate_puffing_example_plots(folder, example_file, time_offsets)