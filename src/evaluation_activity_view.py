#! /usr/bin/python

"""
Activity view plot for smoking detection result evaluation and visualization
"""

import wockets.utils as w_utils
import smdt.annotation as s_annotation
import smdt.raw as s_raw
import smdt.viewer as s_viewer
from datetime import timedelta
import matplotlib.pyplot as pyplot
# import sys
import pandas as pd

def generate_activity_views(raw_filenames, annotation_filenames, prediction_filenames, visualtest_filenames, sessions, time_offsets):
  """Generate activity views for each provided dataset

  Generate synthetic activity view plot in time series with predicted and true
  puffing, smoking, visual test results and labels. Save the plots to files in
  evaluation folder

  Args:
    raw_filenames: list of raw data filepath strings for each data session
    annotation_filenames: list of annotation data filepath strings for each data session
    prediction_filenames: list of model prediction data filepath strings for each data session
    visualtest_filenames: list of visual test result filepath strings for each data session
    sessions: list of session numbers for each data session
    time_offsets: list of label time offsets for each data session 
  """
  for rawfile, annotationfile, predictionfile, visualtestfile,session in zip(raw_filenames, annotation_filenames, prediction_filenames, visualtest_filenames,sessions):
    time_offset = time_offsets[session-1]
    raw_data = w_utils.raw_csv_consolidator([rawfile,], sessions=[session,])
    annotation_data = s_annotation.annotation_csv_consolidator([annotationfile,],sessions=[session,], time_offsets=[time_offset,])
    raw_data = s_raw.preprocess_raw(raw_data, annotation_data, grace_period = timedelta(seconds=0)) # single session, so we don't need group by
    prediction_data = pd.read_csv(predictionfile, names=['prob','puffs'])
    visualtest_data = pd.read_csv(visualtestfile, header=0)
    s_viewer.get_activity_view_plot(raw_data, annotation_data, prediction_data=prediction_data, visualtest_data=visualtest_data, title='session'+str(session))
    filename = "../evaluation_result/activity_views/activity_view_session"+str(session)+'.png'
    # pyplot.show()
    pyplot.savefig(filename)
    pyplot.clf()

if __name__ == "__main__":
  raw_filenames = ["../raw_dataset/session"+ str(n)+"_sensor2.raw.csv" for n in range(1,8,1)]
  annotation_filenames = ["../raw_dataset/session"+str(n)+"_sensor2.annotation.csv" for n in range(1,8,1)]
  prediction_filenames = ["../visual_test_data/session"+str(n)+"/predict/smoking_prediction.csv" for n in range(1,8,1)]
  visualtest_filenames = ["../visual_test_data/session"+str(n)+"/summary.test.csv" for n in range(1,8,1)]
  sessions = range(1,8,1)
  time_offsets = [0,-22,0,20,0,4,-4]
  generate_activity_views(raw_filenames, annotation_filenames, prediction_filenames, visualtest_filenames, sessions, time_offsets)