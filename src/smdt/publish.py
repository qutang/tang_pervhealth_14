""" Generate all publishable figure results
"""

import sys
sys.path.append("../")
import pandas as pd

import info as s_info
import loader as s_loader
import viewer as s_viewer
import matplotlib.pyplot as pyplot
import numpy as np
import matplotlib
import annotation as s_annotation
import raw as s_raw
import wockets.utils as w_utils
from datetime import timedelta

def generate_smoking_detection_plot():
  """Generate time series alignment for each session along with their annotation,
    true puffs and predicted puffs"""

  sessions = [1,3,4,5,6,7]
  sensors = ['DW','NDW']
  window_size = 1000
  prediction_df = s_loader.load_smokeprediction_df(session='all', window_size=window_size, sensor='BFW', validation='kfold')
  fig, axes = pyplot.subplots(len(sessions), 1, figsize=(5,10), sharex=False)
  fig.tight_layout()
  c = 0
  for session in sessions:
   # get prdiction df for this session
   this_prediction_df = prediction_df[prediction_df[s_info.session_col] == session]
   this_prediction_df= this_prediction_df.reset_index(drop=False)
   axes[c].plot(this_prediction_df.index, this_prediction_df[s_info.smokepredprob_col]/this_prediction_df[s_info.smokepredprob_col].max()+2, 'b')
   this_prediction_df[s_info.smokeproportion_col].fillna(0).plot(ax=axes[c],color='c')

   temp = this_prediction_df[s_info.classname_col] == 'puff'
   temp = this_prediction_df[temp][s_info.classname_col] == 'puff'
   axes[c].plot(temp.index, temp - 0.1, '.c')
   temp = this_prediction_df[s_info.predictionname_col] == 'puff'
   temp = this_prediction_df[temp][s_info.predictionname_col] == 'puff'
   axes[c].plot(temp.index, temp + 1.9, '.b')

   # activities
   temp = this_prediction_df[s_info.drinkproportion_col] >= 0.25
   temp = this_prediction_df[temp][s_info.drinkproportion_col]
   axes[c].plot(temp.index, temp+4, 'o', color=s_viewer.jet_cm_arr.to_rgba(0))

   temp = this_prediction_df[s_info.eatproportion_col] >= 0.25
   temp = this_prediction_df[temp][s_info.eatproportion_col]
   axes[c].plot(temp.index, temp+4, 'v', color=s_viewer.jet_cm_arr.to_rgba(1))

   temp = this_prediction_df[s_info.phoneproportion_col] >= 0.3
   temp = this_prediction_df[temp][s_info.phoneproportion_col]
   axes[c].plot(temp.index, temp+4, 's', color=s_viewer.jet_cm_arr.to_rgba(2))

   temp = this_prediction_df[s_info.computerproportion_col] >= 0.3
   temp = this_prediction_df[temp][s_info.computerproportion_col]
   axes[c].plot(temp.index, temp+4, '^', color=s_viewer.jet_cm_arr.to_rgba(3))

   temp = this_prediction_df[s_info.talkproportion_col] >= 0.3
   temp = this_prediction_df[temp][s_info.talkproportion_col]
   axes[c].plot(temp.index, temp+4, 'D', color=s_viewer.jet_cm_arr.to_rgba(5))

   axes[c].set_ylim((-1, 6))
   axes[c].set_xlim((this_prediction_df.index[0], this_prediction_df.index[-1]))
   axes[c].set_yticks([0, 2, 4])
   # axes[c].set_yticklabels(['puff/smoking ground truth', 'puff/smoking prediction', 'other activities'])

   axes[c].set_title("session " + str(session))
   axes[c].set_xticks(np.linspace(this_prediction_df.index[0], this_prediction_df.index[-1], 8))
   xticks = axes[c].get_xticks()
   xticks = xticks*36./60.
   xticks = np.array(xticks, dtype='int')
   axes[c].set_xticklabels(xticks)
   c += 1

  # fig.text(0.5, 0.02, 'time (minutes)', ha='center', va='center')

  fig.subplots_adjust(left=0.05, bottom=0.06)
  pyplot.show()



def generate_prototypical_puff_and_sequence(example='eatsmoking'):
    if example == 'prototypical':
        example_info = {
            'session': [3, 3],
            'STARTTIME': ["12:26:07", "12:25:45"],
            'ENDTIME': ["12:26:17", "12:27:45"],
            'dt': 'clean',
            'sensors': [['NDW', ], ['NDW', ]],
            'titles': ['prototpycail puff', 'prototypical puff sequence']
        }
    elif example == 'non-prototypical':
        example_info = {
            'session': [3, 3],
            'STARTTIME': ["12:10:55", "12:09:25"],
            'ENDTIME': ["12:11:25", "12:11:25"],
            'dt': 'clean',
            'sensors': [['NDW', ], ['NDW', ], ],
            'titles': ['non-prototpycail puff', 'non-prototypical puff sequence']
        }
    elif example == 'body_variety':
        example_info = {
            'session': [1, 1, 3],
            'STARTTIME': ["11:35:27", "12:54:12", "12:38:32"],
            'ENDTIME': ["11:37:27", "12:56:12", "12:40:32"],
            'dt': 'clean',
            'sensors': [['NDW', ], ['NDW', ], ['NDW', ]],
            'titles': ['puff while sitting', 'puff while standing', 'puff while body transition']
        }
    elif example == 'hand_variety':
        example_info = {
            'session': [3, 3, 3],
            'STARTTIME': ["12:11:51", "12:25:00", "12:33:38"],
            'ENDTIME': ["12:13:51",  "12:27:00", "12:35:38"],
            'dt': 'clean',
            'sensors': [['NDW', ], ['NDW',], ['NDW', ]],
            'titles': ['puff while talking',  'puff while using phone', 'puff while eating and using computer']
        }
    elif example == 'eatsmoking':
        example_info = {
            'session': [1, ],
            'STARTTIME': ["11:52:50", ],
            'ENDTIME': ["11:54:50", ],
            'dt': 'clean',
            'sensors': [['NDW', 'DW'], ],
            'titles': ['smoking while eating and drinking',]
        }

    selected_raw_dfs = []
    selected_annotation_dfs = []
    for i in range(0, len(example_info['session'])):
        lbound = w_utils.convert_fromstring(s_info.session_dates[s_info.session_arr.index(example_info['session'][i])] + ' ' + example_info['STARTTIME'][i], s_info.tstr_format)
        rbound = w_utils.convert_fromstring(s_info.session_dates[s_info.session_arr.index(example_info['session'][i])] + ' ' + example_info['ENDTIME'][i], s_info.tstr_format)
        raw_df, annotation_df = s_loader.load_smoking_df(session=example_info['session'][i], sensors=example_info['sensors'][i], kind=example_info['dt'])
        selected_annotation_dfs.append(s_annotation.select_annotation_by_ts(annotation_df, lbound=lbound, rbound=rbound))
        selected_raw_dfs.append(s_raw.select_raw_by_ts(raw_df, lbound=lbound, rbound=rbound))
    s_viewer.get_multistep_view_plot(selected_raw_dfs, labels=selected_annotation_dfs,titles=example_info['titles'], subplots=False, sharex=False, figsize=(9.08,2), time_display='sec')
    pyplot.show()

def generate_all_sensors_comparison():

    example_info = {
        'session': 1,
        'STARTTIME1': "12:02:45",
        'ENDTIME1': "12:04:45",
        'dt': 'clean',
        'sensors': s_info.sensor_codes
    }
    lbound1 = w_utils.convert_fromstring(s_info.session_dates[s_info.session_arr.index(example_info['session'])] + ' ' + example_info['STARTTIME1'], s_info.tstr_format)
    rbound1 = w_utils.convert_fromstring(s_info.session_dates[s_info.session_arr.index(example_info['session'])] + ' ' + example_info['ENDTIME1'], s_info.tstr_format)

    titles = s_info.sensor_codes
    raw_df, annotation_df = s_loader.load_smoking_df(session=example_info['session'], sensors=example_info['sensors'], kind=example_info['dt'])
    # lbound, rbound = s_annotation.generate_random_bounds(annotation_df)
    selected_annotation_df1 = s_annotation.select_annotation_by_ts(annotation_df, lbound=lbound1, rbound=rbound1, by='sensor')
    selected_raw_df1 = s_raw.select_raw_by_ts(raw_df, lbound=lbound1, rbound=rbound1, by='sensor')
    s_viewer.get_multisensor_raw_plot(selected_raw_df1, labels=selected_annotation_df1, titles=s_info.sensor_names, figsize=(12.12, 2))
    pyplot.show()

def generate_ambiguous_activities():

    example_info = {
        'session': [4,4],
        'STARTTIME': ["12:38:00","11:50:07","11:19:58"],
        'ENDTIME': ["12:40:00","11:52:07","11:21:58"],
        'dt': 'clean',
        'sensors': [['DW', 'NDW'], ['DW', 'NDW']]
    }
    for i in range(0, len(example_info['session'])):
        lbound = w_utils.convert_fromstring(s_info.session_dates[s_info.session_arr.index(example_info['session'][i])] + ' ' + example_info['STARTTIME'][i], s_info.tstr_format)
        rbound = w_utils.convert_fromstring(s_info.session_dates[s_info.session_arr.index(example_info['session'][i])] + ' ' + example_info['ENDTIME'][i], s_info.tstr_format)

        raw_df, annotation_df = s_loader.load_smoking_df(session=example_info['session'][i], sensors=example_info['sensors'][i], kind=example_info['dt'])
        # lbound, rbound = s_annotation.generate_bounds_by_labels(annotation_df.groupby(by='sensor').get_group('DW'), duration=timedelta(minutes=2), labels=['eating-a-meal',])
        selected_annotation_df = s_annotation.select_annotation_by_ts(annotation_df, lbound=lbound, rbound=rbound, by='sensor')
        selected_raw_df = s_raw.select_raw_by_ts(raw_df, lbound=lbound, rbound=rbound, by='sensor')
        s_viewer.get_multisensor_raw_plot(selected_raw_df, labels=selected_annotation_df, titles=s_info.sensor_names, figsize=(4.5,5.04), orientation='', time_display='sec')
    pyplot.show()

def generate_ambiguous_activities_with_smoking():

    example_info = {
            'session': [1, ],
            'STARTTIME': ["11:52:50", ],
            'ENDTIME': ["11:54:50", ],
            'dt': 'clean',
            'sensors': [['DW', 'NDW'], ],
            'titles': ['smoking while eating and drinking',]
    }
    for i in range(0, len(example_info['session'])):
        lbound = w_utils.convert_fromstring(s_info.session_dates[s_info.session_arr.index(example_info['session'][i])] + ' ' + example_info['STARTTIME'][i], s_info.tstr_format)
        rbound = w_utils.convert_fromstring(s_info.session_dates[s_info.session_arr.index(example_info['session'][i])] + ' ' + example_info['ENDTIME'][i], s_info.tstr_format)

        raw_df, annotation_df = s_loader.load_smoking_df(session=example_info['session'][i], sensors=example_info['sensors'][i], kind=example_info['dt'])
        # lbound, rbound = s_annotation.generate_bounds_by_labels(annotation_df.groupby(by='sensor').get_group('DW'), duration=timedelta(minutes=2), labels=['eating-a-meal',])
        selected_annotation_df = s_annotation.select_annotation_by_ts(annotation_df, lbound=lbound, rbound=rbound, by='sensor')
        selected_raw_df = s_raw.select_raw_by_ts(raw_df, lbound=lbound, rbound=rbound, by='sensor')
        s_viewer.get_multisensor_raw_plot(selected_raw_df, labels=selected_annotation_df, titles=s_info.sensor_names, figsize=(6.04,2), orientation='horizontal', time_display='sec')
    pyplot.show()

def generate_histogram_and_distribution():
    filenames = ['../../statistics_dataset/interpuff_fit.csv','../../statistics_dataset/puff_freq_fit.csv']
    fignames = ['histogram and distribution of interpuff intervals', 'histogram and distribution of puff frequency']
    from scipy.stats import poisson, gamma, lognorm, norm
    fig, axes = pyplot.subplots(1,2, figsize=(6,2))
    fig.tight_layout()
    df = pd.read_csv(filenames[0])
    axes[0].hist(df['interpuff interval'], normed=1, bins=50, fill=False)
    params = gamma.fit(df['interpuff interval'], floc=0)
    print params
    x = range(0, 300)
    fitted = gamma.pdf(x, params[0], scale=params[2])
    # fitted2 = lognorm.pdf(x, 0.79, scale=25)
    axes[0].plot(x, fitted, linewidth=2)
    axes[0].set_xlabel('interpuff interval(s)')
    axes[0].set_ylabel('probability')
    text = "shape=%.2f\nscale=%.2f" % (params[0], params[2])
    axes[0].annotate(text, xycoords='data', xy=(200, 0.02))

    df = pd.read_csv(filenames[1])
    n, bins, patches = axes[1].hist(df['Puff frequency'], normed=1, bins=10, fill=False, range=(0.1, 4.5))
    x = np.linspace(0,5,200)
    params = gamma.fit(df['Puff frequency'], floc=0)
    print params
    fitted = gamma.pdf(x, params[0], scale=params[2])
    axes[1].plot(x,fitted)
    axes[1].set_xlabel('puff frequency(puffs/mins)')
    text = "shape=%.2f\nscale=%.2f" % (params[0], params[2])
    axes[1].annotate(text, xycoords='data', xy=(3.5, 0.55))
    pyplot.subplots_adjust(bottom=0.15, left=0.1)
    pyplot.show()


if __name__ == "__main__":
    # generate_prototypical_puff_and_sequence(example='prototypical')
    # generate_prototypical_puff_and_sequence(example='non-prototypical')
    # generate_prototypical_puff_and_sequence(example='eatsmoking')
    # generate_prototypical_puff_and_sequence(example='body_variety')
    # generate_prototypical_puff_and_sequence(example='hand_variety')
    # generate_all_sensors_comparison()
    # generate_ambiguous_activities()
    # generate_ambiguous_activities_with_smoking()
    generate_histogram_and_distribution()
    # generate_smoking_detection_plot()