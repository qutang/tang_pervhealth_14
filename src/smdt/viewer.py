#! /usr/bin/python

"""
 Module of smdt data visualization
"""

import sys
from datetime import timedelta
sys.path.append('../')
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm
import matplotlib.colors as colors
import pandas as pd
import numpy as np
import annotation as s_annotation
import wockets.utils as w_utils
import smdt.raw as s_raw
import smdt.info as s_info 

publishable_color_cycle = [str(x) for x in np.linspace(0,0.5,3)]
publishable_line_styles = ['-','--','-.']

jet_map = cm.get_cmap('rainbow')
cNorm  = colors.Normalize(vmin=0, vmax=6)
jet_cm_arr = cm.ScalarMappable(norm=cNorm, cmap=jet_map)
curve_colors = ['blue','cyan','gray']
activity_sequence = ['drinking-beverage', 'eating-a-meal', 'using-phone', 'using-computer', 'reading-paper', 'talking', 'unknown-activity']


def get_activity_view_plot(raw_data, annotation_data, prediction_data=None, visualtest_data=None, title=''):  
  """Get synthetic activity view figure

  Get synthetic activity view figure which contains all the annotation and the figure
  will be created but not shown, to show it in time, call pyplot.show() or you category_names
  save it to some folder by calling pyplot.savefig
  true and predicted smoking and puffing information, and visualtest prediction results

  Args:
    raw_data: raw dataset to be used
    annotation_data: annotation dataset to be used
    prediction_data: prediction dataset of puffing labels and smoking prediction probability
    visualtest_data: visualtest dataset of visual test results
    title: the title of the figure

  Returns:
    handler of activity view figure
  """

  #step 1: get the start rows of each 10s segment of raw dataset
  raw_data[s_info.raw_ts_name]
  subsample_rate = 40*10  #40samples/s*10s
  sub_idxs = range(0, raw_data.shape[0]-1,subsample_rate)
  subsampled_raw = raw_data.ix[sub_idxs, s_info.raw_ts_name]
  subsampled_raw = subsampled_raw.reset_index(drop=True)
  
  #step2: get annotation that is corresponding to each segments
  annotation_arr = []
  for ts in subsampled_raw:
    flag = np.logical_and(annotation_data[s_annotation.st_col] <= ts, annotation_data[s_annotation.et_col] > ts)
    selected_row = annotation_data.ix[flag,:]
    selected_annotations = selected_row[s_annotation.category_names].values[0]
    # except IndexError:
    #   print subsampled_raw
    #   print ts
    #   print annotation_data[annot.annotation_et_col]
    #   print selected_row[annot.annotation_action_names].values
    #   sys.exit(1)
    annotation_arr.append(selected_annotations)

  #step 3: construct activity dataset used for plot, this activity dataset is
  # a dataset subsampled from raw dataset with true labels assigned to each of the segments
  # No prediction results are included 
  activity_data = pd.DataFrame(np.array(annotation_arr), 
                                columns=s_annotation.category_names) #add annotation columns
  activity_data.insert(0,s_info.raw_ts_name,subsampled_raw) #add timestamp column

  #step 4: refine puffing and drinking in each segment of activity dataset
  row_idx = 0
  for lts, rts in zip(subsampled_raw, subsampled_raw.shift(-1))[:-1]:
    selected_annotations = s_annotation.select_annotation_by_ts(annotation_data, lts, rts)
    puff_values = selected_annotations['puffing']
    if 'left-puff' in puff_values.values or 'right-puff' in puff_values.values:
      activity_data['puffing'][row_idx] = 'puff'
    else:
      activity_data['puffing'][row_idx] = 'no-puff'
    drink_values = selected_annotations['activity']
    if 'drinking-beverage' in drink_values.values:
      activity_data['activity'][row_idx] = 'drinking-beverage'
    row_idx += 1

  """
  Plot begins
  """
  activity_fig_handler = pyplot.figure(figsize=(15,10))
  xtimes = activity_data[s_info.raw_ts_name]

  # smoking
  smoking_flags = activity_data['smoking'] == 'smoking'
  smoking_points = smoking_flags.astype(int)
  pyplot.plot(xtimes, smoking_points, color='black')
  
  # posture
  # walk
  walk_flags = activity_data['posture'] == 'walking'
  walk_points = walk_flags.astype(int)*-0.5
  walk_points = walk_points[walk_flags]
  walktimes = xtimes[walk_flags]
  pyplot.plot(walktimes.tolist(), walk_points.tolist(),marker='>',markersize=8,markerfacecolor='None',linestyle='None')
  # stand
  stand_flags = activity_data['posture'] == 'standing'
  stand_points = stand_flags.astype(int)*-0.4
  stand_points = stand_points[stand_flags]
  standtimes = xtimes[stand_flags]
  pyplot.plot(standtimes.tolist(), stand_points.tolist(),marker='|',markersize=15,markerfacecolor='None',linestyle='None',color='black')

  # activities
  # eating
  eat_flags = activity_data['activity'] == 'eating-a-meal'
  eat_points = eat_flags.astype(int)*1.3
  eat_points = eat_points[eat_flags]
  eattimes = xtimes[eat_flags]
  pyplot.plot(eattimes.tolist(), eat_points.tolist(),marker='*',markersize=8,markerfacecolor='None',linestyle='None',color='black')
  # talking
  talk_flags = activity_data['activity'] == 'talking'
  talk_points = talk_flags.astype(int)*1.5
  talk_points = talk_points[talk_flags]
  talktimes = xtimes[talk_flags]
  pyplot.plot(talktimes.tolist(), talk_points.tolist(),marker='o',markersize=8,markerfacecolor='None',linestyle='None',color='black')
  pyplot.ylim([-1,10])
  # using phone
  phone_flags = activity_data['activity'] == 'using-phone'
  phone_points = phone_flags.astype(int)*1.7
  phone_points = phone_points[phone_flags]
  phonetimes = xtimes[phone_flags]
  pyplot.plot(phonetimes.tolist(), phone_points.tolist(),marker='d',markersize=8,markerfacecolor='None',linestyle='None',color='black')
  # using computer
  computer_flags = activity_data['activity'] == 'using-computer'
  computer_points = computer_flags.astype(int)*1.9
  computer_points = computer_points[computer_flags]
  computertimes = xtimes[computer_flags]
  pyplot.plot(computertimes.tolist(), computer_points.tolist(),marker='s',markersize=8,markerfacecolor='None',linestyle='None',color='black')
  # other(unknown) moves
  other_flags = activity_data['activity'] == 'unknown-activities'
  other_points = other_flags.astype(int)*2.1
  other_points = other_points[other_flags]
  othertimes = xtimes[other_flags]
  pyplot.plot(othertimes.tolist(), other_points.tolist(),marker='x',markersize=8,markerfacecolor='None',linestyle='None',color='black')
  # reading
  read_flags = activity_data['activity'] == 'reading-paper'
  read_points = read_flags.astype(int)*2.3
  read_points = read_points[read_flags]
  readtimes = xtimes[read_flags]
  pyplot.plot(readtimes.tolist(), read_points.tolist(),marker='h',markersize=8,markerfacecolor='None',linestyle='None',color='black')
  # in car
  car_flags = activity_data['activity'] == 'in-car'
  car_points = car_flags.astype(int)*2.5
  car_points = car_points[car_flags]
  cartimes = xtimes[car_flags]
  pyplot.plot(cartimes.tolist(), car_points.tolist(),marker='8',markersize=8,markerfacecolor='None',linestyle='None',color='black')

  # puffs
  puff_flags = activity_data['puffing'] == 'puff'
  puff_points = puff_flags.astype(int)*0.9
  puff_points = puff_points[puff_flags]
  pufftimes = xtimes[puff_flags]
  pyplot.plot(pufftimes.tolist(), puff_points.tolist(),marker='^',markersize=8,markerfacecolor='None',linestyle='None',color='black')

  # drinks
  drink_flags = activity_data['activity'] == 'drinking-beverage'
  drink_points = drink_flags.astype(int)*1.1
  drink_points = drink_points[drink_flags]
  drinktimes = xtimes[drink_flags]
  pyplot.plot(drinktimes.tolist(), drink_points.tolist(),marker='v',markersize=8,markerfacecolor='None',linestyle='None',color='black')

  # add prediction data: start from 2.7
  if prediction_data is not None:
    predicttimes = xtimes[range(0,len(prediction_data['prob']),1)]
    predict_values = prediction_data['prob']+2.8
    pyplot.plot(predicttimes.tolist(),predict_values,color='black')

    prepufftimes = xtimes[range(0,len(prediction_data['puffs']),1)]
    prepufftimes = prepufftimes[prediction_data['puffs']==1]
    prepuff_values = prediction_data['puffs'][prediction_data['puffs'] == 1]+1.7
    pyplot.plot(prepufftimes.tolist(),prepuff_values.tolist(),marker='^',markersize=8,markerfacecolor='None',linestyle='None',color='black')

  # add visual test data : start from 4
  if visualtest_data is not None:
    for idx, lx, rx, y in visualtest_data[['index','STARTTIME','ENDTIME','correct']].itertuples(index=False):
      y += 4
      # convert str into datetime
      lx = w_utils.convert_fromstring(lx, '%m/%d/%Y %H:%M:%S')
      rx = w_utils.convert_fromstring(rx, '%m/%d/%Y %H:%M:%S')
      xts = xtimes[np.logical_and(xtimes> lx, xtimes <= rx)]
      pyplot.plot(xts.tolist(),np.ones(xts.shape)*y,color='black',linestyle='-')
      pyplot.annotate(str(idx),(xts.tolist()[0],y+0.1),xycoords='data',size='small')

  # figure settings
  pyplot.ylim([-1,6])
  pyplot.grid('on')
  pyplot.xticks(xtimes[::30], xtimes[::30].apply(lambda x: x.strftime('%H:%M:%S')).tolist())
  pyplot.yticks([-0.5,-0.4,0.9,1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 4, 4.5, 5],
                ['walk','stand','puffs(true)','drink','eat','talk','phone',
                'computer','unknown','read','car','puffs(prediction)',
                'guess(wrong)','guess(half)','guess(right)'])
  pyplot.gcf().autofmt_xdate()
  pyplot.title(title)

  return activity_fig_handler

def _prepare_raw_for_plot(raw_data, useCols=None):
  """Get raw dataset ready for plot

  Add y value irrelevant columns to index list, including timestamp columns, session, subject and sensor.
  And then choose value columns to be plotted. This function is used internally 
  most of the time, but call be called from outside.

  Args:
    raw_data: raw dataset to be processed
    useCols: list of columns to be used in plot, default is None, which means use
    all the columns

  Returns:
    cleaned up raw dataset for plot
  """

  index_list = []
  if w_utils.raw_ts_name in raw_data.columns:
    index_list.append(w_utils.raw_ts_name)
  if 'session' in raw_data.columns:
    index_list.append('session')
  if 'subject' in raw_data.columns:
    index_list.append('subject')
  if 'sensor' in raw_data.columns:
    index_list.append('sensor')
  raw_data = raw_data.set_index(index_list, append=True)
  if None == useCols:
    raw_data = raw_data
  else:
    raw_data = raw_data[useCols]
  return raw_data

def add_labels_to_raw_plot(labels_data, raw_data, ax=pyplot.gca()):
  """Add annotation to plot

  Add annotation to raw dataset plot, this function should be used with raw dataset plot only
  And it's meaningless to use it alone
  The style of annotation is vertical lines with labels

  Args:
    labels_data: the annotation dataset used for labeling
    raw_data: the raw dataset used to identify the time point for the annotation
    ax: the axis to be plotted on, default is current axis
  """
  if labels_data is None:
    return
  for idx, label_times in labels_data[[s_annotation.st_col, s_annotation.et_col]].iterrows():
    xvalue = raw_data.index.get_level_values(level=w_utils.raw_ts_name).tolist().index(label_times[0])
    labelseries = labels_data.ix[idx,s_annotation.category_names].fillna('unknown')
    if idx == 0:
      labellist = np.array(labelseries.tolist())
      labeltext = '>' + '\n>'.join(labellist)
    else:
      newlist = np.array(labelseries.tolist())
      changelist = newlist[newlist != labellist]
      labeltext = '>' + '\n>'.join(changelist)
      # update labellist
      labellist = labellist.astype(changelist.dtype)
      labellist[newlist != labellist] = changelist
    if idx % 2 == 0:
      ax.annotate(labeltext,(xvalue,ax.get_ylim()[0]+0.02*idx*abs(ax.get_ylim()[0])),xycoords='data', size='x-small')
    else:
      ax.annotate(labeltext,(xvalue,ax.get_ylim()[1]-0.02*idx*abs(ax.get_ylim()[1])),xycoords='data', size='x-small')
    ax.axvline(x=xvalue,ymax=1)

def add_color_labels_to_raw_plot(labels_data, raw_data, ax=pyplot.gca() ,ylims=(-3,3)):
  """Add annotation to plot

  Add annotation to raw dataset plot, this function should be used with raw dataset plot only
  And it's meaningless to use it alone
  The style of annotation is thin horizontal color bar with word marker on it

  Args:
    labels_data: the annotation dataset used for labeling
    raw_data: the raw dataset used to identify the time point for the annotation
    ax: the axis to be plotted on, default is current axis
  """

  if labels_data is None:
    return
  for idx, label_times in labels_data[[s_annotation.st_col, s_annotation.et_col]].iterrows():
    # st_xvalue = label_times[0]
    # et_xvalue = label_times[1]
    # xmin = ax.get_xlim()[0]
    # xmax = ax.get_xlim()[1]
    st_xvalue = raw_data.index.get_level_values(level=w_utils.raw_ts_name).tolist().index(label_times[0]) # start time
    et_xvalue = raw_data.index.get_level_values(level=w_utils.raw_ts_name).tolist().index(label_times[1]) # end time
    labelseries = labels_data.ix[idx,s_annotation.category_names].fillna('unknown')
    # plot for activity annotation
    for c in range(0, len(activity_sequence)):
      if labelseries[s_annotation.activity_col] == activity_sequence[c]:
        ax.hlines(xmin=st_xvalue, xmax=et_xvalue, y=ylims[0]+ylims[1]/30., linewidth=4, color=jet_cm_arr.to_rgba(c))

    if labelseries[s_annotation.smoke_col] == 'smoking':
      ax.hlines(xmin=st_xvalue, xmax=et_xvalue, y=ylims[1]-ylims[1]/30., linewidth=3, color='0.3')
    if labelseries[s_annotation.puff_col] == 'left-puff':
      ax.hlines(xmin=st_xvalue, xmax=et_xvalue, y=ylims[1]-ylims[1]/30.*4, linewidth=4, color='green')
    elif labelseries[s_annotation.puff_col] == 'right-puff':
      ax.hlines(xmin=st_xvalue, xmax=et_xvalue, y=ylims[1]-ylims[1]/30.*4, linewidth=4, color='blue')
    # plot for walking annotation
    if labelseries[s_annotation.post_col] == 'walking':
      ax.hlines(xmin=st_xvalue, xmax=et_xvalue, y=ylims[0]+ylims[1]/30.*4, linewidth=4, color='magenta')
    elif labelseries[s_annotation.post_col] == 'standing':
      ax.hlines(xmin=st_xvalue, xmax=et_xvalue, y=ylims[0]+ylims[1]/30.*4, linewidth=4, color='blue')
    elif labelseries[s_annotation.post_col] == 'unknown-posture':
      ax.hlines(xmin=st_xvalue, xmax=et_xvalue, y=ylims[0]+ylims[1]/30.*4, linewidth=4, color='gray')
def get_legends_plot():
  pyplot.figure(figsize=(1.1,4))
  # print out the legend
  pyplot.scatter([[0.1],]*3, [[0.6],[0.75],[0.9]],c=curve_colors,marker='s', s=100, edgecolors='none')
  yspread =np.linspace(0.3,-0.5,len(activity_sequence))
  for c in range(0, len(activity_sequence)):
    pyplot.scatter(0.1, yspread[c], s=100, edgecolors='none', c=jet_cm_arr.to_rgba(c))
  pyplot.scatter([0.1,]*2, [-0.95, -0.8], s=100, edgecolors='none', c=['green', 'blue'])
  pyplot.scatter([0.1,]*3, [-1.25, -1.4, -1.55], s=100, edgecolors='none', c=['magenta', 'blue', 'gray'])

  # add text
  pyplot.annotate('rawx',(0.15,0.9-0.03),xycoords='data')
  pyplot.annotate('rawy',(0.15,0.75-0.03),xycoords='data')
  pyplot.annotate('rawz',(0.15,0.6-0.03),xycoords='data')
  for label, y in zip(activity_sequence, yspread):
    pyplot.annotate(label,(0.15, y-0.03),xycoords='data')
  pyplot.annotate('right puff',(0.15,-0.8-0.03),xycoords='data')
  pyplot.annotate('left puff',(0.15,-0.95-0.03),xycoords='data')
  pyplot.annotate('standing',(0.15,-1.25-0.03),xycoords='data')
  pyplot.annotate('walking',(0.15,-1.4-0.03),xycoords='data')
  pyplot.annotate('unknown-posture',(0.15,-1.55-0.03),xycoords='data')
  pyplot.ylim((-1.6,1))
  pyplot.xlim((0,0.5))
  pyplot.axis('off')
  return pyplot.gcf()

def get_singlesensor_raw_plot(raw_data, labels=None, subplots=False, useCols=None, ax=None, 
                              figsize=(15,10), color='auto',linestyle='auto', ylabel='',
                              title='accel raw', time_display='sec', ytick_display=False):
    """Get raw dataset plot with only one sensor

    Get single sensor raw dataset plot, there are various plot options can be set,
    the figure will be plotted but not shown, to show it call pyplot.show(), to
    save it call pyplot.savefig()

    Args:
      raw_data: raw dataset to be plotted
      labels: annotation dataset to be added, default is None
      subplots: Bool to decide whether to show each axis in different subplot
      useCols: list of columns to be used
      ax: axis handler used to plot raw dataset on, default is False
      figsize: tuple (width, height) in inch to specify the figure size, default is
      (15,10)
      color: color list for each line, default is 'auto' 
      linestyle: linestyle list for each line, default is 'auto'
      ylabel: ylabel string
      title: title string

    Returns:
      figure handler
    """
    # raw_data = _prepare_raw_for_plot(raw_data, useCols=useCols)
    # if 'session' in raw_data.index.names:
    #   raw_data = raw_data.reset_index(level=['session',], drop=True)
    # if 'subject' in raw_data.index.names:
    #   raw_data = raw_data.reset_index(level=['subject',], drop=True)
    # if 'sensor' in raw_data.index.names:
    #   raw_data = raw_data.reset_index(level=['sensor',], drop=True)
    if color == 'auto' and linestyle != 'auto':
      hsubplots = raw_data.plot(subplots=subplots, use_index=True,title=title, ax=ax, sharey=True, figsize=figsize, linestyle=linestyle, color=curve_colors)
    elif color != 'auto' and linestyle == 'auto':
      hsubplots = raw_data.plot(subplots=subplots, use_index=False,title=title, ax=ax, sharey=True, figsize=figsize, color=color)
    elif color == 'auto' and linestyle == 'auto':
      hsubplots = raw_data.plot(subplots=subplots, use_index=False,title=title, ax=ax, sharey=True, figsize=figsize, color=curve_colors)
    else:
      hsubplots = raw_data.plot(subplots=subplots, use_index=False,title=title, ax=ax, sharey=True, figsize=figsize, linestyle=linestyle,color=color)
  
    xticklabels = raw_data.index.get_level_values(level=w_utils.raw_ts_name).tolist()
    if time_display == 'sec':
        xticklabels = np.array(xticklabels) - xticklabels[0]
        xticklabels = np.array([str(x.total_seconds()) for x in xticklabels])
    else:
        xticklabels = np.array([x.strftime("%H:%M:%S.%f")[:-3] for x in xticklabels])
    xvalues = range(0,len(xticklabels),len(xticklabels)/15)
    if np.iterable(hsubplots):
      for h in hsubplots:
        h.set_xticks(xvalues)
        h.set_xticklabels(xticklabels[xvalues], rotation=20,size='small')
        h.legend().set_visible(False)
        h.set_xlabel("")
        h.set_ylabel("")
        if h.get_ylim()[1] > 10:
          ylims = (-180, 180)
          h.set_ylim(ylims)
        else:
          ylims = (-2,2)
          h.set_ylim(ylims)
        add_color_labels_to_raw_plot(labels, raw_data, ax=h, ylims=ylims)
        h.set_yticks([ylims[0]+ylims[1]/30, ylims[0]+ylims[1]/30.*4, -1.2, -0.6, 0, 0.6, 1.2, ylims[1]-ylims[1]/30.*4, ylims[1]-ylims[1]/30])
    else:
      hsubplots.set_xticks(xvalues)
      hsubplots.set_xticklabels(xticklabels[xvalues], rotation=20,size='x-small')
      hsubplots.legend().set_visible(False)
      hsubplots.set_xlabel("")
      hsubplots.set_ylabel("")
      if hsubplots.get_ylim()[1] > 10:
        ylims = (-180, 180)
        hsubplots.set_ylim(ylims)
      else:
        ylims = (-2, 2)
        hsubplots.set_ylim(ylims)
      add_color_labels_to_raw_plot(labels, raw_data, ax=hsubplots, ylims=ylims)
      if ytick_display:
        hsubplots.set_yticks([ylims[0]+ylims[1]/30, ylims[0]+ylims[1]/30.*4, -1.2, -0.6, 0, 0.6, 1.2, ylims[1]-ylims[1]/30.*4, ylims[1]-ylims[1]/30])
        hsubplots.set_yticklabels(['activity', 'posture', '-1.2', '-0.6', '0', '0.6', '1.2', 'puff', 'smoking'])
      else:
        hsubplots.set_yticks([ylims[0]+ylims[1]/30, ylims[0]+ylims[1]/30.*4, -1.2, -0.6, 0, 0.6, 1.2, ylims[1]-ylims[1]/30.*4, ylims[1]-ylims[1]/30])
        hsubplots.set_yticklabels(['', '', '', '', '', '', '', '', ''])
        hsubplots.xaxis.grid(True)
    return pyplot.gcf()

def get_multistep_view_plot(raw_datas, labels, titles, subplots=False, 
                            sharex=True, figsize=(15,10), appear='auto', orientation='horizontal', time_display='sec'):
  """Get multistep process view based on raw dataset

  Plot multistep(several datasets) in different subplots, the figure will be generated
  but not shown. To show it call pyplot.show() or to save it call pyplot.savefig()

  Args:
    raw_datas: list of accelerometer datasets corresponds to each step
    labels: list of annotation datasets corresponds to each step, some can be None
    titles: list of title strings for each step
    subplots: Boolean to decide whether to plot axis in different subplots, default is False
    sharex: Boolean to decide whether to share x value for all subplots, default is True
    figsize: tuple (width, height) to decide the figure size
    appear: 'gray' or 'auto', decide the appearance of lines, default is auto
  
  Returns:
    figure handler
  """

  new_raw_datas = []
  for raw_data in raw_datas:
    raw_data = _prepare_raw_for_plot(raw_data)
    new_raw_datas.append(raw_data)
  raw_datas = new_raw_datas
  
  # decide alignment of subplots,set up subplots
  if subplots:
    subcols = len(raw_datas)
    subrows = len(raw_datas[0].columns)
  else:
    if orientation == 'horizontal':
      subcols = len(raw_datas)
      subrows = 1
    else:
      subcols = 1
      subrows = len(raw_datas)
  fig, subaxes = pyplot.subplots(subrows, subcols,sharex=sharex, squeeze=False, figsize=figsize)
  # for each step
  r = 0
  c = 0
  for raw_data in raw_datas:
    raw_data = _prepare_raw_for_plot(raw_data)
    # plot each axis in different subplot
    if subplots:
      for series_name,one_series in raw_data.iteritems():
        if w_utils.raw_ts_name in raw_data.index.names:
          if orientation == 'horizontal':
            ax = subaxes[c,r]
          else:
            ax = subaxes[r,c]
          if appear == 'gray':
            get_singlesensor_raw_plot(one_series, ax=ax, labels=labels[c], ylabel=series_name, title=titles[c], color=publishable_color_cycle[0], linestyle=publishable_line_styles[0], time_display=time_display)
          elif appear == 'auto':
            get_singlesensor_raw_plot(one_series, ax=ax, labels=labels[c], ylabel=series_name, title=titles[c], time_display=time_display)
          r += 1
        elif w_utils.raw_fq_name in raw_data.index.names:
          # TODO: add spectrum(FFT) plot functionality
          print 'plot spectrum'
      c += 1
      r = 0
    else:
      if w_utils.raw_ts_name in raw_data.index.names:
        if orientation == 'horizontal':
          ax = subaxes[c, r]
        else:
          ax = subaxes[r, c]
        if r == 0:
          get_singlesensor_raw_plot(raw_data, ax=ax, labels=labels[r], title=titles[r], time_display=time_display, ytick_display=True)
        else:
          get_singlesensor_raw_plot(raw_data, ax=ax, labels=labels[r], title=titles[r], time_display=time_display, ytick_display=False)
      elif w_utils.raw_fq_name in raw_data.index.names:
        # TODO: add spectrum(FFT) plot functionality
        print 'plot spectrum'
      r += 1
  fig.tight_layout()
  if orientation == 'horizontal':
    fig.subplots_adjust(bottom=0.15, left=0.08, top=0.9)
    fig.text(0.02, 0.5, 'acceleration (g)', ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.04, 'time (s)', ha='center', va='center')
  else:
    fig.text(0.02, 0.5, 'acceleration (g)', ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.04, 'time (s)', ha='center', va='center')
  return fig
  
def get_multisensor_raw_plot(raw_data, labels=None, useCols=None, subplots=False, orientation='horizontal', titles=None, figsize=(15,4), time_display='sec'):
  """Get multiple sensor raw dataset plot

  Plot raw dataset with multiple sensors' data, the figure will be generated but
  not shown, to show user can call pyplot.show() or to save call pyplot.savefig()

  Args:
    raw_data: raw dataset to be processed
    labels: annotation dataset to be used, default is None
    useCols: specify value columns of raw dataset to be used, default is None which means all columns
    subplots: Boolean to decide whether to plot axis in different subplot, default is False

  Returns:
    figure handler
  """

  raw_data = _prepare_raw_for_plot(raw_data, useCols)
  if 'sensor' in raw_data.index.names:
    # arrange subplot alignment
    unique_sensors = raw_data.index.get_level_values('sensor').unique()
    if subplots:
      subrows = len(raw_data.columns)
      subcols = len(unique_sensors)
    else:
      if orientation == 'horizontal':
          subcols = len(unique_sensors)
          subrows = 1
      else:
          subcols = 1
          subrows = len(unique_sensors)

    fig, subaxes = pyplot.subplots(subrows, subcols,sharex=True, sharey=True, squeeze=False, figsize=figsize)
    fig.tight_layout()
    if orientation == 'horizontal':
        fig.subplots_adjust(left=0.08/figsize[0]*10, bottom=0.15, top=0.9)
        fig.text(0.02, 0.5, 'acceleration (g)', ha='center', va='center', rotation='vertical')
        fig.text(0.5, 0.04, 'time (s)', ha='center', va='center')
    else:
        fig.subplots_adjust(left=0.2, bottom=0.07,top=0.95)
        fig.text(0.04, 0.5, 'acceleration (g)', ha='center', va='center', rotation='vertical')
        fig.text(0.5, 0.02, 'time (s)', ha='center', va='center')
    groupby_data = raw_data.groupby(level='sensor')
    if labels is not None and 'sensor' in labels.columns:
      groupby_labels = labels.groupby('sensor')
    r, c = (0, 0)
    for group_name, one_sensor_data in groupby_data:
      if labels is not None and 'sensor' in labels.columns:
        group_labels = labels.ix[groupby_labels.groups[group_name],:]
        group_labels = group_labels.reset_index(drop=True)
        # print group_labels
      else:
        group_labels = labels
      if subplots:
        for series_name,one_series in one_sensor_data.iteritems():
          if orientation == 'horizontal':
            ax = subaxes[c, r]
          else:
            ax = subaxes[r, c]
          get_singlesensor_raw_plot(one_series, ax=ax, labels=group_labels)
          r += 1
        c += 1
        r = 0
      else:
        if orientation == 'horizontal':
          ax = subaxes[c, r]
        else:
          ax = subaxes[r, c]
        if titles is not None:
          group_name = titles[s_info.sensor_codes.index(group_name)]

        if r == 0:
          get_singlesensor_raw_plot(one_sensor_data, subplots=subplots, ax=ax, labels=group_labels, title=group_name, time_display=time_display, ytick_display=True)
        else:
          get_singlesensor_raw_plot(one_sensor_data, subplots=subplots, ax=ax, labels=group_labels, title=group_name, time_display=time_display, ytick_display=True)
        r += 1
  else:
    fig = get_simple_raw_plots(raw_data, labels=labels, useCols=useCols, subplots=subplots)

  return fig

def get_simple_raw_plots(raw_data, labels=None, useCols=None, subplots=False, by=None):
  """ Get simple raw dataset plots

  Plot raw dataset in different figures as grouped by "by" column, those figures
  will be generated but not shown. To show, call pyplot.show() or to save, call
  pyplot.savefig()

  Args:
    raw_data: raw dataset to be plotted
    labels: annotation dataset to be plotted, default is None
    useCols: list of value columns to be used, default is None which means use all columns
    subplots: Boolean to decide whether to plot axis in subplots, default is False
    by: specify how to group the dataset, each group dataset will be plotted on a figure

  Returns:
    list of figure handlers
  """

  raw_data = _prepare_raw_for_plot(raw_data, useCols)
  if by != None and np.iterable(by):
    fig_handlers = []
    groupby_data = raw_data.groupby(level=by)
    for group_name, group_data in groupby_data:
      fig = get_singlesensor_raw_plot(group_data, labels=labels, subplots=subplots, title=group_name)
      fig_handlers.append(fig)
    return fig_handlers
  else:
    fig = get_singlesensor_raw_plot(raw_data, labels=labels, subplots=subplots)
    return fig



def unit_tests(rawfile, annotfile, predictfile, visualfile):
  raw_data = w_utils.raw_csv_consolidator([rawfile,], sessions=[1,],subjects=[5,],sensors=['wrist',])
  annotation_data = s_annotation.annotation_csv_consolidator([annotfile,],sessions=[1,],subjects=[5,], sensors=['wrist',])
  raw_data = s_raw.preprocess_raw(raw_data, annotation_data, grace_period = timedelta(seconds=0))

  # prediction_data = pd.read_csv(predictfile, names=['prob','puffs'])

  # visualtest_data = pd.read_csv(visualfile, header=0)

  # activity_viewer(raw_data, annotation_data, prediction_data=prediction_data, visualtest_data=visualtest_data)
  # print annotation_data
  # lbound, rbound = annot.generate_random_annotation_bounds(annotation_data, timedelta(minutes=1))
  lbound, rbound = s_annotation.generate_bounds_by_labels(annotation_data, duration=timedelta(minutes=5), labels=['walking','smoking'])

  raw_data = s_raw.preprocess_raw(raw_data, annotation_data)
  raw_data = s_raw.select_raw_by_ts(raw_data, lbound, rbound)
  # annotation_data = annot.smdt_annotation_select_by_ts(annotation_data, lbound, rbound)
  # step1_data = rawModule.smdt_raw_filtering(raw_data, utils.raw_value_names, filt='lowpass')
  # multistep_datas = [raw_data, step1_data]
  # raw_data_simple_viewer(raw_data,by=['session','sensor'],subplots=False, labels=annotation_data)
  # raw_data_multisensor_viewer(raw_data, labels=annotation_data, subplots=False)
  # multistep_data_viewer(multistep_datas, labels=annotation_data, subplots=False, sharex=True)

if __name__ == "__main__":
  import cProfile
  pr = cProfile.Profile()
  pr.enable()
  testfile_raw = "../../test_DW.raw.csv"
  testfile_annotation = "../../test.annotation.csv"
  testfile_prediction = "../../test.prediction.csv"
  testfile_visualtest = "../../test.visualtest.csv"
  unit_tests(testfile_raw, testfile_annotation, testfile_prediction, testfile_visualtest)
  pr.disable()
  pr.create_stats()
  # pr.print_stats(sort=1)