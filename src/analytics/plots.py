#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import seaborn as sns
import datetime
import matplotlib
import collections
from collections import Counter
from matplotlib import pyplot
import logging


COLORS = ['b', 'g', 'r', 'k', 'orange', 'violet',  'olive', 'crimson', 'teal'] 


def pyplot_reset(tex=True):
    pyplot.cla()
    pyplot.clf()
    pyplot.close()

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)    
    matplotlib.rcParams.update({'font.size': 18})
    
    if not tex: return

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42     
    matplotlib.rcParams['text.usetex'] = True  
    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{amssymb}']    
    
    
def savefig(path, **args):
    logging.info("Saving plot to %s" % path)    
    #assert path not in savefig.paths, "Overwritting %s" % path
    if path in savefig.paths:
        logging.warn("Overwritting %s" % path)
    savefig.paths.add(path)
    
    #pyplot.savefig(path, bbox_inches = 'tight', pad_inches = 0, **args)
    pyplot.savefig(path, pad_inches = 0, **args)
    
    if args.pop("reset", True): pyplot_reset()
savefig.paths = set()    


def get_saved_paths():
    return savefig.paths


pyplot_savefig = savefig


def date2month(date):
    return datetime.date(date.year, date.month, 1)


def gen_dates(start_date, end_date):
    day_count = (end_date - start_date).days + 1
    return list(single_date for single_date in 
                [d for d in (start_date + datetime.timedelta(n) for n in range(day_count)) if d <= end_date])


def get_time_bins_borders(min_date, max_date, bucket_calc=date2month):
    return sorted(set(map(bucket_calc, gen_dates(min_date, max_date))))


def get_time_bins_borders2(dates, bucket_calc=date2month):
    return get_time_bins_borders(min(dates), max(dates), bucket_calc)


def get_bucket_stats(xs, ys, xbucket_calc = lambda v: math.floor(v / 7.0)):    
    bucket2ys = {}
    bucket2xs = {}
    for x, y in zip(xs, ys):
        bucket = xbucket_calc(x)
        bucket2ys.setdefault(bucket, []).append(y)
        bucket2xs.setdefault(bucket, []).append(x)
    for x in range(int(min(xs)), int(np.ceil(max(xs)))):
        bucket = xbucket_calc(x)
        bucket2xs.setdefault(bucket, []).append(x)
    
    xs, means, medians, stds, counts = [], [], [], [], []
    for bucket in sorted(bucket2xs):
        xs.append( min(bucket2xs[bucket]) )
        y = bucket2ys.get(bucket, [])
        counts.append(len(y))
        if len(y)>0:
            means.append(np.mean(y))
            medians.append(np.median(y))
            stds.append(np.std(y))
        else:
            means.append(0)
            medians.append(0)
            stds.append(0)            
    return xs, means, medians, stds, counts #xs = left border



def align_y_axis(ax1, ax2, minresax1=None, minresax2=None):
    """ Sets tick marks of twinx axes to line up with 7 total tick marks

    ax1 and ax2 are matplotlib axes
    Spacing between tick marks will be a factor of minresax1 and minresax2"""

    ax1ylims = ax1.get_ybound()
    ax2ylims = ax2.get_ybound()
    
    if minresax1 is None: minresax1=(ax1ylims[1]-ax1ylims[0])/10
    if minresax2 is None: minresax2=(ax2ylims[1]-ax2ylims[0])/10
    
    ax1factor = minresax1 * 6
    ax2factor = minresax2 * 6
    ax1.set_yticks(np.linspace(ax1ylims[0],
                               ax1ylims[1]+(ax1factor -
                               (ax1ylims[1]-ax1ylims[0]) % ax1factor) %
                               ax1factor,
                               7))
    ax2.set_yticks(np.linspace(ax2ylims[0],
                               ax2ylims[1]+(ax2factor -
                               (ax2ylims[1]-ax2ylims[0]) % ax2factor) %
                               ax2factor,
                               7))


def pandas_get_buckets_means_stds_counts_per_group(df, group_column, value_column, 
                                  bucket_calc = lambda v: math.floor(v / 7.0)):    
    bucket2counts = {}
    for n, (_, group) in enumerate(df.groupby(group_column)):
        g_bucket2count = collections.Counter(group[value_column].apply(bucket_calc))
        for bucket, count in g_bucket2count.items():
            bucket2counts.setdefault(bucket, []).append(count)
    n += 1
    #n = len(df[group_column].unique())
    
    bucket2mean = dict((b, sum(counts) / n) for (b, counts) in bucket2counts.items())
    bucket2std = {}
    for b, counts in bucket2counts.items():
        m = bucket2mean[b]
        s = sum((c - m) * (c - m) for c in counts) + sum(m * m for c in range(n - len(counts)))
        bucket2std[b] = math.sqrt(s)    
    return bucket2mean, bucket2std


def pandas_get_buckets_means_stds_values(df, bucket_column, value_column, 
                                  bucket_calc = lambda v: math.floor(v / 7.0)):    
    bucket2values = {}
    for bucket_value, value in zip(df[bucket_column], df[value_column]):
        bucket = bucket_calc(bucket_value)
        bucket2values.setdefault(bucket, []).append(value)
    bucket2mean = dict((b, np.mean(values)) for b, values in bucket2values.items())
    bucket2std = dict((b, np.std(values)) for b, values in bucket2values.items())
    return bucket2mean, bucket2std


def pyplot_parse_params(kwargs):
    if "title" in kwargs:
        pyplot.title(kwargs.pop("title", ""))
    if "xlabel" in kwargs:
        pyplot.xlabel(kwargs.pop("xlabel", ""))
    if "ylabel" in kwargs:
        pyplot.ylabel(kwargs.pop("ylabel", ""))
    if "grid" in kwargs:
        pyplot.grid(bool(kwargs.pop("grid", True)))
    
    legend_loc = kwargs.pop("legend_loc", 0)
    if bool(kwargs.pop("legend", True)):
        pyplot.legend(loc=legend_loc)
    else:
        if pyplot.gca().legend() is not None:
            pyplot.gca().legend().set_visible(False)
        
    if "xmin" in kwargs or "xmax" in kwargs:
        xmin, xmax = pyplot.gca().get_xlim()
        xmin2 = kwargs.pop("xmin", xmin)
        xmax2 = kwargs.pop("xmax", xmax)
        if xmin2 is not None: 
            try:    xmin = float(xmin2)
            except: xmin = xmin2
        if xmax2 is not None: 
            try:    xmax = float(xmax2)
            except: xmax = xmax2
        pyplot.xlim( (xmin, xmax) )    
    
    if "ymin" in kwargs or "ymax" in kwargs:
        ymin, ymax = pyplot.gca().get_ylim()
        ymin2 = kwargs.pop("ymin", ymin)
        ymax2 = kwargs.pop("ymax", ymax)
        if ymin2 is not None: 
            try:    ymin = float(ymin2)
            except: ymin = ymin2
        if ymax2 is not None: 
            try:    ymax = float(ymax2)
            except: ymax = ymax2
        pyplot.ylim( (ymin, ymax) )      

    if "xlim" in kwargs:
        pyplot.gca().set_xlim(kwargs.pop("xlim"))
    if "ylim" in kwargs:
        pyplot.gca().set_ylim(kwargs.pop("ylim"))
    
    if "xlog" in kwargs and bool(kwargs.pop("xlog"))==True:
        pyplot.xscale("log")
    if "ylog" in kwargs and bool(kwargs.pop("ylog"))==True:
        pyplot.yscale("log")
        
    return kwargs


def pyplot_parse_params2(**kwargs):
    pyplot_parse_params(kwargs)


def scatter_plot(x, y, **kwargs):
    equality_line = kwargs.pop("equality_line", True)
    show = kwargs.pop("show", True)
    cmap = kwargs.pop("cmap", "Blues")
    edgecolors = kwargs.pop("edgecolors", "none")
    pyplot_parse_params(kwargs)        
        
    sns.kdeplot(np.array(x), np.array(y), shade=True, shade_lowest=False, cmap=cmap)
    pyplot.scatter(x, y, edgecolors=edgecolors, **kwargs)
    
    if equality_line:
        a = min(x+y)
        b = max(x+y)
        pyplot.plot([a, b], [a, b], "k-", lw=2)    
        pyplot.text((a+b)*0.25, (a+b)*0.25, "y = x", 
                    transform=pyplot.gca().transData, fontsize=10, color="black",
                    verticalalignment='center', horizontalalignment="center", 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))  

    if show:
        pyplot.show()

 
def discrete_density_plot(values, offset=0, **kwargs): 
    show = kwargs.pop("show", True)
    axis = kwargs.pop("axis", pyplot.gca())
    color = kwargs.pop("color", "r")
    pyplot_parse_params(kwargs)

    value2weight = dict((v+offset, float(c)/len(values)) 
                        for v, c in collections.Counter(values).items())

    xs = sorted(value2weight.keys())
    ys = list(map(lambda x: value2weight[x], xs))
    axis.plot(xs, ys, marker='o', color=color, ms=0.01, mec='r', lw=0)
    axis.vlines(xs, 0, ys, colors=color, lw=10, **kwargs)

    if show:
        pyplot.show()
    
    

def densities_plot(label2values, **kwargs):
    show = kwargs.pop("show", True)
    axis = kwargs.pop("axis", pyplot.gca())
    color = kwargs.pop("color", None)
    sns.set_style('whitegrid')    
    
    for i, (label, values) in enumerate(sorted(label2values.items())):
        c = color if color is not None else COLORS[i%len(COLORS)]
        sns.kdeplot(np.array(values), label=label, ax=axis, color=c, **kwargs)

    pyplot_parse_params(kwargs)
    if show:
        pyplot.show()
    

def density_plot(values, **kwargs):
    label = kwargs.pop("label", "data")
    densities_plot({label: values}, **kwargs)


def compare_densities_plot(values1, values2, **kwargs):    
    label1 = kwargs.pop("label1", "data1")
    label2 = kwargs.pop("label2", "data2")
    densities_plot({label1: values1, label2: values2}, **kwargs)



def plot_count_over_time(times, 
                         bucket_calc=date2month, 
                         label_formatter=lambda t: str(t)[:7], 
                         gen_dates = gen_dates,
                         **kwargs):
    """Plots temporal profiles."""  
    show = kwargs.pop("show", True)
    drawstyle = kwargs.pop("drawstyle", "steps-mid")
    pyplot_parse_params(kwargs) 

    start_date, end_date = min(times), max(times)
    start_bucket, end_bucket = bucket_calc(start_date), bucket_calc(end_date)

    bucket2count = Counter(map(bucket_calc, times))     
    bucket2count = dict( filter(lambda bv: bv[0]>=start_bucket and bv[0]<=end_bucket, bucket2count.items()) )        
    for bucket in set(map(bucket_calc, gen_dates(start_date, end_date))):
        bucket2count.setdefault(bucket, 0)
                    
    xs = sorted(bucket2count.keys())
    ys = list(map(lambda x: bucket2count[x], xs))
    p1 = pyplot.plot(xs, ys, drawstyle=drawstyle, **kwargs)
    #pyplot.gcf().autofmt_xdate()
    pyplot.xticks(xs, list(map(label_formatter, xs)), rotation='vertical')
    pyplot.grid(True)
     
    if show:
        pyplot.show()
        
    return p1[0]


def plot_hist(values, **kwargs):
    show = kwargs.pop("show", True)
    bins = kwargs.pop("bins", 50)
    normed = kwargs.pop("normed", 50)    
    binsize = kwargs.pop("binsize", None)
    if binsize is not None:
        bins = list(range(int(math.floor(min(values))), int(math.ceil(max(values))) + binsize, binsize))
    pyplot_parse_params(kwargs) 
            
    n,bins,patches = pyplot.hist(values, bins=bins, normed=normed, **kwargs)
    
    #y1, y2 = pyplot.gca().get_ylim()
    #pyplot.gca().set_ylim((y1, max(n)*1.05))
    
    if show:
        pyplot.show()
    
    return n, bins, patches
        
        