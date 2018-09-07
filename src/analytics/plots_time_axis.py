#!/usr/bin/python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot
import matplotlib.dates as mdates
import sys

sys.path.append("../")
from analytics import time_convert


totime = lambda t: time_convert.convert(t).to_pydatetime()


def generate_ticks(pmin, pmax, fmt='%Y-%m'):
    pmin, pmax = int(pmin), int(pmax)
    formatter = (lambda t: t.strftime(fmt)) if type(fmt)==str else fmt 
    positions, labels = [], []
    for p in range(pmin-100, pmax+100):
        l = formatter( totime(p) )
        if len(labels)==0 or labels[-1]!=l:
            positions.append(p)
            labels.append(l)
    return positions[1: ], labels[1: ]
    
    
def formatter(pmin, pmax):
    if pmax-pmin>400:
        return '%Y'
    elif pmax-pmin>200:
        def f(t):
            y = t.strftime('%Y')
            m = t.strftime('%m')
            m2m = {"01":"01", "02":"01", "03":"01",
                  "04":"04", "05":"04", "06":"04",
                  "07":"07", "08":"07", "09":"07",
                  "10":"10", "11":"10", "12":"10"}
            return "%s-%s" % (y, m2m.get(m, m))
        return f
    else: 
        return '%Y-%m'
    
    
def set_xaxis():    
    pmin, pmax = pyplot.xlim()    
    positions, labels = generate_ticks(pmin, pmax, formatter(pmin, pmax))    
    ax = pyplot.gca().xaxis
    ax.set_ticks(positions)
    ax.set_ticklabels(labels)    
    pyplot.xlim(pmin, pmax)


#totime = lambda t: time_convert.convert(t).to_datetime()
#totimes = lambda ts: list(map(totime, ts))
def set_xaxis2():    
    xmin, xmax = pyplot.xlim()
    ax = pyplot.gca().xaxis
    if xmax-xmin>400:
        ax.set_major_locator(mdates.YearLocator())
        ax.set_minor_locator(mdates.MonthLocator())
        ax.set_major_formatter(mdates.DateFormatter('%Y'))
    else:
        if xmax-xmin>200:
            ax.set_major_locator(mdates.MonthLocator([1,4,7,10]))            
        else:
            ax.set_major_locator(mdates.MonthLocator())
            ax.set_minor_locator(mdates.DayLocator())
        ax.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    #pyplot.gcf().autofmt_xdate()
    
    
def set_yaxis():
    pmin, pmax = pyplot.ylim()
    positions, labels = generate_ticks(pmin, pmax, formatter(pmin, pmax))    
    ax = pyplot.gca().yaxis
    ax.set_ticks(positions)
    ax.set_ticklabels(labels)
    pyplot.ylim(pmin, pmax)
    
    