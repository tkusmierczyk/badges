#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy
import logging

INF = float("inf")


def load_events(df, id2events=None, verbose=False):
    if id2events is None: 
        id2events = dict()
        
    for i, row in enumerate(df.itertuples()):
        if verbose and i>0 and i%500000==0: logging.debug("[load_events] %i/%i" % (i, len(df)))
        identifier, time, event_type = row.id, row.time, row.type
        id2events.setdefault(identifier, dict()).setdefault(event_type, []).append(time)
        
    return id2events


def load_file_events(input_path, id2events=None):
    df = pd.read_csv(input_path, sep="\t")
    return load_events(df, id2events)  


def store_events(id2events):
    ids, times, types = [], [], []
    for i, events in id2events.items():
        for type_val, time_vals in events.items():
            for t in time_vals:
                ids.append(i)
                types.append(type_val)
                times.append(t)    
    return pd.DataFrame({"id": ids,  "time": times, "type": types}) 


def store_file_events(output, id2events):
    logging.debug(" converting events to data frame")
    df = store_events(id2events)
    
    logging.debug(" writing a data frame")
    #df.to_csv(output, sep="\t", index=False)
    output = open(output, "w")
    output.write("id\ttime\ttype\n")
    for i, t, tp in zip(df["id"], df["time"], df["type"]):
        output.write("%i\t%g\t%s\n" % (i, t, tp))    
    output.close()


###################################################################################################
###################################################################################################
###################################################################################################


def extract_process_times(df, force_values = {}):
    max_time = None
    switch_time = 0
    start_time = 0 
    execution_time = INF
    
    for event_type, time in zip(df["type"], df["time"]):
        if event_type=="max_time":
            max_time = time
        elif event_type=="switch_time":
            switch_time = time
        elif event_type=="start_time":
            start_time = time
        elif event_type=="action":
            execution_time = min(time, execution_time)    
    
    max_time    = force_values.pop("max_time", max_time)
    switch_time = force_values.pop("switch_time", switch_time)
    start_time  = force_values.pop("start_time", start_time)
        
    return max_time, switch_time, start_time, execution_time


def extract_processes_times(df):
    ids = sorted(pd.unique(df["id"]))
    id2ix = dict((identifier, ix) for ix, identifier in enumerate(ids))    
    n = len(ids)
    max_times, start_times = numpy.zeros(n), numpy.zeros(n)
    switch_times, execution_times = numpy.zeros(n), (numpy.zeros(n)+INF)
    
    for row in df.itertuples():
        identifier = row.id
        time = row.time
        event_type = row.type
        i = id2ix[identifier]
    
        if event_type=="max_time":
            max_times[i] = time
        elif event_type=="switch_time":
            switch_times[i] = time
        elif event_type=="start_time":
            start_times[i] = time
        elif event_type=="action":
            execution_times[i] = min(time, execution_times[i])
    
    return max_times, switch_times, start_times, execution_times
