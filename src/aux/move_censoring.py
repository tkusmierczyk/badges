#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Simulates right censoring of the data."""

import sys
sys.path.append("../")

import argparse
import numpy
import random

import pandas as pd
from aux.events_io import load_events, store_file_events, store_events

import logging


INF = float("inf")


def move_censoring2(df, args): 
    print("updating %i/%i actions" % 
          (sum(df.loc[df["type"]=="action", "time"]>=args.max), 
           len(df.loc[df["type"]=="action", "time"]) ))
    df.loc[df["type"]=="action", "time"] = df.loc[df["type"]=="action", "time"].apply(lambda v: v if v<=args.max else float("inf"))
    print(df.head())
    
    if not args.keep_inf:
        print("removing inf from %i actions" % sum(df["type"]=="action"))
        df = df[(df["type"]!="action") | ((df["type"]=="action") & (df["time"]!=float("inf")))]
        print("%i actions left" % sum(df["type"]=="action"))


    print("updating max_time")
    df.loc[df["type"]=="max_time", "time"] = df.loc[df["type"]=="max_time", "time"].apply(lambda v: min(v,args.max))
    print(df.head())

    ###########################################################################
    
    print("filtering out start_time>args.max")
    id2events = {}
    for i, events in load_events(df).items():
        if events["start_time"][0]>args.max: 
            continue
        id2events[i] = events
    
    ###########################################################################
    
    print("storing to %s" % args.output)
    store_file_events(args.output, id2events)
    
    
def move_censoring(df, mint=-INF, maxt=+INF, keep_inf=True):
    logging.info("censoring: mint=%f maxt=%f keep_inf=%s" % (mint, maxt, keep_inf))
    valid_ids = set(i for i, tp, t in zip(df["id"], df["type"], df["time"]) 
                        if tp=="start_time" and t>=mint and t<=maxt)
    logging.info("%i valid ids selected out of %i" % (len(valid_ids), len(df["id"].unique())))
    
    ids, types, times = [], [], []
    for i, tp, t in zip(df["id"], df["type"], df["time"]):
        if i not in valid_ids: continue
        if tp=="max_time":  t = min(t, maxt)
        elif tp!="kernel_time" and tp!="kernel_progress":               
            t = t if t<=maxt else INF
        if t==INF and not keep_inf: continue
        
        ids.append(i)
        types.append(tp)
        times.append(t)

    logging.info("%i rows kept out of %i" % (len(ids), len(df)))        
    df = pd.DataFrame({"id": ids, "time": times, "type": types})
    
    logging.info("dealing 1with types: kernel_time and kernel_progess")
    id2events = load_events(df)
    counter = 0
    for i, events in id2events.items():
        if "kernel_time" not in events: continue
        times, progress = [], []
        for t, p in zip(events["kernel_time"], events["kernel_progress"]):
            if t>maxt: continue
            times.append(t)
            progress.append(p)
        if counter<10: 
            logging.debug(" max progress: old=%f new=%f" % (max(events["kernel_progress"]), max(progress+[-INF])))
            counter += 1
        events["kernel_time"], events["kernel_progress"] = times, progress
            
    df = store_events(id2events)
    return df[ ["id", "time", "type"] ]


def move_censoring1(df, args):
    return move_censoring(df, mint=args.min, maxt=args.max, keep_inf=args.keep_inf)


def main(argv):
    parser = argparse.ArgumentParser(description="Filters a data subset fitting into time window.")

    parser.add_argument("-i", "--input", dest='input', 
                        help="a list of input TSV files with samples (three cols: id, time, type).",
                        required=True)

    parser.add_argument("-min", "--min_time", dest='min', 
                        help="min start time",
                        required=False, type=float, default=-INF)
    
    parser.add_argument("-max", "--max_time", dest='max', 
                        help="new censoring",
                        required=False, type=float, default=+INF)

    parser.add_argument("-a", "--action", "--skip_inf", dest='keep_inf', 
                        help="set this flag to skip actions with time==inf",
                        required=False, action='store_false', default=True) 
                                
    parser.add_argument("-o", "--output", dest='output', 
                        help="output TSV file",
                        required=True)
    
    args = parser.parse_args(argv)    
    print("Parsed args: %s" % args)
    
    ###########################################################################    
        
    random.seed(123)
    numpy.random.seed(123)
        
    ###########################################################################
        
    print("reading data from %s" % args.input)
    df = pd.read_csv(args.input, sep="\t")
    print(df.head())

    ###########################################################################
    
    df = move_censoring1(df, args)
    
    print("writing to %s" % args.output)
    df.to_csv(args.output, sep="\t", index=False, header=True)

    
if __name__=="__main__":
    main(sys.argv[1: ])


