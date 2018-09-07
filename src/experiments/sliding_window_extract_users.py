#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Extracts users appearing in each of the (sliding) window."""

import pandas as pd
import argparse
import os
import logging
from multiprocessing import Pool
import numpy as np

import sys
sys.path.append("../")

from aux.sharing import VAR
from aux import parsing 

from aux.move_censoring import move_censoring

    
INF = float("inf")


def print2(txt):
    logging.info(txt)
    print("[%s] %s" % (os.getpid(), txt))
    

def simulation(params):
    df = params["df"]
    t = params["t"]
    
    endt = t + params["window"]
    window_center = 0.5 * (t + endt)
    
    print2("extracting for t=%s" % window_center)
    sdf = move_censoring(df, t, endt, keep_inf=True)
    VAR["sdf"] = sdf
    num_badges_within_window = sum([1 for b in params["badges"] if b > t + params["margin"] and b < endt - params["margin"]])
    assert num_badges_within_window <= 1, "more than one badge captured within a window! (model will not fit)"
    alt_model = num_badges_within_window > 0
    num_badges_until_window = sum([1 for b in params["badges"] if b < endt - params["margin"]])
    badge_no = num_badges_until_window * int(alt_model) - 1
    #badge_center_distance = abs(badge_time - window_center) #dist between badge and window center
    badge_time = min([b for b in params["badges"] if b > t + params["margin"] and b < endt - params["margin"]] + [INF])
    
    
    users = list(sdf["id"].unique())
    return (window_center, badge_time, badge_no, users)
    

def prepare_configurations(df, args, params):

    maxt = int(max(df[df["time"]<INF]["time"]))
    print2("max time = %i" % maxt)
    print2("badges = %s" % args.badges)
    print2("window = %s" % args.window)
    print2("margin = %s" % args.margin)
    print2("step = %s" % args.step)
    
    windows = range(args.start, min(args.end, maxt-args.window), args.step)
    configurations = []
    for t in windows:
        cparams = params.copy()
        cparams["df"] = df
        cparams["t"] = t 
        cparams["margin"] = args.margin
        cparams["window"] = args.window
        cparams["badges"] = args.badges
        configurations.append(cparams)
    
    return configurations    


def setup(args):
    params = parsing.parse_dictionary(args.params)
    print("params: %s" % params)
    
    try:    params["randomize"] = args.randomize
    except: pass        
    try:    verbose = args.verbose
    except: verbose = False    
    try:    debug = args.debug
    except: debug = False    
    
    np.random.seed(123)
    level = logging.DEBUG if debug else logging.INFO            
    if args.output is None:
        args.output = "%s_%i.tsv" % (os.path.basename(__file__), os.getpid())    
    
    fmt = '[%(process)4d][%(asctime)s][%(levelname)-5s][%(module)s:%(lineno)d/%(funcName)s] %(message)s'
    if args.output is not None:
        logfile = "%s.log" % (args.output)
        logging.basicConfig(filename=logfile, level=level, format=fmt)
        print("logging to %s\n" % logfile)
    else:
        logging.basicConfig(level=level, format=fmt)
    
    if verbose:
        logging.getLogger().addHandler(logging.StreamHandler())
    
    return params



def main(argv):

    parser = argparse.ArgumentParser(description="""Extracts users appearing in each of the (sliding) window.""")
    
    parser.add_argument("-i", "--input", dest='input', 
                        help="a list of input TSV files with samples (three cols: id, time, type).",
                        required=True)    
    parser.add_argument("-o", "--output", dest='output', 
                        help="output file", 
                        required=False, default=None)        
    parser.add_argument("-p", "--params", dest='params', 
                        help="comma-separated params: option=value", 
                        required=False, default="")    

    parser.add_argument("-v", "--verbose", dest='verbose', 
                        help="verbose", 
                        required=False, default=False, action="store_true")
    parser.add_argument("-d", "--debug", dest='debug', 
                        help="debug", 
                        required=False, default=False, action="store_true")
    parser.add_argument("-c", "--cpu", dest='cpu', 
                        help="num processes to use", 
                        required=False, type=int, default=1)           

    parser.add_argument("-z", "--start", dest='start', type=int, help="start time",  
                         required=False, default=0)    
    parser.add_argument("-e", "--end", dest='end', type=int, help="end time",  
                         required=False, default=100000)    
    parser.add_argument("-w", "--window", dest='window', type=int, help="window size",  
                         required=False, default=90)
    parser.add_argument("-m", "--margin", dest='margin', type=int, help="window margin",  
                         required=False, default=0)
    parser.add_argument("-s", "--step", dest='step', type=int,
                        help="how much the sliding window should be moved each time",  
                         required=False, default=5)        
        
    parser.add_argument("-b", "--badges", dest='badges', 
                         help="badge introduction times",  
                         required=False, default=[], type=float, nargs="+",)       
            
    args = parser.parse_args(argv)    
    params = setup(args)
    

    #########################################################################    
    #########################################################################
    
    print2("Reading data from %s" % args.input)
    df = pd.read_csv(args.input, sep="\t")
    print2(df.head())
    print2(" %i ids loaded" % len(df["id"].unique()))

    #########################################################################
    
    print2("Running sliding windows")
    configurations = prepare_configurations(df, args, params)
    pool = Pool(args.cpu)
    data = [simulation(c) for c in configurations] if args.cpu<=1 else pool.map(simulation, configurations)
        
    #########################################################################

    print2("Saving results to %s" % (args.output))
    f = open(args.output, "w")
    f.write("time\tbadge_time\tbadge_no\tusers\n")
    for time, badge_time, badge, users in data:
        f.write("%s\t%s\t%s\t%s\n" % (time, badge_time, badge, ",".join(map(str, users))))
    
    #########################################################################


if __name__=="__main__":
    main(sys.argv[1: ])    
    
    