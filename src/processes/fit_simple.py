#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Fits the simple (intensities shared between users) survival model in two cases.
   First, assuming intensity to be constant (H0) and second, allowing for intensity change around certain time point (H1)."""

import sys
sys.path.append("../")

import argparse
import pandas as pd
import numpy

from processes.process import OneSurvivalProcess 
from aux.events_io import extract_processes_times, load_events
from aux.sharing import VAR

from datetime import datetime
import logging
 
 
LOG_DBGINFO = 15
INF = float("inf")


def build_processes(df, **constructor_args):
    logging.log(LOG_DBGINFO, " parsing events")
    id2events = load_events(df, verbose=True)
    id2p = {}
    logging.log(LOG_DBGINFO, " constructing processes")    
    for i, (identifier, events) in enumerate(id2events.items()):
        if i%500000==0: logging.debug("%i/%i" % (i, len(id2events)) )
        if i>20: constructor_args["verbose"] = 1
        id2p[identifier] = OneSurvivalProcess.create_from_events(events, **constructor_args)
    return id2p


def fit_hazard_rates(id2p, f=1.0):
    ll_a1_weight = sum(p.ll_a1_weight() for p in id2p.values())
    ll_a2_weight = sum(p.ll_a2_weight() for p in id2p.values())
    ll_loga1_weight = sum(p.ll_loga1_weight() for p in id2p.values())
    ll_loga2_weight = sum(p.ll_loga2_weight() for p in id2p.values())
    a1 = (ll_loga1_weight + (1 - f) * ll_loga2_weight) / -(ll_a1_weight + (1 - f) * ll_a2_weight) if (ll_a1_weight + (1 - f) * ll_a2_weight) < 0 else 0
    #a2 = (f*ll_loga2_weight) / -(f*ll_a2_weight) if (f*ll_a2_weight)<0 else 0
    a2 = (ll_loga2_weight) / -(ll_a2_weight) if (ll_a2_weight) < 0 else 0
    
    #ll = 0 if a1 > 0 or a2 > 0 else float("nan")
    if a1<=0 and a2<=0: return 0.0, 0.0, 0.0
    
    ll = 0.0
    if a1 > 0:
        ll += numpy.log(a1) * ll_loga1_weight + a1 * ll_a1_weight + (1 - f) * (a1 * ll_a2_weight + numpy.log(a1) * ll_loga2_weight)
    if a2 > 0:
        ll += f * (a2 * ll_a2_weight + numpy.log(a2) * ll_loga2_weight)
    
    return a1, a2, ll


def run_fitting(args):
    logging.info("Parsed args: %s" % args)
        
    indata = args.input   
    output_path = args.output   
    verbose = args.verbose or args.debug
    f = args.switching_fraction
    assert f>=0 and f<=1.0
    
    process_constructor_args = {}
    if args.switch_time is not None: process_constructor_args["switch_time"] = args.switch_time
    if args.hypothesis0: process_constructor_args["switch_time"] = INF #force estimation of a1 over all data points
    if args.max_time is not None: process_constructor_args["max_time"] = args.max_time
    if args.start_time is not None: process_constructor_args["start_time"] = args.start_time 
    if not args.verbose: process_constructor_args["verbose"] = 0 
    if args.debug: process_constructor_args["verbose"] = 3
    logging.info("process constructor args: %s" % process_constructor_args)

    ###########################################################################
    
    logging.log(LOG_DBGINFO, "[%s] reading data from %s" % (str(datetime.now()), indata))    
    df = VAR[indata[7: ]] if indata.startswith("shared_") else pd.read_csv(indata, sep="\t")  
    ids = sorted(pd.unique(df["id"]))
    
    if verbose:
        logging.log(LOG_DBGINFO, " #users=%s" % len(ids))    
        max_times, switch_times, start_times, execution_times = extract_processes_times(df)
        if args.switch_time is not None:
            switch_times = args.switch_time
        logging.log(LOG_DBGINFO, " #not executed actions=%s" % sum(execution_times>max_times))
        logging.log(LOG_DBGINFO, " #executed before switching=%s" % sum(execution_times<=switch_times))
        logging.log(LOG_DBGINFO, " #executed after switching=%s" % sum(numpy.logical_and(execution_times>switch_times, execution_times<=max_times)))
        logging.log(LOG_DBGINFO, " #started before switching=%s" % sum(start_times<switch_times)) #start_times==switch_times means that started after switching 
        logging.log(LOG_DBGINFO, " #started after switching=%s" % sum(numpy.logical_and(start_times>=switch_times, start_times<=max_times)))
        logging.log(LOG_DBGINFO, " start_times:\n  "+str(pd.Series(start_times).describe()).replace("\n", "\n  "))
        logging.log(LOG_DBGINFO, " execution_times:\n  "+str(pd.Series(execution_times).describe()).replace("\n", "\n  "))

    ###########################################################################
    
    logging.log(LOG_DBGINFO, "[%s] calculating ll" % str(datetime.now()))
    id2p = build_processes(df, **process_constructor_args)

    logging.log(LOG_DBGINFO, "[%s] summing weights and estimating alphas" % str(datetime.now()))
    a1, a2, ll = fit_hazard_rates(id2p, f)
            
    #a1 is estimated over all the data points anyways by forcing switching_time=inf above
    #therefore we do not need to do this here and we can simply overwrite a2 (that would otherwise be 0)
    if args.hypothesis0: a2 = a1 
    
    ###########################################################################
    
    if output_path is not None:
        logging.info("saving params to %s" % output_path)    
        df = pd.DataFrame()
        df["id"] = ids 
        df["a1"] = a1
        df["a2"] = a2
        df.to_csv(output_path, sep="\t", index=False)

    ###########################################################################    
                
    logging.info("a0=%f a1=%f ll=%f" % (a1, a2, ll))
    
    return a1, a2, None, ll 
    


def main(argv):
    parser = argparse.ArgumentParser(description="""
    Fits the simple (intensities shared between users) survival model in two cases.
    First, assuming intensity to be constant (H0) and second, allowing for intensity change around certain time point (H1).
    """)

    parser.add_argument("-i", "--input", dest='input', help="input TSV file with samples (three cols: id, time, type).",
                        metavar="input.tsv", required=True)
    parser.add_argument("-o", "--output", dest='output', help="output TSV file with fitted parameters",
                        metavar="output.tsv", required=False, default=None)

    parser.add_argument("-h0", "--hypothesis0", 
                        dest='hypothesis0', 
                        help="force a2==a1", 
                        action='store_true', 
                        default=False, required=False)

    parser.add_argument("-v", "--verbose", dest='verbose', help="print additional info", 
                        required=False, default=False, action='store_true')
    parser.add_argument("-d", "--debug", dest='debug', help="print additional debug info", 
                        required=False, default=False, action='store_true')
        
    parser.add_argument("-st", "--switch_time", dest='switch_time', type=float,
                         help="set (force) time of switching between different intensities",  required=False, default=None)    
    parser.add_argument("-mt", "--max_time", dest='max_time', type=float,
                         help="set (force) max time value",  required=False, default=None)    
    parser.add_argument("-tt", "-zt", "--start_time", dest='start_time', type=float,
                         help="set (force) start time value",  required=False, default=None)
                    
    parser.add_argument("-f", "--fraction", dest='switching_fraction', type=float,
                         help="what fraction in (0, 1] of users switch from alpha0 -> alpha1 after switching time",  
                         required=False, default=1.0) 
                    

    parser.add_argument("-m", "--mode", dest='mode', type=int,
                         help="IGNORED",  
                         required=False, default=0) #need to be here -> otherwise -m is interpreted as -mt 

    args, _ = parser.parse_known_args(argv)    
    
    if args.verbose: logging.basicConfig(level=LOG_DBGINFO)
    if args.debug:   logging.basicConfig(level=logging.DEBUG)
        
    ###########################################################################    
    
    return run_fitting(args)
        

if __name__=="__main__":
    main(sys.argv[1: ])
    