#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Fits the robust (bayesian with priors) survival model under H1."""

import sys
sys.path.append("../")

import argparse
import pandas as pd
import numpy

from aux.events_io import extract_processes_times
from aux.sharing import VAR

from datetime import datetime
import processes.fit_bayes as model 
 
import logging


LOG_DBGINFO = 15


def run_fitting(args):
    logging.info("Parsed args: %s" % args) 

    indata = args.input   
    verbose = args.verbose or args.debug

    process_constructor_args = {}
    if args.switch_time is not None: process_constructor_args["switch_time"] = args.switch_time
    if args.max_time is not None: process_constructor_args["max_time"] = args.max_time
    if args.start_time is not None: process_constructor_args["start_time"] = args.start_time 
    if not args.verbose: process_constructor_args["verbose"] = 0 
    if args.debug: process_constructor_args["verbose"] = 3
    logging.log(LOG_DBGINFO, "process constructor args: %s" % process_constructor_args)

    ###########################################################################
    
    logging.log(LOG_DBGINFO, "[%s] reading data from %s" % (str(datetime.now()), indata))    
    df = VAR[indata[7: ]] if indata.startswith("shared_") else pd.read_csv(indata, sep="\t")  
    ids = sorted(pd.unique(df["id"]))
    
    if verbose:
        logging.log(LOG_DBGINFO, " #users=%s" % len(ids))    
        max_times, switch_times, start_times, execution_times = extract_processes_times(df)
        logging.log(LOG_DBGINFO, " #not executed actions=%s" % sum(execution_times>max_times))
        logging.log(LOG_DBGINFO, " #executed before switching=%s" % sum(execution_times<=switch_times))
        logging.log(LOG_DBGINFO, " #executed after switching=%s" % sum(numpy.logical_and(execution_times>switch_times, execution_times<=max_times)))
        logging.log(LOG_DBGINFO, " #started before switching=%s" % sum(start_times<switch_times)) #start_times==switch_times means that started after switching 
        logging.log(LOG_DBGINFO, " #started after switching=%s" % sum(numpy.logical_and(start_times>=switch_times, start_times<=max_times)))
        logging.log(LOG_DBGINFO, " start_times:\n  "+str(pd.Series(start_times).describe()).replace("\n", "\n  "))
        logging.log(LOG_DBGINFO, " execution_times:\n  "+str(pd.Series(execution_times).describe()).replace("\n", "\n  "))

    ###########################################################################
    
    logging.log(LOG_DBGINFO, "[%s] preparing processes" % str(datetime.now()))
    id2p = model.build_processes(df, **process_constructor_args)
    processes = list(id2p.values())
    
    processes0  = [p for p in processes if p.present_before_switching()]
    processes1  = [p for p in processes if p.present_after_switching()]
    logging.info(" %i processes before switching, %i after switching" % (len(processes0), len(processes1)))

    ###########################################################################


    logging.log(LOG_DBGINFO, "[%s] fitting t<badge for %i processes" % (str(datetime.now()), len(processes0)))
    lifetimes0  = [p.lifetime1() for p in processes0]
    executions0 =  [int(p.when_executed()==-1) for p in processes0]
    
    #assert args.lambda0 is not None, "For fitting two-part bayes survival lambda0 must be given!"
    if args.lambda0 is None:
        _, lambda0 = model.fit_params(lifetimes0, executions0, mode=args.lambda_fitting_mode)
    else:
        lambda0 = args.lambda0        
    r0  = model.fit_r0(lifetimes0, executions0, lambda0)
    ll0 = model.calc_ll(lifetimes0, executions0, r0, lambda0)
    
    ###########################################################################
    
    logging.log(LOG_DBGINFO, "[%s] fitting t>badge for %i processes" % (str(datetime.now()), len(processes1)))
    lifetimes1  = [p.lifetime2() for p in processes1]
    executions1 =  [int(p.when_executed()==+1) for p in processes1]
    
    if args.lambda0 is None:        
        _, lambda1 = model.fit_params(lifetimes1, executions1, mode=args.lambda_fitting_mode)
    else:
        lambda1 = args.lambda0
    r1 = model.fit_r0(lifetimes1, executions1, lambda1)
    ll1 = model.calc_ll(lifetimes1, executions1, r1, lambda1)

    ###########################################################################
    
    logging.info("r0=%f l0=%f ll0=%f r1=%f l1=%f ll1=%f ll=%f" % (r0, lambda0, ll0, r1, lambda1, ll1, (ll0+ll1)))
    return r0, lambda0, ll0, r1, lambda1, ll1
    


def main(argv):
    parser = argparse.ArgumentParser(description="""Fits the robust (bayesian with priors) survival model under H1.""")

    parser.add_argument("-i", "--input", dest='input', help="input TSV file with samples (three cols: id, time, type).",
                        metavar="input.tsv", required=True)


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
                    
    parser.add_argument("-l", "--lambda", dest='lambda0', type=float,
                         help="lambda value",  
                         required=False, default=None)  
    parser.add_argument("-mode", "--lambda_mode", "--lambda_fitting_mode", "--fitting_mode",
                        dest='lambda_fitting_mode', type=int,
                        help="lambda fitting mode",  
                        required=False, default=None) 

    args, _ = parser.parse_known_args(argv)    
    if args.verbose: logging.basicConfig(level=logging.INFO)
    if args.debug:   logging.basicConfig(level=logging.DEBUG)
    
    #previously if None was set: automatically args.lambda_fitting_mode was 0
    assert args.lambda0 is not None or args.lambda_fitting_mode is not None,  "One of: lambda0 or lambda_fitting_mode must be set!"
    assert not (args.lambda0 is not None and args.lambda_fitting_mode is not None), "Only one parameter out of lambda0 or lambda_fitting_mode should be set!"
        
    ###########################################################################    
    
    return run_fitting(args)
        

if __name__=="__main__":
    main(sys.argv[1: ])
    