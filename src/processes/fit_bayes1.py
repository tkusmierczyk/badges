#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Fits the robust (bayesian with priors) survival model under H0."""

import sys
sys.path.append("../")

import argparse
import pandas as pd

from aux.sharing import VAR
from datetime import datetime
import processes.fit_bayes as model 

import logging


LOG_DBGINFO = 15
       

def run_fitting(args):
    logging.info("Parsed args: %s" % args) 

    indata = args.input   
    
    process_constructor_args = {}
    process_constructor_args["switch_time"] = float("inf")
    if args.max_time is not None: process_constructor_args["max_time"] = args.max_time
    if args.start_time is not None: process_constructor_args["start_time"] = args.start_time 
    if not args.verbose: process_constructor_args["verbose"] = 0 
    if args.debug: process_constructor_args["verbose"] = 3
    logging.info("process constructor args: %s" % process_constructor_args)

    ###########################################################################
    
    logging.log(LOG_DBGINFO, "[%s] reading data from %s" % (str(datetime.now()), indata))    
    df = VAR[indata[7: ]] if indata.startswith("shared_") else pd.read_csv(indata, sep="\t")  
    #ids = sorted(pd.unique(df["id"]))

    ###########################################################################
    
    logging.log(LOG_DBGINFO, "[%s] preparing processes" % str(datetime.now()))
    id2p = model.build_processes(df, **process_constructor_args)
    processes = list(id2p.values())
    lifetimes = [p.lifetime1() for p in processes]
    executions =  [int(p.when_executed()!=0) for p in processes]
    logging.info(" %i executed out out %i " % (sum(executions), len(executions)))
        
    logging.log(LOG_DBGINFO, "[%s] fitting parameters (l0=%s fitting_mode=%s)" % 
                (str(datetime.now()), args.lambda0, args.lambda_fitting_mode))
    if args.lambda0 is None:
        _, lambda0 = model.fit_params(lifetimes, executions, mode=args.lambda_fitting_mode)
    else:
        lambda0 = args.lambda0
    
    r0 = model.fit_r0(lifetimes, executions, lambda0)
    ll = model.calc_ll(lifetimes, executions, r0, lambda0)
    
    logging.info("r0=%f l0=%f ll=%f" % (r0, lambda0, ll))
    return r0, lambda0, ll 
    

def main(argv):
    parser = argparse.ArgumentParser(description="""Fits the robust (bayesian with priors) survival model under H0.""")

    parser.add_argument("-i", "--input", dest='input', 
                        help="input TSV file with samples (three cols: id, time, type).",
                        metavar="input.tsv", required=True)

    parser.add_argument("-v", "--verbose", dest='verbose', help="print additional info", 
                        required=False, default=False, action='store_true')
    parser.add_argument("-d", "--debug", dest='debug', help="print additional debug info", 
                        required=False, default=False, action='store_true')
         
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
    if args.verbose: logging.basicConfig(level=LOG_DBGINFO)
    if args.debug:   logging.basicConfig(level=logging.DEBUG)
    
    #previously if None was set: automatically args.lambda_fitting_mode was 0
    assert args.lambda0 is not None or args.lambda_fitting_mode is not None,  "One of: lambda0 or lambda_fitting_mode must be set!"
    assert not (args.lambda0 is not None and args.lambda_fitting_mode is not None), "Only one parameter out of lambda0 or lambda_fitting_mode should be set!"
    
    ###########################################################################    
    
    return run_fitting(args)
        

if __name__=="__main__":
    main(sys.argv[1: ])
    
