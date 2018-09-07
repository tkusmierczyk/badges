#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" Calculates p-values from various models 
    (Fisher=counts, Simple=simple survival, Robust='bayesian' with priors) 
    on real world data using sliding window.
    Outputs a TSV file containing the test statistics for each time point."""

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

from processes.fit_simple import main as fit
from processes.fit_bayes1 import main as fit1
from processes.fit_bayes2 import main as fit2

from testing.fisher_test import fisher_exact_test
from testing.wilks_test import test


    
INF = float("inf")
COLUMNS = [ "wc", "t", "endt", "size", "alt_model", "badge_no", "badge_time", "prev_badges",
            "FHf_a", "FHf_b", "FHf_c", "FHf_d", "FHfH0_pval", 
            "FH1_a", "FH1_b", "FH1_c", "FH1_d", "FH1H0_pval", 
            "SH0_a0", "SH0_a1", "SH0_ll", 
            "SHf_a0", "SHf_a1", "SHf_ll", "SHfH0_pval", "SHfH0_LLR",
            "SH1_a0", "SH1_a1", "SH1_ll", "SH1H0_pval", "SH1H0_LLR",
            "BH0_r0", "BH0_lambda0", "BH0_ll", "BH0_a0", 
            "BHf_r0", "BHf_lambda0", "BHf_r1", "BHf_lambda1", "BHf_ll", "BHf_a0", "BHf_a1", "BHfH0_pval", "BHfH0_LLR",
            "BH1_r0", "BH1_lambda0", "BH1_r1", "BH1_lambda1", "BH1_ll", "BH1_a0", "BH1_a1", "BH1H0_pval", "BH1H0_LLR" ]


def print2(txt):
    logging.info(txt)
    print("[%s] %s" % (os.getpid(), txt))
    

def simulation(params):
    df = params["df"]
    t = params["t"]
    lambda0 = params.get("lambda", None)
    
    endt = t + params["window"]
    window_center = 0.5 * (t + endt)
    
    print2("[0] Preprocessing")
    sdf = move_censoring(df, t, endt, keep_inf=True)
    VAR["sdf"] = sdf
    num_badges_within_window = sum([1 for b in params["badges"] if b > t + params["margin"] and b < endt - params["margin"]])
    assert num_badges_within_window <= 1, "more than one badge captured within a window! (model will not fit)"
    alt_model = num_badges_within_window > 0
    badge_time = min([b for b in params["badges"] if b > t + params["margin"] and b < endt - params["margin"]] + [INF])
    num_badges_until_window = sum([1 for b in params["badges"] if b < endt - params["margin"]])
    badge_no = num_badges_until_window * int(alt_model) - 1
    #badge_center_distance = abs(badge_time - window_center) #dist between badge and window center
    row = [window_center, t, endt, len(sdf["id"].unique()), 
        int(alt_model), badge_no, badge_time, num_badges_until_window]
    
    print2("[1] Fisher test on counts")
    #TODO testing period should be given as an argument
    Hf_a, Hf_b, Hf_c, Hf_d, Hf_fisher_p = fisher_exact_test(sdf, window_center, int(params["window"] * 0.25))
    row.extend([Hf_a, Hf_b, Hf_c, Hf_d, Hf_fisher_p])
    
    a, b, c, d, fisher_p = fisher_exact_test(sdf, badge_time, int(params["window"] * 0.25))
    row.extend([a, b, c, d, fisher_p])
    
    print2("[2] Simple survival model")
    H0_a0, H0_a1, _, H0_ll = fit(["-i", "shared_sdf", "-v", "-h0"])
    Hf_a0, Hf_a1, _, Hf_ll = fit(["-i", "shared_sdf", "-v", "-st", str(window_center)]) #fake badge
    H1_a0, H1_a1, _, H1_ll = fit(["-i", "shared_sdf", "-v", "-st", str(badge_time)]) #true badge if any exists
    row.extend([H0_a0, H0_a1, H0_ll, 
                Hf_a0, Hf_a1, Hf_ll, test(Hf_ll, H0_ll, 1), Hf_ll-H0_ll,
                H1_a0, H1_a1, H1_ll, test(H1_ll, H0_ll, 1), H1_ll-H0_ll])
    
    print2("[3] Robust survival model")    
    BH0_r0, BH0_lambda0, BH0_ll = fit1(["-i", "shared_sdf"] + 
        (["-mode", str(params.get("fitting_mode", "0"))] if lambda0 is None else ["-l", str(lambda0)]))
    BH0_a0 = float(BH0_r0) / BH0_lambda0

    BH1_r0, BH1_lambda0, Bll0, BH1_r1, BH1_lambda1, Bll1 = fit2(["-i", "shared_sdf", "-st", str(badge_time), "--lambda", str(BH0_lambda0)])
    BH1_ll, BH1_a0, BH1_a1 = Bll0 + Bll1, float(BH1_r0) / BH1_lambda0, float(BH1_r1) / BH1_lambda1
    
    BHf_r0, BHf_lambda0, Bll0, BHf_r1, BHf_lambda1, Bll1 = fit2(["-i", "shared_sdf", "-st", str(window_center), "--lambda", str(BH0_lambda0)])
    BHf_ll, BHf_a0, BHf_a1 = Bll0 + Bll1, float(BHf_r0) / BHf_lambda0, float(BHf_r1) / BHf_lambda1
    
    row.extend([BH0_r0, BH0_lambda0, BH0_ll, BH0_a0, 
                BHf_r0, BHf_lambda0, BHf_r1, BHf_lambda1, BHf_ll, BHf_a0, BHf_a1, test(BHf_ll, BH0_ll, 1), BHf_ll-BH0_ll, 
                BH1_r0, BH1_lambda0, BH1_r1, BH1_lambda1, BH1_ll, BH1_a0, BH1_a1, test(BH1_ll, BH0_ll, 1), BH1_ll-BH0_ll])
        
    print2("RESULTS[t=%s]: %s" % (window_center, " \t".join("%s=%g" % (col, val) for (col, val) in zip(COLUMNS, row))))
    return row



def prepare_configurations(df, args, params):

    maxt = int(max(df[df["time"]<INF]["time"]))
    print2("max time = %i" % maxt)
    print2("badges = %s" % args.badges)
    print2("window = %s" % args.window)
    print2("margin = %s" % args.margin)
    print2("step = %s" % args.step)
    
    if args.global_lambda:        
        VAR["sdf"] = df
        fitting_mode = str(params.get("fitting_mode", "0"))
        print2("fitting one global lambda over whole data: mode = %s" % fitting_mode)  
        _, params["lambda"], _ = fit1(["-i", "shared_sdf", "-mode", fitting_mode])        
        print2("fitting one global lambda over whole data set: lambda=%g" % params["lambda"])        

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

    parser = argparse.ArgumentParser(description="""
        Calculates p-values from various models 
        (Fisher=counts, Simple=simple survival, Robust='bayesian' with priors) 
        on real world data using sliding window.
        Outputs a TSV file containing the test statistics for each time point.""")
    
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
    parser.add_argument("-g", "--global_lambda", dest='global_lambda', 
                         help="fixes one lambda (in Flexible Survival Model) for all sliding windows",  
                         required=False, default=False, action="store_true")         
 
            
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
    data = pd.DataFrame(data).rename(columns= dict(enumerate(COLUMNS)))
    data.to_csv(args.output, sep="\t", index=False, header=True)
    
    #########################################################################


if __name__=="__main__":
    main(sys.argv[1: ])    