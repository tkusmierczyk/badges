#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Calculation of p-values from Fisher exact test on counts (before badge vs after badge)."""

import sys
sys.path.append("../")

import argparse
import pandas as pd
from scipy.stats import fisher_exact

from aux.events_io import load_events
from aux.sharing import VAR
import logging


def count_survived_before_after_switching(df, switch_time, testing_period):
    inf = float("inf")
    logging.info("parsing user events")
    id2events = load_events(df, verbose=True)    
    
    a, b, c, d = 0, 0, 0, 0
    for i, events in enumerate(id2events.values()):
        if i%500000 == 0: logging.debug("%i/%i" % (i, len(id2events)))
            
        assert len(events["start_time"])==1            
        start_time = events["start_time"][0]
        
        assert len(events["max_time"])==1
        max_time = events["max_time"][0]
        
        actions = events.get("action", [inf]) 
        assert len(actions)==1
        action = actions[0]
        
        if (start_time < switch_time and start_time > switch_time - testing_period) or (start_time > max_time - testing_period):
            continue
        
        if start_time < switch_time:
            if action <= start_time + testing_period:
                a += 1
            else:
                c += 1
        elif action <= start_time + testing_period:
            b += 1
        else:
            d += 1
    
    return a, b, c, d



#if __name__=="__main__":
#    argv = sys.argv[1: ]

def fisher_exact_test(df, switch_time, testing_period):
    a, b, c, d = count_survived_before_after_switching(df, switch_time, testing_period)
    p = fisher_exact([[a, b], [c, d]])[1]
    ids = sorted(pd.unique(df["id"]))
    logging.info("skipped=%i" % (len(ids) - (a + b + c + d)))
    logging.info("a=%i #start<switching fired" % a)
    logging.info("b=%i #start>switching fired" % b)
    logging.info("c=%i #start<switching not fired" % c)
    logging.info("d=%i #start>switching not fired" % d)
    logging.info("p=%.10f" % p)
    return a, b, c, d, p



def main(argv):
    parser = argparse.ArgumentParser(description="""Calculates p-values from Fisher exact test on counts 
                                                    (before badge vs after badge).""")

    parser.add_argument("-i", "--input", dest='input', 
                        help="input TSV file with samples (three cols: id, time, type).",
                        metavar="input.tsv", required=True)
        
    parser.add_argument("-st", "--switch_time", dest='switch_time', type=float, 
                        required=True, default=None,
                         help="switching time between different intensities")    
    parser.add_argument("-tp", "--testing_period", dest='testing_period', type=float,
                         help="how many time units we wait after the start_time to check if action happened",  
                         required=True, default=None)    
                        
    args = parser.parse_args(argv)    
    logging.info("Parsed args: %s" % args)
    switch_time = args.switch_time
    testing_period = args.testing_period
    
    logging.basicConfig(level=logging.INFO)
    
    ###########################################################################    
            
    logging.info("reading data from %s" % args.input)
    df = VAR[args.input[7: ]] if args.input.startswith("shared_") else pd.read_csv(args.input, sep="\t") 
    ids = sorted(pd.unique(df["id"]))
    logging.info(" %i users identified" % len(ids))

    ###########################################################################    
    
    a, b, c, d, p = fisher_exact_test(df, switch_time, testing_period)
    
    ###########################################################################
    
    return p
    

if __name__=="__main__":
    main(sys.argv[1: ])
    



