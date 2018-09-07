#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Sampling from non-homogenous Poisson temporal point processes with known maximum."""

import sys
import pandas as pd
import argparse
import random
import numpy
import inspect

from importlib.machinery import SourceFileLoader
import logging


def yield_poisson_process1(intensity, T):
    N = numpy.random.poisson(intensity*T)
    return sorted(numpy.random.uniform(low=0.0, high=T, size=N))


def yield_poisson_process2(intensity, T):
    t = random.expovariate(intensity)
    while t<T:
        yield t
        t += random.expovariate(intensity)        

                
def yield_point_process(intensity_function, update_status, max_intensity_value, T, 
                        yield_poisson_process=yield_poisson_process1):
    for t in yield_poisson_process(max_intensity_value, T):
        
        intensity_value = intensity_function(t)
        assert intensity_value<=max_intensity_value, \
                "intensity_value=%f max_intensity_value=%f" % (intensity_value,max_intensity_value)
        
        if random.uniform(0, max_intensity_value) <= intensity_value:
            yield t
            update_status(t)
               

def merge_samples2(processes):
    df = None
    for i, p in enumerate(processes):
        p_df = p.get_data()
        p_df["id"] = i
        df = p_df if df is None else df.append(p_df)    
    cols = sorted(df.columns.tolist())
    cols.remove("id")
    df = df[["id"] + cols]
    return df


def merge_samples(processes):
    times, types, ids = [], [], []
    for i, p in enumerate(processes):
        try:
            p_times, p_types = p.get_events()
            times.extend(p_times)
            types.extend(p_types)
            ids.extend(i for _ in range(len(p_times)))
        except AttributeError as e:
            logging.warn("WARNING: Process does not support get_events():", e)
            p_df = p.get_data() #was to slow
            times.extend(p_df["time"])
            types.extend(p_df["type"])
            ids.extend(i for _ in range(len(p_df)))            
    df = pd.DataFrame({"id": ids, "type": types, "time": times})
    df = df[["id", "time", "type"]]
    return df


#if __name__=="__main__":
#    argv = sys.argv[0]
def run(argv):
    parser = argparse.ArgumentParser(description="""
    Samples from non-homogenous Poisson temporal point processes with known maximum.""")
    parser.add_argument("-d", "--description", dest='description',
                        help="""python file containing a point process factory
                         (a class that generates objects with the following functions: 
                         max_time(), max_intensity(), intensity(t))  ... (TODO)
                         Unparsed here command line arguments will be passed to the constructor of the factory. 
                         """,
                        metavar="process_description.py", required=True)
    parser.add_argument("-o", "--output", dest='output', 
                        help="output TSV file with samples",
                        metavar="output.tsv", required=False, default=None)
    parser.add_argument("-l", "--loglikelihood", dest='loglikelihood', 
                        help="output TSV file with loglikelihood",
                        metavar="loglikelihood.tsv", required=False, default=None)
    parser.add_argument("--seed", dest='seed', 
                        help="seed value for random generators",
                        required=False, default=123, type=int)
    parser.add_argument("-vv", "--verbose", dest='verbose', 
                        help="verbose", 
                        required=False, default=False, action="store_true")    
    args, unknown_args = parser.parse_known_args(argv)    
    process_args = " ".join(unknown_args)
    if args.verbose:
        fmt = '[%(process)4d][%(asctime)s][%(levelname)-5s][%(module)s:%(lineno)d/%(funcName)s] %(message)s'        
        logging.basicConfig(level=logging.INFO, format=fmt) 
        logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Parsed args: %s" % args)    
    logging.info("Unparsed args: %s" % process_args)    
            
    random.seed(args.seed)
    numpy.random.seed(args.seed)

    ####################################################################    
    
    logging.info("preparing a process factory from the description file %s" % args.description)
    process_module = SourceFileLoader("process", args.description).load_module()
    classes = [obj for name, obj in inspect.getmembers(process_module) if inspect.isclass(obj) and "Factory" in name]
    if len(classes)!=1:
        logging.error("Exactly one <*Factory> class expected in the process generator file!") 
        sys.exit(-1)
    process_factory = classes[0](process_args)
    
    ####################################################################        
    
    logging.info("constructing processes")
    processes = list(process_factory.yield_processes()) 
    
    logging.info("generating samples for %i processes" % len(processes))
    for i, p in enumerate(processes):
        if len(processes)<=20 or i%(len(processes)//20)==0: logging.debug(" %i/%i" % (i, len(processes)))
        list(yield_point_process(p.intensity, p.update, p.max_intensity(), p.max_time()))
                                                    
    ####################################################################  
          
    logging.info("merging samples")
    samples = merge_samples(processes)
    if args.output is not None:
        print("saving samples to %s" % args.output)
        samples.to_csv(args.output, sep="\t", index=False)
    
    ####################################################################  
                  
    if args.loglikelihood is not None:
        logging.info("saving likelihoods to %s" % args.loglikelihood)
        df = pd.DataFrame()
        df["ll"] = pd.Series(p.ll() for p in processes)
        df.to_csv(args.loglikelihood, sep="\t", index=False)
        logging.info("ll=%f" % df["ll"].sum())
    
    ####################################################################  

    return {"samples": samples, "processes": processes}


def main(argv):
    output = run(argv)
    return output["samples"]
    
    
if __name__=="__main__":
    main(sys.argv[1: ])
    
    
    