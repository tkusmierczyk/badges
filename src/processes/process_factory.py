#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")

import argparse
import numpy as np
from processes.process import OneSurvivalProcess

import logging


LOG_DBGINFO = 15
EPS = np.nextafter(0,1)


class OneSurvivalProcessFactory:
    """
    A factory that produces processes. 
    """
    
    def __init__(self, args=""):
        parser = argparse.ArgumentParser() 
        parser.add_argument("-n", "--num_processes", dest='n', 
                            help="number of processes to be created", 
                            type=int, metavar="n", required=False, default=1000)        
              
        parser.add_argument("-mt", "--max_time", dest='max_time', 
                            help="max time (necessary for integrating)",
                            type=float, metavar="max_time", required=True)
        parser.add_argument("-st", "--switch_time", dest='switch_time', 
                            help="time to switch between different hazard rates", 
                            type=float, metavar="switch_time", required=False, default=0.0)        
        parser.add_argument("-sf", "--switch_fraction", dest='switch_fraction', 
                            help="faction of users who 'switch' their hazard rates", 
                            type=float, metavar="switch_fraction", required=False, default=1.0)        
        
        parser.add_argument("-a1", "--alpha1", dest='a1', 
                            help="hazard rate until the switching moment", 
                            type=float, metavar="a1", required=False, default=None)        
        parser.add_argument("-a2", "--alpha2", dest='a2', 
                            help="hazard rate after the switching moment", 
                            type=float, metavar="a2", required=False, default=None)
        
        parser.add_argument("-rand", "--randomize", dest='randomize', 
                            help="if flag set then hazard rates will be drawn from gamma dist", 
                            required=False, default=False, action="store_true")        
        
        parser.add_argument("-r1", "--r1", dest='r1', 
                            help="if -rand set: hazard rate 1 ~ Gamma(r1, lambda1)", 
                            type=float, required=False, default=0.5)        
        parser.add_argument("-l1", "--lambda1", dest='lambda1', 
                            help="if -rand set: hazard rate 1 ~ Gamma(r1, lambda1)", 
                            type=float, required=False, default=1000)        
        
        parser.add_argument("-r2", "--r2", dest='r2', 
                            help="if -rand set: hazard rate 2 ~ Gamma(r2, lambda2)", 
                            type=float, required=False, default=50)        
        parser.add_argument("-l2", "--lambda2", dest='lambda2', 
                            help="if -rand set: hazard rate 2 ~ Gamma(r2, lambda2)", 
                            type=float, required=False, default=10000)        
        
        parser.add_argument("-zt", "--start_min_time", dest='start_min_time', 
                            help="if this parameter is set then starting times are sampled ~U[start_min_time, start_max_time)",
                            type=float, metavar="min_time", required=False, default=0.0)
        parser.add_argument("-zx", "--start_max_time", dest='start_max_time', 
                            help="if this parameter is set then starting times are sampled ~U[start_min_time, start_max_time)",
                            type=float, metavar="start_max_time", required=False, default=float("inf"))        
        
        #parser.add_argument("-p", "--parameters", dest='parameters',
        #                    help="a TSV file with columns <a,b,alpha1,mean1,std1,alpha2,mean2,std2, etc.>",
        #                    metavar="parameters.tsv", required=True)
        
        parser.add_argument("-v", "--verbose", dest='verbose', help="print additional (debug) info", 
                        required=False, default=False, action='store_true')
        
        help_msg = parser.format_help()        
        logging.log(LOG_DBGINFO, ("\n======================[OneSurvivalProcessFactory arguments description]=======================\n%s" + 
                                  "\n======================[/OneSurvivalProcessFactory arguments description]======================") % help_msg)
        
        args = parser.parse_args(args.split())        
        logging.log(LOG_DBGINFO, "[OneSurvivalProcessFactory] Parsed args: %s" % args)
        self._params = args
        self._verbose = 3 if args.verbose else 2
        
        assert args.randomize or args.a1 is not None, "a1 must be randomly drawn with -rand or given explicitly with -a1"
        assert args.randomize or args.a2 is not None, "a2 must be randomly drawn with -rand or given explicitly with -a2"
        
        
    def yield_start_times(self):
        while True:
            yield np.random.uniform(low=max(0, self._params.start_min_time), 
                                    high=min(self._params.max_time, self._params.start_max_time), 
                                    size=1)[0]
        
    def yield_processes(self):
        switching_count = 0
        nonswitching_count = 0
            
        start_time_generator = self.yield_start_times()
        for _ in range(self._params.n):
            total_count = switching_count+nonswitching_count
            switching_fraction = 0 if total_count==0 else float(switching_count) / total_count
            if self._params.verbose and total_count%100==0:
                logging.info("[OneSurvivalProcessFactory] %i processes. %.2f switching hazard rate" % 
                      (total_count, switching_fraction))
            verbose = self._verbose if total_count<10 else 1
            
            if self._params.randomize:
                a1 = float(np.random.gamma(shape=self._params.r1, scale=1.0/self._params.lambda1, size=1))
                a2 = float(np.random.gamma(shape=self._params.r2, scale=1.0/self._params.lambda2, size=1))                
            else:
                a1 = self._params.a1
                a2 = self._params.a2
                
            if a1<=0:   a1 = EPS
            if a2<=0:   a2 = EPS
                
            if switching_fraction<=self._params.switch_fraction:
                switching_count += 1
                yield OneSurvivalProcess(max_time=self._params.max_time, 
                                         switch_time=self._params.switch_time, 
                                         start_time=next(start_time_generator),                                     
                                         a1=a1, a2=a2, 
                                         verbose=verbose)
            else:
                nonswitching_count += 1
                yield OneSurvivalProcess(max_time=self._params.max_time, 
                                         switch_time=self._params.switch_time, 
                                         start_time=next(start_time_generator),                                     
                                         a1=a1, a2=a1, 
                                         verbose=verbose)
        
        logging.info("[OneSurvivalProcessFactory] %i processes. %.2f switching hazard rate" % 
                      (switching_count+nonswitching_count, switching_fraction))
