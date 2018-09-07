#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" Model of a survival process with hazard rate a1+trend*time until switching point and a2+trend*time after that. """

import sys
sys.path.append("../")

import numpy
import pandas as pd
import logging


INF = float("inf")



class TrendSurvivalProcess:
    """
    Survival process with hazard rate a1+trend*time until switching point and a2+trend*time after that.
    """
    
    def __init__(self, max_time=None, start_time=None, 
                 switch_time=0.0, execution_time=INF, 
                 a1=None, a2=None, trend=None,
                 verbose=2, environment=numpy):
        assert start_time is not None, "start_time is None"
        assert max_time is not None, "max_time is None"
        assert a1 is not None, "a1 is None"
        assert a2 is not None, "a2 is None"
        assert trend is not None, "trend is None"
        assert start_time<=max_time, "start_time=%f max_time=%f" % (start_time, max_time)
        assert start_time!=INF, "start_time is INF"
        assert max_time!=INF, "max_time is INF"
        assert start_time<=execution_time, "executed before started"
        
        self._start_time = start_time       
        self._max_time = max_time             
        self._switch_time = switch_time   
        self._execution_time = execution_time

        self._a1 = a1
        self._a2 = a2
        self._trend = trend
                
        self._max_intensity = max(a1*(1+self._trend*self._max_time), a2*(1+self._trend*self._max_time)) #max(a1+trend*self._max_time, a2+trend*self._max_time)
        self._verbose = verbose        
        self._loginf(self)
        
    @classmethod  
    def fmt(self, txt):
        return " ".join( p for p in str(txt).replace("\n", " ").split(" ") if len(p)>0 )
          
    def _logwrn(self, txt):
        if self._verbose >= 1: 
            logging.warn("[TrendSurvivalProcess][WRN] %s" % self.fmt(txt))

    def _loginf(self, txt):
        if self._verbose >= 2: 
            logging.info("[TrendSurvivalProcess][INF] %s" % self.fmt(txt))

    def _logdbg(self, txt):
        if self._verbose >= 3: 
            logging.debug("[TrendSurvivalProcess][DBG] %s" % self.fmt(txt))
                
    def __str__(self):
        return ("""start_time=%f switch_time=%f max_time=%f 
                    a1=%s a2=%s trend=%s
                    max_intesity=%f execution_time=%s""" % 
                    (self._start_time, self._switch_time, self.max_time(), 
                     self._a1, self._a2, self._trend, self.max_intensity(),
                     self._execution_time))
                
    ###########################################################################                
                
    def max_time(self):
        return self._max_time
    
    def max_intensity(self):
        return self._max_intensity  
        
    def update(self, execution_time, params=None):
        assert execution_time==INF or self._execution_time==INF      
        self._execution_time = execution_time
    
    def intensity(self, t, params=None):
        assert t>=0 and t<=self._max_time 
        if t<self._start_time:      return 0.0
        if t>self._execution_time:  return 0.0
        if t<=self._switch_time:    return self._a1*(1.0 + self._trend*t) 
        return self._a2*(1.0 + self._trend*t) 
                            
    ###############################################################################################
                     
    def get_data(self):
        times, actions = self.get_events()                
        return pd.DataFrame({"time": times, "type": actions})
    
    
    def get_events(self):
        times, actions = [], []
        
        def append(times_subset, type_str):
            times.extend(times_subset)
            actions.extend(type_str for _ in times_subset)
        
        append([self._execution_time], "action")        
        append([self._start_time], "start_time")
        append([self._switch_time], "switch_time")
        append([self._max_time], "max_time")
                
        return times, actions
        
    
    
