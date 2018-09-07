#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Model of a survival process with one hazard rate until switching point and another after that."""

import sys
sys.path.append("../")

from aux.events_io import extract_process_times


import numpy
import pandas as pd
import cvxpy            
import copy
import logging


INF = float("inf")



class OneSurvivalProcess:
    """
    Survival process with hazard rate a1 until switching point and a2 after that.
    """
    
    def __init__(self, max_time=None, start_time=None, 
                 switch_time=0.0, execution_time=INF, 
                 a1=cvxpy.Variable(name="a1"), 
                 a2=cvxpy.Variable(name="a2"),
                 verbose=2, environment=numpy):
        assert start_time is not None, "start_time is None"
        assert max_time is not None, "max_time is None"        
        assert start_time<=max_time, "start_time=%f max_time=%f" % (start_time, max_time)
        assert start_time!=INF, "start_time is INF"
        assert max_time!=INF, "max_time is INF"
        assert start_time<=execution_time, "executed before started"
        
        self._start_time = start_time       
        self._max_time = max_time             
        self._switch_time = switch_time #min(max(switch_time, start_time), max_time) #bound switch time to [start_time, max_time]   
        
        self._a1 = a1
        self._a2 = a2
        
        self._op = cvxpy if type(a1)==cvxpy.Variable or type(a2)==cvxpy.Variable else environment
        self._execution_time = copy.copy(execution_time)

        self._max_intensity = max(a1, a2) if self._op==numpy else 10.0
        self._verbose = verbose        
        self._loginf(self)
        
    @classmethod  
    def fmt(self, txt):
        return " ".join( p for p in str(txt).replace("\n", " ").split(" ") if len(p)>0 )
          
    def _logwrn(self, txt):
        if self._verbose >= 1: 
            logging.warn("[OneSurvivalProcess][WRN] %s" % self.fmt(txt))

    def _loginf(self, txt):
        if self._verbose >= 2: 
            logging.info("[OneSurvivalProcess][INF] %s" % self.fmt(txt))

    def _logdbg(self, txt):
        if self._verbose >= 3: 
            logging.debug("[OneSurvivalProcess][DBG] %s" % self.fmt(txt))
                
    def __str__(self):
        return ("""start_time=%f switch_time=%f max_time=%f 
                    a1=%s a2=%s 
                    max_intesity=%f execution_time=%s
                    lt1=%s lt2=%s when_exec=%i""" % 
                    (self._start_time, self._switch_time, self.max_time(), 
                     self._a1, self._a2, self.max_intensity(),
                     self._execution_time,
                     self.lifetime1(), 
                     self.lifetime2(), self.when_executed()))
                
    ###########################################################################                
                
    def max_time(self):
        return self._max_time
    
    def max_intensity(self):
        return self._max_intensity  
        
    def update(self, execution_time, params=None):
        assert execution_time==INF or self._execution_time==INF      
        self._execution_time = execution_time
    
    def lifetime1(self):
        """Returns lifetime before switching time or None."""
        if self._switch_time < self._start_time: return None
        return min([self._execution_time, self._switch_time, self._max_time]) - min(self._switch_time, self._start_time)
    
    def lifetime2(self):
        """Returns lifetime after switching time or None."""
        if self._execution_time <= self._switch_time: return None
        return max(min(self._max_time, self._execution_time), self._switch_time) - max(self._start_time, self._switch_time)

    def when_executed(self):
        """Returns -1 if execution time is before switching time 
           and 1 if after and 0 if censored or invalid (execution time < start time)."""
        if self._execution_time<self._start_time:      return 0
        if self._execution_time>self._max_time:        return 0
        if self._execution_time<=self._switch_time:    return -1
        return +1
    
    def present_before_switching(self):
        if self._start_time == self._switch_time: logging.warn("WARNING: OneSurvivalProcess: _start_time==_switch_time")
        return self._start_time <= self._switch_time
    
    def present_after_switching(self):
        return self._execution_time > self._switch_time
            
    def intensity_selector(self, t, params=None):
        assert t>=0 and t<=self._max_time 
        if t<self._start_time:      return 0
        if t>self._execution_time:  return 0
        if t<=self._switch_time:     return -1
        #else (t>self._switch_time)
        return +1
    
    def intensity(self, t, params=None):
        assert t>=0 and t<=self._max_time 
        if t<self._start_time:      return 0.0
        if t>self._execution_time:  return 0.0
        if t<=self._switch_time:     return self._a1
        #else (t>self._switch_time)
        return self._a2
                                        
    def calc_ll(self, execution_time): 
        assert self._max_time!=self._start_time #TODO 
        assert self._start_time<execution_time, "start_time=%f < execution_time=%f" % (self._start_time, execution_time)
        ##assert execution_time<=self._max_time, "execution_time=%f <= self._max_time=%f" % (execution_time, self._max_time)
             
        closing_time = min(self._max_time, execution_time)
        if self._switch_time>self._start_time:
            if closing_time<=self._switch_time:
                integral = (self._start_time - closing_time) * self._a1
            else: #closing_time>self._switch_time
                integral = (self._start_time - self._switch_time) * self._a1 + \
                           (self._switch_time - closing_time) * self._a2
        else: #_switch_time<=_start_time
            integral = (self._start_time - closing_time) * self._a2

        if execution_time<=self._max_time:
            return self._op.log(self.intensity(execution_time)) + integral 
        else:
            return integral            
    
    def ll(self):    
        return self.calc_ll(self._execution_time)

    def ll_a1_weight(self):
        lt = self.lifetime1()
        if lt is None: return 0
        return  -lt
        #integral = 0
        #closing_time = min(self._max_time, self._execution_time)
        #if self._switch_time>self._start_time:
        #    if closing_time<=self._switch_time:
        #        integral = (self._start_time - closing_time)
        #    else: #closing_time>self._switch_time
        #        integral = (self._start_time - self._switch_time)
        #else: #_switch_time<=_start_time
        #    integral = 0
        #return integral                      
                    
    def ll_a2_weight(self):
        lt = self.lifetime2()
        if lt is None: return 0
        return  -lt
        #integral = 0
        #closing_time = min(self._max_time, self._execution_time)
        #if self._switch_time>self._start_time:
        #    if closing_time<=self._switch_time:
        #        integral = 0
        #    else: #closing_time>self._switch_time
        #        integral = (self._switch_time - closing_time)
        #else: #_switch_time<=_start_time
        #    integral = (self._start_time - closing_time) 
        #return integral         


    def ll_loga1_weight(self):
        """returns 0/1"""
        return int(self._execution_time<=self._max_time and
                   #self.intensity(self._execution_time) is self._a1) 
                   self.intensity_selector(self._execution_time)==-1)
        
    def ll_loga2_weight(self):
        """returns 0/1"""
        return int(self._execution_time<=self._max_time and 
                   #self.intensity(self._execution_time) is self._a2)
                   self.intensity_selector(self._execution_time)==+1) 
    
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
        
    
    
    @classmethod
    def create_from_data(self, df, **constructor_args):  
        """
            Creates a process from dataframe.
        
            Parsed (known) constructor args: max_time, switch_time, start_time.
        """
        
        constructor_args = copy.copy(constructor_args)
        max_time, switch_time, start_time, execution_time = extract_process_times(df, constructor_args)
        
        p = OneSurvivalProcess(max_time=max_time, switch_time=switch_time, 
                               start_time=start_time, execution_time=execution_time,
                               **constructor_args)
        return p

    @classmethod
    def create_from_events(self, events, **constructor_args):  
        """
            Creates a process from events dictionary {event_type: [list of times]}.
        
            Parsed (known) constructor args: max_time, switch_time, start_time.
        """
        
        
        assert len(events.get("max_time", []))<=1
        max_time = events.get("max_time", [INF])[0]
        
        assert len(events.get("switch_time", []))<=1
        switch_time = events.get("switch_time", [INF])[0]

        assert len(events.get("start_time", []))<=1
        start_time = events.get("start_time", [INF])[0]
        
        #assert len(events.get("action", []))<=1
        #if len(events.get("action", []))>1:
        #    logging.warn("WARNING: more than one action found in the data! the first will be used!")
        execution_time = min(events.get("action", [INF]))
    
        constructor_args = copy.copy(constructor_args)
        max_time    = constructor_args.pop("max_time", max_time)
        switch_time = constructor_args.pop("switch_time", switch_time)
        start_time  = constructor_args.pop("start_time", start_time)
                                    
        p = OneSurvivalProcess(max_time=max_time, switch_time=switch_time, 
                               start_time=start_time, execution_time=execution_time,
                               **constructor_args)
        return p
