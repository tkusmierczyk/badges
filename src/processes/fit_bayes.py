#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Base functions for fitting of the robust (bayesian with priors) survival model.


    if mode==4:
        lambda0 = fit_lambda0_gridsearch1(lifetimes, executions, 
                    lambdas=[1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0],
                    calc_ll=calc_ll_cv)
    elif mode==3:
        lambda0 = fit_lambda0_gridsearch(lifetimes, executions, 
                                         pow_base=10, min_pow=-2, max_pow=10,
                                         calc_ll=calc_ll_cv, improvement=0.01)    
    elif mode==2:
        lambda0 = fit_lambda0_gridsearch(lifetimes, executions, 
                                         pow_base=10, min_pow=-2, max_pow=10,
                                         calc_ll=calc_ll_cv)    
        pw = math.log10(lambda0)
        lambda0 = fit_lambda0_bisearch(lifetimes, executions, 
                                       prec=0.00001, left=10**(pw-1), right=10**(pw+1),
                                       calc_ll=calc_ll_cv)
        lambda0 = fit_lambda0_grad(lifetimes, executions, lambda0, 
                                   ll_prec=0.0001, lambda_prec=0.0001, 
                                   max_iter_ca=10, max_iter_ga=20,
                                   eps=0.000000001, gamma=0.9, eta=0.1,
                                   calc_ll=calc_ll_cv)    
    elif mode==1:    
        #start = time.time()
        #lambda0 = fit_lambda0_bisearch(lifetimes, executions)
        lambda0 = fit_lambda0_gridsearch(lifetimes, executions)  
        lambda0 = fit_lambda0_grad(lifetimes, executions, lambda0)
        #elapsed1 = time.time()-start
    elif mode==0:

"""

import sys
sys.path.append("../")

import argparse
import pandas as pd
import numpy as np

from processes.process import OneSurvivalProcess 
from aux.events_io import extract_processes_times, load_events
from aux.sharing import VAR

import math
from matplotlib import pyplot
from datetime import datetime
from processes.fit_simple import fit_hazard_rates
 
import logging
import os



LOG_DBGINFO = 15


def build_processes(df, **constructor_args):
    logging.log(LOG_DBGINFO, " parsing events")
    id2events = load_events(df, verbose=True)
    id2p = {}
    logging.log(LOG_DBGINFO, " constructing processes")    
    for i, (identifier, events) in enumerate(id2events.items()):
        if i%500000==0: logging.debug("[build_processes] %i/%i" % (i, len(id2events)) )
        if i>20: constructor_args["verbose"] = 1
        id2p[identifier] = OneSurvivalProcess.create_from_events(events, **constructor_args)
    return id2p


#def calc_ll(lifetimes, executions, r0, lambda0, r_reg=0, lambda_reg=0):
#    s1 = r0 * sum( np.log(lambda0 / (lambda0 + lt)) for lt in lifetimes)
#    s2 = sum( np.log(r0 / (lambda0 + lt)) for lt, e in zip(lifetimes, executions) if e!=0)
#    return s1 + s2 - 2*r_reg*r0 - 2*lambda_reg*lambda0


def calc_ll(lifetimes, executions, r0, lambda0):
    assert len(lifetimes)==len(executions)
    if len(lifetimes)==0: return float("nan")
    
    s1 = r0 * sum( np.log(lambda0 / (lambda0 + lt)) for lt in lifetimes)
    s2 = sum( np.log(r0 / (lambda0 + lt)) for lt, e in zip(lifetimes, executions) if int(e)!=0)
    return s1 + s2 


def fit_r0(lifetimes, executions, lambda0):
    if sum(executions)==0: return 0.0
    
    nom, den = 0.0, 0.0
    for lt, e in zip(lifetimes, executions):
        nom += int(int(e)!=0)
        den += np.log(lambda0 / (lambda0 + lt))
    r0 = - nom / den
    #sanity checks
    #ll  = calc_ll(lifetimes, executions, r0, lambda0)
    #ll1 = calc_ll(lifetimes, executions, r0*0.5, lambda0)
    #assert ll>=ll1, "ll>=ll1 violated"
    #ll2 = calc_ll(lifetimes, executions, r0*2.0, lambda0)
    #assert ll>=ll2, "ll>=ll2 violated"
    #ll3 = calc_ll(lifetimes, executions, r0*100.0, lambda0)
    #assert ll>=ll3, "ll>=ll3 violated"
    #ll4 = calc_ll(lifetimes, executions, r0+0.1, lambda0)
    #assert ll>=ll4, "%f=ll>=ll4=%f violated" % (ll, ll4)
    #ll5 = calc_ll(lifetimes, executions, max(r0-0.1, 0.000001), lambda0)
    #assert ll>=ll5, "%f=ll>=ll5=%f violated" % (ll, ll5)    
    
    #if r0<=0:
    #    logging.warn("WARNING: r0=%g <= 0. CHANGING TO 0.000000001" % r0)
    #    r0 = 0.000000001
    assert r0>=0.0, "r0>=0.0 violated r0=%s" % r0    
    return r0


def chunks(l, n=10):
    l = list(l)
    step = int(math.ceil( float(len(l)) / n ))
    for i in range(0, len(l), step):
        yield l[i: min(i+step, len(l))]


def calc_ll_cv(lifetimes, executions, r0, lambda0, cv=5):
    """r0 is ignored"""
    assert len(lifetimes)==len(executions)
    if len(lifetimes)==0: return float("nan")
    
    parts = list(chunks(zip(lifetimes, executions), cv))
    cv = len(parts)
    lls = []
    for k in range(cv):
        train = [row for part in (parts[ :k] + parts[(k + 1): ]) for row in part]
        test = parts[k]
        
        l_train, e_train = list(zip(*train))
        l_test, e_test = list(zip(*test))
        
        r0 = fit_r0(l_train, e_train, lambda0)
        ll_test = calc_ll(l_test, e_test, r0, lambda0)
        lls.append(ll_test)
    logging.debug("CV LLs=%s" % lls)
    return np.mean(lls)
        
 
def verbose_print2(p, txt, pp=100):
    if p>0 and p%pp==0:
        logging.info(txt)
        
        
def dbg_print2(p, txt, pp=100):
    if p>0 and p%pp==0:
        logging.debug(txt)
        
        
def fit_lambda0_bisearch(lifetimes, executions, 
                         prec=0.0001, left=2**(-31), right=2**31,
                         calc_ll=calc_ll):    
    logging.log(LOG_DBGINFO, "started")
    right_ll = calc_ll(lifetimes, executions, fit_r0(lifetimes, executions, right), right)    
    while True:
        cent = (left + right) * 0.5        
        cent_ll = calc_ll(lifetimes, executions, fit_r0(lifetimes, executions, cent), cent)
        if right_ll < cent_ll:
            right = cent
            right_ll = cent_ll
        else:
            left = cent                    
        logging.debug("left=%f right=%f right_ll=%f cent_ll=%f" % 
                      (left, right, right_ll, cent_ll))    
        if abs(left - right) < prec: break
    lambda0 = cent
    logging.info("init with lambda0 = %f ll = %f" % (cent, cent_ll))
    return lambda0



def fit_lambda0_gridsearch1(lifetimes, executions, 
                            lambdas=[1.0, 10.0, 100.0, 1000.0, 10000.0], 
                            calc_ll=calc_ll, improvement=0):
    best_lambda0 = 1
    best_ll = -float("inf")
    logging.log(LOG_DBGINFO, "grid search started")
    for lambda0 in lambdas:
        r0 = fit_r0(lifetimes, executions, lambda0)
        ll = calc_ll(lifetimes, executions, r0, lambda0)
        mean = r0 / lambda0
        logging.debug("lambda0=%f r0=%f mean=%.10f ll=%f" % (lambda0, r0, mean, ll))
        if ll > best_ll * (1.0 - improvement):
            best_ll = ll
            best_lambda0 = lambda0
    
    logging.info("init with lambda=%f => ll=%f" % (best_lambda0, best_ll))
    lambda0 = best_lambda0
    return lambda0


def fit_lambda0_gridsearch(lifetimes, executions, 
                           pow_base=1.2, min_pow=-80, max_pow=120,
                           calc_ll=calc_ll, improvement=0):
    lambdas = [float(pow_base) ** pw for pw in np.array(range(int(max_pow-min_pow))) + min_pow]
    lambda0 = fit_lambda0_gridsearch1(lifetimes, executions, lambdas, calc_ll, improvement)
    return lambda0


def fit_lambda0_grad(lifetimes, executions, lambda0=0.00000001,
                     ll_prec = 0.000001, lambda_prec = 0.000001,
                     max_iter_ca = 100, max_iter_ga = 1000,
                     eps = 0.000000001, gamma = 0.9, eta =0.001,
                     calc_ll = calc_ll):    

    prev_lambda0 = -float("inf")
    for i in range(max_iter_ca):
        if abs(prev_lambda0-lambda0) < prev_lambda0*lambda_prec: break
        prev_lambda0 = lambda0

        logging.debug("[%i] finding next r" % i)
        r0 = fit_r0(lifetimes, executions, lambda0)
        logging.log(LOG_DBGINFO, "[%i] lambda=%f => r=%f" % (i, lambda0, r0))
        
        ###############################################################################
        logging.debug("[%i] finding next lambda with gradient ascent" % (i))
        #lambda0 = 0.000001 #better start from the previous position instead of some crap
        prev_lambda0_ga, Eg2 = -float("inf"), 0
        prev_ll = -float("inf")     
        for j in range(max_iter_ga):
            
            dbg_print2(j, "[%i,%i] gradient calculation for lambda" % (i, j))
            grad = 0
            for k, (lt, exec_flag) in enumerate(zip(lifetimes, executions)):
                #dbg_print2(k, "[%i, %i]  %i processes => grad=%f" % (i, j, k, grad))
                grad += r0*lt / (lambda0*lambda0+lambda0*lt)
                if int(exec_flag)!=0:
                    grad -= 1 / (lambda0+lt)
            
            Eg2 = gamma*Eg2 + (1-gamma)*grad*grad
            lambda0 = lambda0 + eta / np.sqrt(Eg2+eps) * grad
            ll = calc_ll(lifetimes, executions, r0, lambda0)
            dbg_print2(j, "[%i,%i]  r=%f grad=%f Eg2=%f => lambda=%f, ll=%f" % 
                           (i, j, r0, grad, Eg2, lambda0, ll))
            
            if abs(lambda0 - prev_lambda0_ga) < prev_lambda0_ga*lambda_prec*0.1:
                logging.debug(" lambda converged") 
                break 
            if abs(ll - prev_ll) < abs(prev_ll)*ll_prec:
                logging.debug(" ll converged") 
                break 
            if lambda0 <= 0:
                logging.warn("WARNING: lambda0=%s <= 0. RESTORING TO %s AND STOPPING" % (lambda0, prev_lambda0_ga))
                lambda0 = prev_lambda0
                break
            
            prev_lambda0_ga = lambda0
            prev_ll = ll
        ###############################################################################    
            
        r0 = fit_r0(lifetimes, executions, lambda0)
        logging.log(LOG_DBGINFO, "[%i] %i iterations of lambda gradient ascent performed, prev_lambda=%f => lambda=%f r=%f ll=%f" % 
                                    (i, j, prev_lambda0, lambda0, r0, ll))
        
    logging.info("%i iterations of coordinate ascent performed => lambda=%f r=%f ll=%f" % (i,  lambda0, r0, ll))    
    assert r0>=0, "pid=%i r0=%g lambda0=%g" % (os.getpid(), r0, lambda0)
    assert lambda0>=0, "pid=%i r0=%g lambda0=%g" % (os.getpid(), r0, lambda0)
    return lambda0    


def fit_params(lifetimes, executions, mode=0):
    logging.info("fitting mode = %s" % mode)
    
    if mode==4:
        lambda0 = fit_lambda0_gridsearch1(lifetimes, executions, 
                    lambdas=[1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0],
                    calc_ll=calc_ll_cv)
    elif mode==3:
        lambda0 = fit_lambda0_gridsearch(lifetimes, executions, 
                                         pow_base=10, min_pow=-2, max_pow=10,
                                         calc_ll=calc_ll_cv, improvement=0.01)    
    elif mode==2:
        lambda0 = fit_lambda0_gridsearch(lifetimes, executions, 
                                         pow_base=10, min_pow=-2, max_pow=10,
                                         calc_ll=calc_ll_cv)    
        pw = math.log10(lambda0)
        lambda0 = fit_lambda0_bisearch(lifetimes, executions, 
                                       prec=0.00001, left=10**(pw-1), right=10**(pw+1),
                                       calc_ll=calc_ll_cv)
        lambda0 = fit_lambda0_grad(lifetimes, executions, lambda0, 
                                   ll_prec=0.0001, lambda_prec=0.0001, 
                                   max_iter_ca=10, max_iter_ga=20,
                                   eps=0.000000001, gamma=0.9, eta=0.1,
                                   calc_ll=calc_ll_cv)    
    elif mode==1:    
        #start = time.time()
        #lambda0 = fit_lambda0_bisearch(lifetimes, executions)
        lambda0 = fit_lambda0_gridsearch(lifetimes, executions)  
        lambda0 = fit_lambda0_grad(lifetimes, executions, lambda0)
        #elapsed1 = time.time()-start
    elif mode==0:
        #start = time.time()    
        lambda0 = fit_lambda0_gridsearch(lifetimes, executions, pow_base=10, min_pow=-9, max_pow=9)
        pw = math.log10(lambda0)        
        lambda0 = fit_lambda0_bisearch(lifetimes, executions, prec=0.00001, left=10**(pw-1), right=10**(pw+1))    
        lambda0 = fit_lambda0_grad(lifetimes, executions, lambda0, 
                                   ll_prec=0.0001, lambda_prec=0.0001, 
                                   max_iter_ca=20, max_iter_ga=100,
                                   eps=0.000000001, gamma=0.9, eta=0.1)    
        #elapsed2 = time.time()-start
    else:
        raise ValueError("[fit_params] Unknown (=%s) mode value!" % mode)
                
    r0 = fit_r0(lifetimes, executions, lambda0)    
    #print("slow: %.3s -> %g (%g) fast: %.3s -> %g  (%g)" % (elapsed1, lambda0_1, calc_ll(lifetimes, executions, r0, lambda0_1),                                                 
    #                                                        elapsed2, lambda0,  calc_ll(lifetimes, executions, r0, lambda0)))    
    return r0, lambda0

 
def plot_ll_2d(lifetimes, executions, args, 
               l_max=100, l_min=0.01, r_max=1, r_min=0.01,
               l_c=None, r_c=None):
    ls = sorted(set(list((np.asarray(range(args.grid)))/args.grid*(l_max-l_min)+l_min)))
    rs = sorted(set(list((np.asarray(range(args.grid)))/args.grid*(r_max-r_min)+r_min)))
    v = np.zeros((len(ls), len(rs)))
    lsticks = []
    for i, l in enumerate(ls):
        logging.info(" plot generation: %i/%i" % (i, len(ls)))
        lsticks.append(i)
        rsticks = []
        for j, r in enumerate(rs):
            rsticks.append(j)
            v[i, j] = calc_ll(lifetimes, executions, r, l, args.r_reg, args.l_reg)
            #logging.info(" r=%f lambda=%f => ll=%f" % (r, l, v[i, j]))
    #logging.info(v)
    #logging.info("min=%f max=%f" % (np.min(v), np.max(v)))
    
    def fmt(x):
        if x>0: return "%.1f" % x
        return str(round(x, -int(math.floor(math.log10(abs(x))))))

    if l_c is not None and r_c is not None:
        
        for i in range(len(ls)):
            if ls[i-1]<=l_c and ls[i]>=l_c:
                l_c = i-1 + (l_c-ls[i-1])/(ls[i]-ls[i-1])
                break 
            
        for i in range(len(rs)):
            if rs[i-1]<=r_c and rs[i]>=r_c:
                r_c = i-1 + (r_c-rs[i-1])/(rs[i]-rs[i-1])
                break 
        
        pyplot.axhline(y=l_c)
        pyplot.axvline(x=r_c)

    pyplot.imshow(v)    
    pyplot.yticks(list(range(len(lsticks))))
    pyplot.gca().set_yticklabels(list(map(fmt, ls)))
    pyplot.ylabel("lambda")
    pyplot.xticks(list(range(len(rsticks))))    
    pyplot.gca().set_xticklabels(list(map(fmt, rs)), rotation='vertical')
    pyplot.xlabel("r")
    
    pyplot.colorbar()#norm=matplotlib.colors.Normalize(vmin=ll*0.9, vmax=ll*1.1))        
    #pyplot.clim(ll*0.9, ll*1.1)
    
    

def run_analysis(args):
    logging.info("Parsed args: %s" % args) 

    indata = args.input   
    #output_path = args.output   
    verbose = args.verbose or args.debug
    
    process_constructor_args = {}
    process_constructor_args["switch_time"] = float("inf") #TURN OFF SWITCHING BETWEEN RATES!
    if args.max_time is not None: process_constructor_args["max_time"] = args.max_time
    if args.start_time is not None: process_constructor_args["start_time"] = args.start_time 
    if not args.verbose: process_constructor_args["verbose"] = 0 
    if args.debug: process_constructor_args["verbose"] = 3
    logging.info("process constructor args: %s" % process_constructor_args)

    ###########################################################################
    
    logging.info("[%s] reading data from %s" % (str(datetime.now()), indata))    
    df = VAR[indata[7: ]] if indata.startswith("shared_") else pd.read_csv(indata, sep="\t")  
    ids = sorted(pd.unique(df["id"]))
    
    if verbose:
        logging.log(LOG_DBGINFO, "[Stats before updates. These can change: For example switching will be off!]")
        logging.log(LOG_DBGINFO, " #users=%s" % len(ids))    
        max_times, switch_times, start_times, execution_times = extract_processes_times(df)
        logging.log(LOG_DBGINFO, " #not executed actions=%s" % sum(execution_times>max_times))
        logging.log(LOG_DBGINFO, " #executed before switching=%s" % sum(execution_times<=switch_times))
        logging.log(LOG_DBGINFO, " #executed after switching=%s" % sum(np.logical_and(execution_times>switch_times, execution_times<=max_times)))
        logging.log(LOG_DBGINFO, " #started before switching=%s" % sum(start_times<switch_times)) #start_times==switch_times means that started after switching 
        logging.log(LOG_DBGINFO, " #started after switching=%s" % sum(np.logical_and(start_times>=switch_times, start_times<=max_times)))
        logging.log(LOG_DBGINFO, " start_times:\n  "+str(pd.Series(start_times).describe()).replace("\n", "\n  "))
        logging.log(LOG_DBGINFO, " execution_times:\n  "+str(pd.Series(execution_times).describe()).replace("\n", "\n  "))

    ###########################################################################
    
    logging.info("build_processes")
    id2p = build_processes(df, **process_constructor_args)

    logging.info("extracting lifetimes and execution flags")
    processes = list(id2p.values())
    lifetimes = [p.lifetime1() for p in processes]
    executions =  [int(p.when_executed()!=0) for p in processes]
    logging.info(" %i executed out out %i " % (sum(executions), len(executions)))
    
    logging.info("fitting average (common for all users) hazard value")
    a1, _, ll = fit_hazard_rates(id2p)
    logging.info(" alpha=%.10f ll=%f" % (a1, ll))

    logging.info("searching for (lambda, r) that would give a mean close to alpha") 
    r0, lambda0 = fit_params(lifetimes, executions, args)
        
    logging.info("plotting ll grid")
    plot_ll_2d(lifetimes, executions, args, 
               l_max=lambda0*2, l_min= lambda0*0.5,#max(0.00001, lambda0-(2*lambda0-lambda0)), 
               r_max=r0*2, r_min=r0*0.5, #r_min=max(0.00001, r0-(2*r0-r0)))
               r_c=r0, l_c=lambda0)
    pyplot.show()


def main(argv):
    parser = argparse.ArgumentParser(description="""
    Base functions for fitting of the robust (bayesian with priors) survival model.""")

    parser.add_argument("-i", "--input", dest='input', help="input TSV file with samples (three cols: id, time, type).",
                        metavar="input.tsv", required=True)
    parser.add_argument("-o", "--output", dest='output', help="output TSV file with fitted parameters",
                        metavar="output.tsv", required=False, default=None)


    parser.add_argument("-v", "--verbose", dest='verbose', help="print additional info", 
                        required=False, default=False, action='store_true')
    parser.add_argument("-d", "--debug", dest='debug', help="print additional debug info", 
                        required=False, default=False, action='store_true')
        
    #parser.add_argument("-st", "--switch_time", dest='switch_time', type=float,
    #                     help="set (force) time of switching between different intensities",  required=False, default=None)    
    parser.add_argument("-mt", "--max_time", dest='max_time', type=float,
                         help="set (force) max time value",  required=False, default=None)    
    parser.add_argument("-tt", "-zt", "--start_time", dest='start_time', type=float,
                         help="set (force) start time value",  required=False, default=None)

    parser.add_argument("--grid", dest='grid', type=int,
                         help="grid precision",  
                         required=False, default=20) 

    parser.add_argument("--r_range", dest='r_range', type=float,
                         help="r range",  
                         required=False, default=1) 

    parser.add_argument("--r_reg", dest='r_reg', type=float,
                         help="r regularization coefficient",  
                         required=False, default=0) 

    parser.add_argument("--l_range", dest='l_range', type=float,
                         help="lambda range",  
                         required=False, default=100) 

    parser.add_argument("--l_reg", dest='l_reg', type=float,
                         help="lambda regularization coefficient",  
                         required=False, default=0) 

    args, _ = parser.parse_known_args(argv)    

                
    ###########################################################################    
    
    if args.verbose: logging.basicConfig(level=LOG_DBGINFO)
    if args.debug:   logging.basicConfig(level=logging.DEBUG)

    
    return run_analysis(args)
        

if __name__=="__main__":
    main(sys.argv[1: ])
    
