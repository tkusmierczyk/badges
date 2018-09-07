#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" Calculates p-values from various models 
    (Fisher=counts, Simple=simple survival, Robust='bayesian' with priors) 
    using sliding window over synthetically generated data.
    Outputs a TSV file containing the test statistics for each time point."""
    
from multiprocessing import Pool
import pandas as pd
import argparse
import os
import logging
import traceback
import random
import parser
import sys
sys.path.append("../")


from synthetic.thinning import run as synthesis

from testing.wilks_test import test
from testing.fisher_test import fisher_exact_test

from processes.fit_simple import main as fit
from processes.fit_bayes1 import main as fit1
from processes.fit_bayes2 import main as fit2

from aux.move_censoring import move_censoring
from aux.sharing import VAR
from aux.timer import run
from aux import parsing



COLUMNS = [ "N", "T", "ST", "sf", "gen", "trial_no", 
            "l0", "l1", "r0", "r1", "a0", "a1", "trend",
            "wc", "t", "endt", "size", 
            "alt_model", "badge_no", "badge_time", "prev_badges",
            #####################################################
            "FHf_a", "FHf_b", "FHf_c", "FHf_d", "FHfH0_pval", 
            "FH1_a", "FH1_b", "FH1_c", "FH1_d", "FH1H0_pval",
            ##################################################### 
            "SH0_a0", "SH0_a1", "SH0_ll", 
            "SHf_a0", "SHf_a1", "SHf_ll", "SHfH0_pval", "SHfH0_LLR", 
            "SH1_a0", "SH1_a1", "SH1_ll", "SH1H0_pval", "SH1H0_LLR", 
            #####################################################
            "BH0_r0", "BH0_lambda0", "BH0_ll", "BH0_a0",             
            "BHf_r0", "BHf_lambda0", "BHf_r1", "BHf_lambda1", "BHf_ll", "BHf_a0", "BHf_a1", "BHfH0_pval",  "BHfH0_LLR", 
            "BH1_r0", "BH1_lambda0", "BH1_r1", "BH1_lambda1", "BH1_ll", "BH1_a0", "BH1_a1", "BH1H0_pval",  "BH1H0_LLR"]
assert len(set(COLUMNS))==len(COLUMNS)

LOG_DBGINFO = 15
    
INF = float("inf")
NAN = float("nan")


def _print2(txt):
    print(txt)
    logging.info(txt)
    
    
def _div(a, b):
    if a==0 and b==0: return 0
    return a / b


def _frange(f, to, step):
    while f < to:
        yield f
        f += step
    
    
def data_generator(params):
    N = int(params.get("N", 10000))
    T = float(params.get("T", 360))
    ST = float(params.get("ST", T*0.5))
    
    trial_no = params["trial_no"]     
    
    lambda0 = float(params.get("lambda0", 100))
    lambda1 = lambda0
    r0 = params.get("r0", 0.01) 
    r1 = params.get("r1", 0.1)   
    
    data = synthesis(("""-d ../processes/process_factory.py                 
                 -st %(st)s -mt %(T)s -zt 0        
                 -n %(n)s -sf 1.0
                 --randomize
                 --r1 %(r0)s --lambda1 %(lambda0)s 
                 --r2 %(r1)s --lambda2 %(lambda1)s                 
                 --seed %(seed)i
                 """ % { "r0": r0, "lambda0": lambda0,
                         "r1": r1, "lambda1": lambda1,
                         "st": ST, "T": T, "n": N, 
                         "seed": 123*trial_no}).split()) 
    return data     
    
    
def trend_data_generator(params):
    N = int(params.get("N", 10000))
    T = float(params.get("T", 360))
    ST = float(params.get("ST", T*0.5))
    
    trial_no = params["trial_no"]     
    
    lambda0 = float(params.get("lambda0", 100))
    lambda1 = lambda0
    r0 = params.get("r0", 0.01) 
    r1 = params.get("r1", 0.1)    
    trend = float(params.get("trend", 0))     
    noswitch = bool(int(params.get("noswitch", 0)))     
    
    data = synthesis(("""-d ../processes/trend_process_factory.py                 
                 -st %(st)s -mt %(T)s -zt 0        
                 -n %(n)s -sf 1.0
                 --randomize
                 --r1 %(r0)s --lambda1 %(lambda0)s 
                 --r2 %(r1)s --lambda2 %(lambda1)s
                 --trend %(trend)s                
                 --seed %(seed)i
                 %(noswitch)s
                 """ % { "r0": r0, "lambda0": lambda0,
                         "r1": r1, "lambda1": lambda1,
                         "trend": trend,
                         "st": ST, "T": T, "n": N,
                         "noswitch": "-ns" if noswitch else "", 
                         "seed": 123*trial_no}).split()) 
    return data  
    

def simulation(params):
    try:
        trial_no = params["trial_no"]
        window = float(params["window"])
        step = float(params["step"])
        
        N = int(params.get("N", 10000))
        T = float(params.get("T", 360))
        ST = float(params.get("ST", T*0.5))
        hazard_ratio = float(params.get("hazard_ratio", 1.5)) 
        
        fitting_mode = params.get("fitting_mode", None)
        independent_lambdas = params.get("independent_lambdas", None)         
        if independent_lambdas is not None: 
            if fitting_mode is None:
                raise ValueError("If independent_lambdas (for bayes1 and bayes2) are requested fitting_mode must be set!")  
            independent_lambdas = bool(independent_lambdas)
        
        lambda0 = float(params["lambda0"])
        lambda1 = lambda0
        r0 = params["r0"] 
        r1 = params["r1"]   
        alpha0 = r0/lambda0
        alpha1 = r1/lambda0        
        
        ######################################################################################
    
        header = "[%i][a0=%g a1=%g #=%s r0=%g l0=%g r1=%g l1=%g]" % \
                (os.getpid(), alpha0, alpha1, trial_no, r0, lambda0, r1, lambda0)
        logging.info("\n#########################\n%s" % header)            

        ###########################################################################################
        
        _print2("%s[Thinning] generating data" % (header))
        simulation = params["data_generator"](params)                              
        processes  = simulation["processes"]
        data = simulation["samples"]

        influences = [int(p._a2/p._a1 >  hazard_ratio) for p in processes if p._start_time<ST]
        switching = float(sum(influences))/len(processes) #len(influences)
                      
        windows = list(_frange(0.0, T-window, step))
        _print2("%s[Fitting] sliding window fitting (windows=%s)" % (header, str(windows)[:100]))                        
        results = []
        for i, t in enumerate(windows):
            #if i % 100 == 0: _print2("%s progress: %i/%i" % (header, i, len(windows)))
            endt = t + window
            window_center = t + window*0.5
            badge_time = ST                    
            num_badges_within_window = int(badge_time>t and badge_time<endt)
            alt_model = num_badges_within_window > 0
            num_badges_until_window = int(badge_time<endt)
            badge_no = num_badges_until_window * int(alt_model) - 1
            logging.info(">=================")
            sdf = run(move_censoring, data, t, endt, keep_inf=True, method_label="censoring")        
            VAR["sdf"] = sdf 
    
            #Store params
            row = [N, T, ST, switching, params["data_generator"].__name__[:1], trial_no, 
                   lambda0, lambda1, r0, r1, alpha0, alpha1, float(params.get("trend", 0)),
                   window_center, t, endt, len(sdf["id"].unique()), 
                   int(alt_model), badge_no, badge_time, num_badges_until_window]
            
            resutls_string = " \t".join("%10s" % ("%s=%s" % (col, val)) for (col, val) in zip(COLUMNS[ : len(row)], row))
            _print2("[%i][%i/%i] >>> fitting_mode=%s independent_lambdas=%s %s" % 
                    (os.getpid(), i, len(windows), fitting_mode, independent_lambdas, resutls_string) )

            
            #_print2("======")
            #TODO testing period should be given as an argument
            FHf_a, FHf_b, FHf_c, FHf_d, FHf_pval = fisher_exact_test(sdf, window_center, int(params["window"] * 0.25))
            row.extend([FHf_a, FHf_b, FHf_c, FHf_d, FHf_pval])
            
            FH1_a, FH1_b, FH1_c, FH1_d, FH1_pval = fisher_exact_test(sdf, badge_time, int(params["window"] * 0.25))
            row.extend([FH1_a, FH1_b, FH1_c, FH1_d, FH1_pval])
            
            #_print2("=========")
            SH0_a0, SH0_a1, _, SH0_ll = fit(["-i", "shared_sdf", "-v", "-h0"])
            SHf_a0, SHf_a1, _, SHf_ll = fit(["-i", "shared_sdf", "-v", "-st", str(window_center)]) #fake badge
            SH1_a0, SH1_a1, _, SH1_ll = fit(["-i", "shared_sdf", "-v", "-st", str(badge_time)]) #true badge if any exists
            row.extend([SH0_a0, SH0_a1, SH0_ll, 
                        SHf_a0, SHf_a1, SHf_ll, test(SHf_ll, SH0_ll, 1), SHf_ll-SH0_ll, 
                        SH1_a0, SH1_a1, SH1_ll, test(SH1_ll, SH0_ll, 1), SH1_ll-SH0_ll])
            
            #_print2("============") 
            fit1_lambda_params = ["-mode", str(fitting_mode)] if (fitting_mode is not None) else ["-l", str(lambda0)]
            BH0_r0, BH0_lambda0, BH0_ll = fit1(["-i", "shared_sdf"] + fit1_lambda_params)
            BH0_a0 = float(BH0_r0) / BH0_lambda0
                        
            fit2_lambda_params = ["-mode", str(fitting_mode)] if independent_lambdas else ["-l", str(BH0_lambda0)] 
            BH1_r0, BH1_lambda0, Bll0, BH1_r1, BH1_lambda1, Bll1 = fit2(["-i", "shared_sdf", "-st", str(badge_time)] + fit2_lambda_params)
            BH1_ll, BH1_a0, BH1_a1 = Bll0 + Bll1, float(BH1_r0) / BH1_lambda0, float(BH1_r1) / BH1_lambda1
            
            BHf_r0, BHf_lambda0, Bll0, BHf_r1, BHf_lambda1, Bll1 = fit2(["-i", "shared_sdf", "-st", str(window_center)] + fit2_lambda_params)
            BHf_ll, BHf_a0, BHf_a1 = Bll0 + Bll1, float(BHf_r0) / BHf_lambda0, float(BHf_r1) / BHf_lambda1
            
            row.extend([BH0_r0, BH0_lambda0, BH0_ll, BH0_a0, 
                        BHf_r0, BHf_lambda0, BHf_r1, BHf_lambda1, BHf_ll, BHf_a0, BHf_a1, test(BHf_ll, BH0_ll, 1), BHf_ll-BH0_ll, 
                        BH1_r0, BH1_lambda0, BH1_r1, BH1_lambda1, BH1_ll, BH1_a0, BH1_a1, test(BH1_ll, BH0_ll, 1), BH1_ll-BH0_ll])
            
            #_print2("===============")
            resutls_string = " \t".join("%10s" % ("%s=%s" % (col, val)) for (col, val) in zip(COLUMNS, row))
            _print2("[%i][%i/%i] <<< fitting_mode=%s independent_lambdas=%s %s" % 
                    (os.getpid(), i, len(windows), fitting_mode, independent_lambdas, resutls_string) )
            
            results.append(row)
    except Exception as e:
        logging.error("ERROR: %s\n" % str(e))
        logging.error(traceback.format_exc())
        traceback.print_exc()
        print(str(e))
        sys.exit(-1) 
                  
    return results


def prepare_configurations(params):
    alpha0 = float(params.pop("alpha0", 0.0001))
     
    mode = int(params.get("mode", params.get("fast", "0")))
    if mode == 1:
        alpha1scales = [10]
        numtrials = 3
        lambdas = [10, 100]
    elif mode == 2:
        numtrials = int(params.pop("numtrials", 100)) #how many experiment runs    
        alpha1scales = [1.0, 1.5, 2, 3, 5, 10, 30, 100]    
        lambdas = [10.0, 100, 1000.0]
    else: 
        numtrials = int(params.pop("numtrials", 100)) #how many experiment runs    
        alpha1scales = [1.0, 1.25, 1.5, 2, 5, 10, 20, 100]    
        lambdas = [0.1, 1.0, 10.0, 100.0, 1000.0]        
    _print2("[alpha0=%f alpha1scales=%s lambdas=%s numtrials=%i]" % (alpha0, alpha1scales, lambdas, numtrials))  
        
    #########################################################################
    
    configurations = []
    for alpha1scale in alpha1scales:
            for lambda0 in lambdas:                    
                for trial_no in range(numtrials):
                    #lambda0 = float(numpy.random.gamma(shape=1, scale=1000, size=1))
                    #lambda1 = float(numpy.random.gamma(shape=1, scale=1000, size=1))
                    alpha1 = alpha1scale * alpha0            
                    cparams = params.copy()
                    r0 = alpha0 * lambda0
                    r1 = alpha1 * lambda0

                    cparams["lambda0"] = lambda0
                    cparams["r0"] = r0
                    cparams["r1"] = r1 
                    cparams["trial_no"] = trial_no                     
                           
                    configurations.append(cparams)
    
    return configurations


def prepare_params(args):
    params = parsing.parse_dictionary(args.params)
    try:    verbose = args.verbose
    except: verbose = False    
    try:    debug = args.debug
    except: debug = False    
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARN)    
    random.seed(123)
    fmt = '[%(process)4d][%(asctime)s][%(levelname)-5s][%(module)s:%(lineno)d/%(funcName)s] %(message)s'
    if args.output is not None:
        if args.logfile:
            logfile = "%s.log" % (args.output)
            logging.basicConfig(filename=logfile, level=level, format=fmt)
            print("logging to %s\n" % logfile)
        else:
            logging.basicConfig(level=level, format=fmt)
    else:
        logging.basicConfig(level=level, format=fmt)
    if verbose:
        logging.getLogger().addHandler(logging.StreamHandler())
    if args.output is None:
        args.output = "%s_%i" % (os.path.basename(__file__), os.getpid())
    return params



if __name__=="__main__":
    

    parser = argparse.ArgumentParser(description=""" 
        Calculates p-values from various models 
        (Fisher=counts, Simple=simple survival, Robust='bayesian' with priors) 
        using sliding window over synthetically generated data.
        Outputs a TSV file containing the test statistics for each time point.""")
    
    parser.add_argument("-v", "--verbose", dest='verbose', help="print additional info", 
                        required=False, default=False, action='store_true')
    parser.add_argument("-d", "--debug", dest='debug', help="print additional debug info", 
                        required=False, default=False, action='store_true')
    
    
    parser.add_argument("-o", "--output", dest='output', 
                        help="output files prefix", 
                        required=False, 
                        default="%s" % (os.path.basename(__file__)))
                        #default="%s_%i" % (os.path.basename(__file__), os.getpid()))
    parser.add_argument("-l", "--logfile", dest='logfile', 
                        help="don't create log file", 
                        required=False, default=True, action="store_false")
    parser.add_argument("-c", "--cpu", dest='cpu', 
                        help="num processes to use", 
                        required=False, type=int, default=1)    

    parser.add_argument("-w", "--window", dest='window', type=float, help="window size",  
                         required=False, default=60)
    parser.add_argument("-s", "--step", dest='step', type=float,
                        help="how much the sliding window should be moved each time",  
                         required=False, default=1)      
      
    parser.add_argument("-p", "--params", dest='params', 
                        help="comma-separated params: option=value", 
                        required=False, default="")
            
    args = parser.parse_args(sys.argv[1: ])    
    
    params = prepare_params(args)    
    params["window"] = args.window
    params["step"] = args.step
    params["trial_no"] = 1
    params["data_generator"] = trend_data_generator if "trend" in params else data_generator    
    
    #########################################################################
    
    configurations = prepare_configurations(params)
    print("%i configurations to simulate on %i cpus" % (len(configurations), args.cpu))
    #configurations = [params]       
    if args.cpu>1:         
        processes = Pool(args.cpu)
        results = processes.map(simulation, configurations)
    else:
        results = [simulation(c) for c in configurations]
    
    data = []
    for partial_results in results:
        data.extend(partial_results)

    #########################################################################

    print("Saving results to %s" % (args.output+".tsv"))    
    data = pd.DataFrame(data).rename(columns=dict(enumerate(COLUMNS)))
    data.to_csv(args.output+".tsv", sep="\t", index=False, header=True)
    
    #########################################################################
    #########################################################################
    