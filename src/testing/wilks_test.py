#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Calculation of p-values from likelihood ratio (LL1-LL0) test of nested models using Chi2 distribution (Wilks theorem)."""

import sys
sys.path.append("../")

import argparse

from scipy.stats import chi2


def test(ll1, ll0, freedom=1):
    if ll1==0 and ll0==0: return 1.0
    return 1-chi2.cdf(2*(ll1-ll0), freedom)


def llr_from_pvalue(pvalue, freedom=1):
    if pvalue==0.0:
        print("[llr_from_pvalue] WARNING: pvalue==0.0. Changing to eps!") 
        pvalue = 10**(-31)         
    llr = chi2.ppf(1-pvalue, freedom)*0.5    
    return llr  


def main(argv):
    parser = argparse.ArgumentParser(description=
                                     """Calculates p-values of likelihood ratio (LL1-LL0) test
                                        of nested models
                                        using Chi2 distribution (Wilks theorem).""")
    parser.add_argument("-ll1", dest='ll1', type=float,
                        help="Log Likehood for H1 (more general model)",
                        required=True)
    parser.add_argument("-ll0", dest='ll0', type=float,
                        help="Log Likehood for H0 (nested model)",
                        required=True)
    parser.add_argument("-f", "--freedom", dest='freedom', type=float,
                        help="Difference between number of parameters of H1 model and H0 model",
                        required=True)
    
    args = parser.parse_args(argv)

            
    ###############################################################################################    
    
    print(test(args.ll1, args.ll0, args.freedom))

        
    
if __name__=="__main__":
    main(sys.argv[1: ])
    
    