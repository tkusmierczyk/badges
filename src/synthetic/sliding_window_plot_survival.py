#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Plotting intensity values extracted from output data calculated in sliding_window.py"""

import sys
import argparse
import pandas as pd
import numpy as np
import logging

sys.path.append("../")

from aux import parsing
from testing.bootstrap import unify_column_names, calc_empirical_pvalue, extract_closest_value

from matplotlib import pyplot
from analytics import plots


def plot_horizontal_level(level=None, level_label=None):
    xmin, xmax = pyplot.gca().get_xlim()
    if level is not None:
        pyplot.axhline(y=level, color="red", lw=4, ls="--")
        if not level_label:
            level_label = "p=%.2f" % level
        pyplot.text((xmin+xmax)*0.5, level, level_label, 
                    color="red", verticalalignment="bottom", horizontalalignment="right", fontsize=20)



def plot_arrow(ax1, arrow=None):
    if arrow is not None:
        ymin, ymax = pyplot.gca().get_ylim()
        bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="none", ec="k", lw=2)
        ax1.text(1, (ymin + ymax) * 0.5, "better", ha="center", va="center", rotation=arrow, size=25, bbox=bbox_props)


def plot_nochange_region(ax1):
    ymin, ymax = pyplot.gca().get_ylim() #props = dict(boxstyle='round', facecolor='lightgray', alpha=0.25, edgecolor="none")
    #ax1.annotate("no effect here", xy=(1.0, 0.5*(ymin+ymax)), #xytext=(pos, min(1.0, ymax + 0.2)),
    #                 color="black", fontsize=20, #arrowprops=dict(facecolor='red', shrink=0.05, edgecolor="none"),
    #                 horizontalalignment='center', verticalalignment='center', rotation=90,
    #                 bbox=props)
    props = dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor="none")
    ax1.annotate(r"no effect here", xy=(1.0, ymax * 0.95), #xytext=(pos, min(1.0, ymax + 0.2)),
        color="black", fontsize=20, #arrowprops=dict(facecolor='red', shrink=0.05, edgecolor="none"),
        horizontalalignment='center', verticalalignment='top', rotation=90, bbox=props)
    pyplot.axvline(x=1.0, color="black", lw=4, ls="--")
    

def plot_data(data, 
              ylabel="?", 
              columns=["F", "S", "B"], 
              labels=["Counts", "Survival", "Robust"],
              colors = ["dodgerblue", "salmon", "limegreen", "dimgrey"],
              markers = ["h", "o", "*", "."],
              level=0.05, level_label=None,  
              arrow=None, leg_loc=1, **kwargs):
    
    
    #xs = list(range(len(data)))
    #pyplot.xticks(xs, list(map(str, data["ratio"])), rotation=0)
    xs = list(map(float, data["ratio"]))
    pyplot.xscale("log")
    
    pyplot.tick_params(axis='x', which='major', labelsize=25)
    pyplot.tick_params(axis='y', which='major', labelsize=25)
    pyplot.ylabel(ylabel, fontsize=25)
    pyplot.xlabel(r"effect strength, ${\mathbb{E}}[\Delta P]$", fontsize=25)
    pyplot.grid(True)
    
    ax1 = pyplot.gca()#.twinx()
    #ax2 = pyplot.gca()
    
    for i in range(len(columns)):
        column = columns[i]
        label = labels[i]
        color = colors[i%len(colors)] 
        marker = markers[i%len(markers)]
        if column not in data.columns:
            print("WARNING: [plot_data] no column=%s in data" % column)
            continue

        ax1.plot(xs, data[column], label=label, color=color, lw=4, 
                     marker=marker, markeredgecolor="none", markersize=10)
        #ax1.plot(xs, data[column+"-l"], label=None, color=color, lw=1, ls="--")
        #ax1.plot(xs, data[column+"-u"], label=None, color=color, lw=1, ls="--")
        #ax1.fill_between(xs, data[column+"-l"], data[column+"-u"], facecolor=color, edgecolor="None", alpha=0.2)    
        
    #ax1.legend(fontsize=20)
    leg = ax1.legend(fontsize=20, fancybox=False, loc=leg_loc)
    leg.get_frame().set_alpha(1.0)
    ax1.tick_params(axis='y', which='major', labelsize=20)
    frame = leg.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    frame.set_linewidth(0.5)
    
    pyplot.tick_params(axis='x', which='major', labelsize=22)
    pyplot.tick_params(axis='y', which='major', labelsize=22)    
        
    #ax1.set_ylim((0.0, 1.0))
    #ax1.tick_params(axis='y', which='major', labelsize=20)
    
    #ax2.plot(xs, data["sf"], color=COL0, lw=0.5, ls="--")
    #ax2.fill_between(xs, [0 for _ in xs], data["sf"], facecolor=COL0, edgecolor="None", alpha=0.25)
    #ax2.set_ylim((0.0, 1.0))
    #ax2.tick_params(axis='y', which='major', labelsize=20, labelcolor=COL0)
    
    #plot_influenced_users(ax1, data)
    #ax1.set_xlim((-0.75, len(xs)-1+0.75))
    ax1.set_xlim((min(0.5, min(xs)/2), max(xs)*2.0))

    plot_horizontal_level(level, level_label)
    plot_nochange_region(ax1)    
            
    #ax1.set_ylabel(r"$|\{{\alpha^u_1}/{\alpha^u_0} > 1.5\}|/|U|$", fontsize=25, color=COL0)
    #ax1.set_ylabel(r"influenced users ($p_f$)", fontsize=25, color=COL0)
    #pyplot.gcf().subplots_adjust(bottom=0.21, top=0.79)# right=0.85, left=0.15)
    pyplot.gcf().subplots_adjust(bottom=0.15, left=0.15)# right=0.85, left=0.15)
    

#################################################################################################################


def extract_values(ldf, ratios, statistic_column, badge_time, verbose=False,
                   column_extractor=calc_empirical_pvalue):
    lower, center, upper = [], [], []
    for ratio in ratios:
        rdf = ldf[ldf["a1a0ratio"] == ratio]
        vals = [column_extractor(tdf, statistic_column, badge_time) for _, tdf in rdf.groupby(["trial_no"])]
        center.append(np.mean(vals))
        lower.append(np.mean(vals) - np.std(vals) / np.sqrt(len(vals)))
        upper.append(np.mean(vals) + np.std(vals) / np.sqrt(len(vals)))
        if verbose:
            print("    l0=", l0, "ratio=", ratio, "method=", statistic_column, "vals=", ", ".join(map(lambda v: "%g" %v, vals)))                                            
    return center, lower, upper


def empirical_pvalues(ldf, ratios, statistic_column, badge_time, verbose=False):        
    return extract_values(ldf, ratios, statistic_column, badge_time, verbose,
                   column_extractor=calc_empirical_pvalue)


def theoretic_pvalues(ldf, ratios, statistic_column, badge_time, verbose=False):
    return extract_values(ldf, ratios, statistic_column, badge_time, verbose,
                   column_extractor=extract_closest_value)
    
    
def empirical_pvalues_rejections(ldf, ratios, statistic_column, badge_time, verbose=False):    
    def f(tdf, statistic_column, badge_time):
        return int(calc_empirical_pvalue(tdf, statistic_column, badge_time)<0.05)  
    return extract_values(ldf, ratios, statistic_column, badge_time, verbose,
                   column_extractor=f)


def theoretic_pvalues_rejections(ldf, ratios, statistic_column, badge_time, verbose=False):
    def f(tdf, statistic_column, badge_time):
        return int(extract_closest_value(tdf, statistic_column, badge_time)<0.05)  
    return extract_values(ldf, ratios, statistic_column, badge_time, verbose,
                   column_extractor=f)


#################################################################################################################

    
def fraction_of_executed_until_t_mc(shape_r, lambda_rate, t=10, N=10000000):
    l = np.random.gamma(shape=shape_r, scale=1.0/lambda_rate, size=N) #sample N user ls
    execution_probability = 1.0-np.exp(-t*l)
    return float(sum(execution_probability))/N
    

def load_data(args, params):
    logging.info("reading data from %s" % args.input)
    df = pd.read_csv(args.input, sep="\t")
    df = unify_column_names(df) 

    for fix in ["trial_no", "l0", "l1", "r0", "r1", "a0", "a1"]: 
        if params.get(fix, None) is not None:    
            print(" filtering rows with %s" % fix)
            df = df[df[fix]==float(params[fix])]

    logging.info(df.head())
    logging.info("Columns: %s" % df.columns)
    
    assert len(df["a0"].unique())==1
    assert len(df["N"].unique())==1
    assert len(df["ST"].unique())==1
    assert len(df["T"].unique())==1

    N = df["N"].unique()[0]
    T = df["T"].unique()[0]
    badge_time = df["ST"].unique()[0]
    alpha0 = df["a0"].unique()[0]    
    numtrials = len(df["trial_no"].unique())
    
    print("N=%i T=%f ST=%f alpha0=%f numtrials=%i" % (N, T, badge_time, alpha0, numtrials))
    
    df["a1a0ratio"] = df["a1"]/alpha0
    df = df[df["l0"]==df["l1"]]
    df = df[df["a1a0ratio"]>=args.minratio] #select only with ratio >= val
    
    
    #TRANSFORMING RATIOS INTO DIFFERENCES BETWEEN EXECUTION PROBABILITY
    ratio2effectstrength = {}
    HORIZON = 10
    lambda_rate = df["l0"].unique()[0]    
    shape_r0 = alpha0*lambda_rate    
    print("calculating survival probability in time<%s for alpha0 = %s" % (HORIZON, alpha0))
    P0 = fraction_of_executed_until_t_mc(shape_r0, lambda_rate, t=HORIZON, N=10000000)
    for alpha1 in set(round(df["a1"],6)):
        print("calculating survival probability in time<%s alpha1 = %s" % (HORIZON, alpha1))
        shape_r1 = alpha1*lambda_rate  
        P1 = fraction_of_executed_until_t_mc(shape_r1, lambda_rate, t=HORIZON, N=10000000)
        effect_strength = P1-P0
        ratio2effectstrength[round(alpha1/alpha0,6)] = effect_strength
        
    df["a1a0ratio"] = df["a1a0ratio"].apply(lambda ratio: round(ratio,6))
    df["a1a0ratio"] = df["a1a0ratio"].apply(lambda ratio: ratio2effectstrength[round(ratio,6)])
    df["a1a0ratio"] = df["a1a0ratio"].apply(lambda ratio: round(ratio,6))
    ratios = sorted(df["a1a0ratio"].unique())
    
    #arrow_rejections, arrow_pvalues = None, None
    #if "switch" in args.input:
    #    noswitch = "noswitch" in args.input    
    #    arrow_rejections = (270 if noswitch else 90)
    #    arrow_pvalues = (90 if noswitch else 270)
    return df, ratios, badge_time
    
    

if __name__=="__main__":
    
    np.random.seed(123)
    
    parser = argparse.ArgumentParser(description="""
    Plotting intensity values ( E(survival1)-E(survival0) ) extracted from output data calculated in sliding_window.py""")
    
    parser.add_argument("-i", "--input", dest='input', 
                        help="input TSV file", 
                        required=True)

    parser.add_argument("-p", "--params", dest='params', 
                        help="comma-separated params: option=value", 
                        required=False, default="")
                           
    parser.add_argument("-d", "--debug", dest='debug', 
                        help="print debug information", 
                        action="store_true", default=False)
    
    parser.add_argument("-minr", "--minratio", dest='minratio', 
                        help="min value of ratio=a1/a0 to plot", 
                        required=False, default=1.01, type=float)    
    

    args = parser.parse_args(sys.argv[1: ])        
    params = parsing.parse_dictionary(args.params)
    output = args.input
    
    logging.basicConfig(level=logging.INFO)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    
            
    ##############################################################################################
    df, ratios, badge_time = load_data(args, params)
    
    ##############################################################################################
    print("Empirical vs theoretic p-values rejections")
    for l0, ldf in df.groupby("l0"):
    #####################################################################
        data = pd.DataFrame({"ratio": ratios})      

        data["St"], lower, upper = theoretic_pvalues_rejections(ldf, ratios, "SHfH0_pval", badge_time)
        data["Se"], lower, upper = empirical_pvalues_rejections(ldf, ratios, "SHfH0_LLR", badge_time)        
        data["Be"], lower, upper = empirical_pvalues_rejections(ldf, ratios, "BHfH0_LLR", badge_time)
          
        plots.pyplot_reset()                    
        pyplot.ylim((0, 1.0))                    
        plot_data(data, ylabel=r"$H_0$ rejection probability", 
                  #level=0.05, level_label="5\%",
                  columns=["St", "Se", "Be"], 
                  labels=["basic theoretic", "basic bootstrap", "robust bootrsap"], 
                  leg_loc=2)
        pyplot.ylim((0, 1.0))  
        pyplot.gcf().subplots_adjust(bottom=0.17, left=0.18)                                  
        plots.savefig("%s_l%g_rejections_survival.pdf" % (args.input, l0))            
    #####################################################################    
        
    ##############################################################################################
    print("Empirical vs theoretic p-values")    
    for l0, ldf in df.groupby("l0"):
    #####################################################################
        data = pd.DataFrame({"ratio": ratios})      

        data["St"], lower, upper = theoretic_pvalues(ldf, ratios, "SHfH0_pval", badge_time)
        data["Se"], lower, upper = empirical_pvalues(ldf, ratios, "SHfH0_LLR", badge_time)        
        data["Be"], lower, upper = empirical_pvalues(ldf, ratios, "BHfH0_LLR", badge_time)
                 
        plots.pyplot_reset()                 
        pyplot.ylim((0, 1.0))                    
        plot_data(data, ylabel=r"$p$-value", level=0.05, 
                  columns=["St", "Se", "Be"], labels=["basic theoretic", "basic bootstrap", "robust bootrsap"], leg_loc=1)
        pyplot.ylim((0, 1.0)) 
        pyplot.gcf().subplots_adjust(bottom=0.17, left=0.18)                                   
        plots.savefig("%s_l%g_p_survival.pdf" % (args.input, l0))            
    #####################################################################        
    

