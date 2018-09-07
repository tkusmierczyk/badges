#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Analysis (including empirical p-value calculation) and plotting of output data from sliding_window.py"""

import sys
import argparse
import pandas as pd
import numpy as np
import logging
import matplotlib
from scipy.stats import chi2
from matplotlib import pyplot
import seaborn as sns; sns.set()

sys.path.append("../")

from aux import parsing
from analytics import plots
from testing.bootstrap import unify_column_names, calc_empirical_pvalue, extract_closest_value





def plot_horizontal_level(level=None, level_label=None):
    xmin, xmax = pyplot.gca().get_xlim()
    if level is not None:
        pyplot.axhline(y=level, color="red", lw=4, ls="--")
        if not level_label:
            level_label = "p=%.2f" % level
        pyplot.text((xmin+xmax)*0.5, level, level_label, 
                    color="red", verticalalignment="bottom", horizontalalignment="center", fontsize=20)



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
              arrow=None, **kwargs):
    
    
    #xs = list(range(len(data)))
    #pyplot.xticks(xs, list(map(str, data["ratio"])), rotation=0)
    xs = list(map(float, data["ratio"]))
    pyplot.xscale("log")
    
    pyplot.tick_params(axis='x', which='major', labelsize=20)
    pyplot.tick_params(axis='y', which='major', labelsize=20)
    pyplot.ylabel(ylabel, fontsize=25)
    pyplot.xlabel(r"effect strength, $k_1/k_0$", fontsize=25)
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
        
    ax1.legend(fontsize=20)
    #ax1.set_ylim((0.0, 1.0))
    ax1.tick_params(axis='y', which='major', labelsize=20)
    
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
    


def calc_sensitivity(tdf, statistic_column):
    virtual_badge_statistics = tdf[tdf["badge_no"]<0][statistic_column]
    return sum(1.0 for pf in virtual_badge_statistics if pf<0.05) / len(virtual_badge_statistics)


def plot_llr_dist(virtual_badge_statistics):
    #plots.pyplot_reset()
    plots.densities_plot({"empirical": virtual_badge_statistics}, show=False, 
                         bw=params.get("bw", 0.1), shade=True, color="#2171b5", lw=2, cut=0)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    #pyplot.yscale('log')
    #pyplot.ylim((0.001, 100))
    
    
    xs = 2*np.linspace(0, min(max(virtual_badge_statistics),15))
    ys = [chi2.pdf(x, 1) for x in xs]
    pyplot.plot(xs, ys, lw=3, ls="--", label="theoretic", color="blue")
    
    #plots.pyplot_parse_params2(legend=False)
    pyplot.legend(fontsize=20, loc=1)
    pyplot.xlabel("test statistic ($LLR$)", fontsize=25)
    pyplot.ylabel("density", fontsize=25)
    pyplot.tick_params(axis='both', which='major', labelsize=20)
    pyplot.gcf().subplots_adjust(bottom=0.15, left=0.15)
    pyplot.grid(True)
    #xmin, xmax = pyplot.xlim()    
    #plots.savefig(args.output+"_llr_distribution.pdf", reset=False)    



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
    df["a1a0ratio"] = df["a1a0ratio"].apply(lambda ratio: round(ratio,6))
    ratios = sorted(df["a1a0ratio"].unique())
    
    #arrow_rejections, arrow_pvalues = None, None
    #if "switch" in args.input:
    #    noswitch = "noswitch" in args.input    
    #    arrow_rejections = (270 if noswitch else 90)
    #    arrow_pvalues = (90 if noswitch else 270)
    return df, ratios, badge_time


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="""
        Analysis (including empirical p-value calculation) and plotting of output data from sliding_window.py""")
    
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
    ##############################################################################################
    ##############################################################################################
    print("Empirical vs theoretic p-values rejections")
    for l0, ldf in df.groupby("l0"):
    #####################################################################
        data = pd.DataFrame({"ratio": ratios})      

        data["St"], lower, upper = theoretic_pvalues_rejections(ldf, ratios, "SHfH0_pval", badge_time)
        data["Se"], lower, upper = empirical_pvalues_rejections(ldf, ratios, "SHfH0_LLR", badge_time)        
        data["Be"], lower, upper = empirical_pvalues_rejections(ldf, ratios, "BHfH0_LLR", badge_time)
          
        plots.pyplot_reset()                    
        plot_data(data, ylabel=r"$H_0$ rejection probability", 
                  #level=0.05, level_label="5\%",
                  columns=["St", "Se", "Be"], labels=["basic theoretic", "basic bootstrap", "robust bootrsap"])
        pyplot.legend(fontsize=20, loc=4)
        plots.savefig("%s_l%g_rejections.pdf" % (args.input, l0))            
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
                  columns=["St", "Se", "Be"], labels=["basic theoretic", "basic bootstrap", "robust bootrsap"])
        plots.savefig("%s_l%g_p.pdf" % (args.input, l0))            
    #####################################################################        
    
    ##############################################################################################    
    #print("Created plots:\n %s" % "\n ".join(plots.get_saved_paths()))
    #sys.exit(0)    
    #print("WARNING: The code below is not used in the final version")
    
    ##############################################################################################
    print("Empirical (bootstrap) p-values")    
    for l0, ldf in df.groupby("l0"):
        data = pd.DataFrame({"ratio": ratios})      
        for statistic_column in ["SHfH0_LLR", "BHfH0_LLR"]:                
            center, lower, upper = empirical_pvalues(ldf, ratios, statistic_column, badge_time)
            data[statistic_column[0]] = center
            data[statistic_column[0]+"-l"] = lower
            data[statistic_column[0]+"-u"] = upper 
        plots.pyplot_reset()            
        plot_data(data, ylabel=r"$p$-value", level=0.05)
        plots.savefig("%s_l%g_p-bootstrap.pdf" % (args.input, l0))            


    ##############################################################################################
    print("Theoretic (chi2/fisher) p-values")
    for l0, ldf in df.groupby("l0"):
        data = pd.DataFrame({"ratio": ratios})      
        for statistic_column in ["SHfH0_pval"]: #["FHfH0_pval", "SHfH0_pval"]:                
            center, lower, upper = theoretic_pvalues(ldf, ratios, statistic_column, badge_time)
            data[statistic_column[0]] = center
            data[statistic_column[0]+"-l"] = lower
            data[statistic_column[0]+"-u"] = upper
        plots.pyplot_reset()
        plot_data(data, ylabel="$p$-value")
        plots.savefig("%s_l%g_p-theory.pdf" % (args.input, l0))    
        
    

    ##############################################################################################
    print("Sensitivity (fraction of fake badges with theoretic p-value<0.05)")
    for l0, ldf in df.groupby("l0"):    
        data = pd.DataFrame({"ratio": ratios})      
        for statistic_column in ["FHfH0_pval", "SHfH0_pval", "BHfH0_pval"]:                
            lower, center, upper = [], [], []  
            for ratio in ratios:
                rdf = ldf[ldf["a1a0ratio"]==ratio]
                vals = [calc_sensitivity(tdf, statistic_column) for trial_no, tdf in rdf.groupby(["trial_no"])]        
                print("    l0=",l0,"ratio=",ratio, "method=", statistic_column, "sensitivity=", ",".join(map(lambda v: "%g" %v, vals)))                                        
                center.append(np.mean(vals))
                lower.append(np.mean(vals)-np.std(vals)/np.sqrt(len(vals)))
                upper.append(np.mean(vals)+np.std(vals)/np.sqrt(len(vals)))
            data[statistic_column[0]] = center
            data[statistic_column[0]+"-l"] = lower
            data[statistic_column[0]+"-u"] = upper  
        plots.pyplot_reset()                       
        plot_data(data, ylabel=r"sensitivity ($S$)", level=None)        
        pyplot.ylim((0, 0.5))
        plots.savefig("%s_l%g_S.pdf" % (args.input, l0))            
    
    ##############################################################################################    
    print("LLR distribution")    
    for (l0, ratio), ldf in df.groupby(["l0", "a1a0ratio"]):
        if l0 not in [1.0, 10.0, 100.0, 1000.0]: continue
        if ratio not in [10.0, 100.0]: continue
        for statistic_column in ["SHfH0_LLR", "BHfH0_LLR"]:
            plots.pyplot_reset()
            plot_llr_dist(ldf[ldf["badge_no"]<0][statistic_column])
            print("creating a file: %s_l%g_r%g_%s.pdf" % (args.input, l0, ratio, statistic_column))
            pyplot.title("method:%s lambda=%g ratio=%g" % (statistic_column[0], l0, ratio))            
            plots.savefig("%s_l%g_r%g_%s.pdf" % (args.input, l0, ratio, statistic_column))



