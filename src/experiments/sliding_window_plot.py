#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Visualises outputs of the sliding_window.py."""

import pandas as pd
import argparse
import numpy
from scipy import stats
from matplotlib import pyplot
import seaborn as sns; sns.set()
import sys
import math
import logging

sys.path.append("../")

from analytics import plots
from aux import parsing
from analytics.plots_time_axis import set_xaxis
from testing.bootstrap import ensure_llr_values, unify_column_names
from testing.bootstrap import extract_closest_value, calc_empirical_pvalue, find_badge_rows

    
INF = float("inf")
NAN = float("nan")
COLORS = ["dodgerblue", "salmon", "limegreen"]

TIMECOL = "wc"

    
    
def _fmt_p(p, label="p"): 
    if p==1.0: return r"$%s=1.0$" % label
    if p<0.00000001: return r"$%s<10^{-9}$" % label
    if p<0.001: return r"$%s<0.001$" % label
    return r"$%s=%.3f$" % (label, p)


def _smooth(y, halfwin=5, agg=numpy.mean):
    if halfwin<=0: return numpy.array(y)
    return numpy.array([ agg(y[i-halfwin: i+halfwin]) for i in range(len(y))])


def _smooth_around_badges(x, y, badges=[INF], halfwin=6):
    #return x, y
    x, y  = numpy.array(x), numpy.array(y)
    badge_vals = [y[numpy.argmin(abs(b-x))] for b in badges]

    #gaussian window        
    w = numpy.array([stats.norm.pdf(shift, loc=0, scale=halfwin*0.33) for shift in range(-halfwin, halfwin)])
    w = w / sum(w)
    
    gety = lambda ixstart, ixend: numpy.array([1.0 if i<0 or i>=len(y) else y[i] for i in range(ixstart, ixend)])
    def badge_influence(p):
        for b, bv in zip(badges, badge_vals):
            if abs(b-p)<=halfwin: 
                return [bv for _ in range( 100*int(abs(b-p)) )]
        return []
    return x, [ y[i] if min(abs(p-badges))<halfwin*0.75 else numpy.dot(w, gety(i-halfwin, i+halfwin)) 
               for i, p in zip(range(len(y)), x) ]


def _set_time_axis(params):
    if int(params.get("maptime", 1)) == 1:
        pyplot.xlabel(r"time, $\tau_i$", fontsize=25)
        set_xaxis()
    else:
        pyplot.xlabel(r"time, $\tau_i$ (days)", fontsize=25)


def _smoothing(x, y, args, params):
    if params.get("smoothing_mode", -1) < 0:    
        return x, y
    if params.get("smoothing_mode", 0) == 0:    
        return _smooth_around_badges(x, y, args.badges, params.get("smoothing", 5))
    return x, _smooth(y, params.get("smoothing", 5))    


def _plot_badge_labels(args):
    for v, l in zip(args.badges, args.labels):
        ymin, _ = pyplot.gca().get_ylim()
        pyplot.text((v), ymin, l, rotation=90, fontsize=25, color="black",
                    verticalalignment="bottom", horizontalalignment="right")

def _plot_badges(args):
    for v, _ in zip(args.badges, args.labels):
        pyplot.axvline(x=(v), color="black", lw=2)
    _plot_badge_labels(args)
    
        
def _plot_badges2(args):
    for v, l in zip(args.badges, args.labels):
        _, ymax = pyplot.gca().get_ylim()
        pyplot.axvline(x=(v), color="black", lw=2)
        pyplot.text((v), ymax*0.95, l, rotation=90, 
                    verticalalignment="top", horizontalalignment="right")
    

def _plot_pvalue_threshold(th, verticalalignment="bottom", badge=None):
    xmin, xmax = pyplot.gca().get_xlim()
    pyplot.axhline(y=th, color="red", lw=4, ls="--")
    
    if badge is not None and abs(badge-xmin)<abs(xmax-badge): 
        pyplot.text(xmax, th, " p=%s" % th, color="red",
                verticalalignment=verticalalignment, horizontalalignment="right", fontsize=20)
    else:
        pyplot.text(xmin, th, " p=%s" % th, color="red",
                verticalalignment=verticalalignment, horizontalalignment="left", fontsize=20)


def plot_legend(**kwargs):
    ax1 = pyplot.gca()
    fontsize = kwargs.pop("fontsize", 20)
    loc = kwargs.pop("loc", 20)
    leg = ax1.legend(fontsize=fontsize, fancybox=False, loc=loc)
    leg.get_frame().set_alpha(1.0)
    frame = leg.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    frame.set_linewidth(0.5)    


def pyplot_parse_params(kwargs):
    if "title" in kwargs:
        pyplot.title(kwargs.pop("title", ""))
    if "xlabel" in kwargs:
        pyplot.xlabel(kwargs.pop("xlabel", ""), fontsize=25)
    if "ylabel" in kwargs:
        pyplot.ylabel(kwargs.pop("ylabel", ""), fontsize=25)
    if "grid" in kwargs:
        pyplot.grid(bool(kwargs.pop("grid", True)))
    
    legend_loc = kwargs.pop("legend_loc", 0)
    if bool(kwargs.pop("legend", True)):
        plot_legend(loc=legend_loc)
    else:
        if pyplot.gca().legend() is not None:
            pyplot.gca().legend().set_visible(False)
        
    if "xmin" in kwargs or "xmax" in kwargs:
        xmin, xmax = pyplot.gca().get_xlim()
        xmin2 = kwargs.pop("xmin", xmin)
        xmax2 = kwargs.pop("xmax", xmax)
        if xmin2 is not None: 
            try:    xmin = float(xmin2)
            except: xmin = xmin2
        if xmax2 is not None: 
            try:    xmax = float(xmax2)
            except: xmax = xmax2
        pyplot.xlim( (xmin, xmax) )    
    
    if "ymin" in kwargs or "ymax" in kwargs:
        ymin, ymax = pyplot.gca().get_ylim()
        ymin2 = kwargs.pop("ymin", ymin)
        ymax2 = kwargs.pop("ymax", ymax)
        if ymin2 is not None: 
            try:    ymin = float(ymin2)
            except: ymin = ymin2
        if ymax2 is not None: 
            try:    ymax = float(ymax2)
            except: ymax = ymax2
        pyplot.ylim( (ymin, ymax) )      

    if "xlim" in kwargs:
        pyplot.gca().set_xlim(kwargs.pop("xlim"))
    if "ylim" in kwargs:
        pyplot.gca().set_ylim(kwargs.pop("ylim"))
    
    if "xlog" in kwargs and bool(kwargs.pop("xlog"))==True:
        pyplot.xscale("log")
    if "ylog" in kwargs and bool(kwargs.pop("ylog"))==True:
        pyplot.yscale("log")
        
    return kwargs


def pyplot_parse_params2(**kwargs):
    pyplot_parse_params(kwargs)





if __name__=="__main__":
    
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="""Visualises outputs of the sliding_window.py.""")
    
    parser.add_argument("-p", "--params", dest='params', 
                        help="comma-separated params: option=value", 
                        required=False, default="")
    
    parser.add_argument("-i", "--input", dest='input', 
                        help="an input TSV file",
                        required=True)    
    parser.add_argument("-o", "--output", dest='output', 
                        help="output files prefix", 
                        required=False, default=None)    

    parser.add_argument("-z", "--start", dest='start', type=int, help="start time",  
                         required=False, default=0)    
    parser.add_argument("-e", "--end", dest='end', type=int, help="end time",  
                         required=False, default=100000)    

    parser.add_argument("-n", "--neighbourhood", dest='neighbourhood', type=float,
                         help="neighbourhood size",  
                         required=False, default=180)       

    parser.add_argument("-m", "--model", dest='model', 
                         help="model B for robust survival/S for survival/F for fisher",  
                         required=False, default="B")       
    
    parser.add_argument("-b", "--badges", dest='badges', type=float, nargs="+",
                         help="badge introduction times",  
                         required=False, default=[])       
    parser.add_argument("-l", "--labels", dest='labels', nargs="+",
                         help="badge labels",  
                         required=False, default=[])       
    
    args = parser.parse_args(sys.argv[1: ])    
    print(" args: %s" % args)
    params = parsing.parse_dictionary(args.params)
    print(" params: %s" % params)
    
    if params.get("xmin", None) is not None:
        params["xmin"] = (params["xmin"]) 
    if params.get("xmax", None) is not None:
        params["xmax"] = (params["xmax"]) 

    if args.output is None:
        args.output = args.input+"_sliding"
        
    if len(args.labels)>0:
        assert len(args.labels)==len(args.badges)
    else:
        args.labels = [("badge at %.0f" % b) for b in args.badges]

    assert len(args.badges)>0
    
    
    
    model = args.model[0]
    print("model = %s" % model)
    print("TODO: CURRENTLY MODEL SELECTION WORKS ONLY FOR LLR-DIST PLOTS")
    
    #########################################################################    
    #########################################################################
    
    print("=================================")
    print("reading data from %s" % args.input)
    data = pd.read_csv(args.input, sep="\t")
    data = unify_column_names(data)
    data = ensure_llr_values(data) 
    data = data.sort_values([TIMECOL])
    
    print("=================================")
    print("filtering rows")
    #data subsets selection: for plotting sample plots from syntethic data    
    for fix in ["trial_no", "l0", "l1", "r0", "r1", "a0", "a1"]: 
        if params.get(fix, None) is not None:    
            print(" filtering rows with %s" % fix)
            data = data[data[fix]==float(params[fix])]
    print(" filtering data to fit the interval [%s, %s]" % (args.start, args.end))
    data = data[(data[TIMECOL]>=args.start) & (data[TIMECOL]<=args.end)]

    print("=================================")
    print("updating data with derived ratios")
    data["BH1_a0"] = data["BH1_r0"]/data["BH1_lambda0"]
    data["BH1_a1"] = data["BH1_r1"]/data["BH1_lambda1"]
    data["BHf_a0"] = data["BHf_r0"]/data["BHf_lambda0"]
    data["BHf_a1"] = data["BHf_r1"]/data["BHf_lambda1"]
    data["BHf_a1a0"] = data["BHf_a1"]/data["BHf_a0"]
    data["BH1_a1a0"] = data["BH1_a1"]/data["BH1_a0"]

    data["BH1_s0"] = data["BH1_r0"].apply(math.sqrt)/data["BH1_lambda0"]
    data["BH1_s1"] = data["BH1_r1"].apply(math.sqrt)/data["BH1_lambda1"]
    data["BHf_s0"] = data["BHf_r0"].apply(math.sqrt)/data["BHf_lambda0"]
    data["BHf_s1"] = data["BHf_r1"].apply(math.sqrt)/data["BHf_lambda1"]  


      
    print("=================================")
    print("extracting fitted alphas for particular models (time intervals)")
    mask = data["alt_model"]==0

    data["a0"] = data["BHf_a0"]
    data["a1"] = data["BHf_a1"]
    data.loc[~mask, "a0"] = data.loc[~mask, "BH1_a0"]
    data.loc[~mask, "a1"] = data.loc[~mask, "BH1_a1"]
    
    data["s0"] = None
    data["s1"] = None 
    data.loc[~mask, "s0"]   = data.loc[~mask, "BH1_s0"]
    data.loc[mask, "s0"]    = data.loc[mask, "BHf_s0"] 
    data.loc[~mask, "s1"]   = data.loc[~mask, "BH1_s1"]
    data.loc[mask, "s1"]    = data.loc[mask, "BHf_s1"] 
        
    badges = args.badges
    alphas = numpy.zeros((len(data), len(badges)+1))*float("nan")
    stds = numpy.zeros((len(data), len(badges)+1))*float("nan")
    for i, (a0, s0, a1, s1, alt_model, num_badges) in enumerate(zip(data["a0"], data["s0"], 
                                                                    data["a1"], data["s1"], 
                                                                    data["alt_model"], 
                                                                    data["prev_badges"])):
        alphas[i, num_badges] = a1
        stds[i, num_badges] = s1
        if alt_model!=0: 
            alphas[i, num_badges-1] = a0
            stds[i, num_badges-1] = s0 
    
    alphas = pd.DataFrame(alphas).rename(columns=dict((i, "fitted_alpha%i" % i) for i in range(alphas.shape[1])))
    alphas[pd.isnull(alphas)] = None 

    stds = pd.DataFrame(stds).rename(columns=dict((i, "fitted_std%i"%i) for i in range(stds.shape[1])))
    stds[pd.isnull(stds)] = None 
    
    #data = pd.concat([data, alphas, stds], axis=1, ignore_index=True)
    for col1, col2 in zip(alphas.columns, stds.columns):
        data[col1] = list(alphas[col1])
        data[col2] = list(stds[col2])    
    
    
    print("=================================")
    print("Data preview:")
    print(data.head())
    print(len(data), "rows")
    
    #data.to_csv("/tmp/enriched.csv", header=True, sep="\t", index=False)

    #########################################################################
    #########################################################################
    #########################################################################
        
    print("=================================")    
    print("alphas over time")
    plots.pyplot_reset()
    for i in range(len(args.badges)+1):
        print(" printing %s" % ("fitted_alpha%i" % i))
        means, stds = numpy.array(data["fitted_alpha%i" % i]), numpy.array(data["fitted_std%i" % i])
        means, stds = _smooth(means, 10), _smooth(stds, 10)
        p = pyplot.plot(list(data[TIMECOL]), means, label=r"$\mathbb{E}[\lambda_%i(u)]$" % i, lw=3, color=COLORS[i%len(COLORS)])
        #pyplot.plot(data[TIMECOL], means+stds, color=p[-1].get_color(), lw=1)
    pyplot.ylabel("average intensity (days)", fontsize=25)
    plots.pyplot_parse_params2(xmin=params.get("xmin", None), xmax=params.get("xmax", None), ymax=params.pop("ymax", None))
    pyplot.grid(True)
    _plot_badges(args)
    xmin, xmax = pyplot.xlim()
    plot_legend(fontsize=20, loc=(1 if args.badges[0]<(xmin+xmax)+0.5 else 2))
    _set_time_axis(params)        
    pyplot.tick_params(axis='both', which='major', labelsize=22)
    pyplot.gcf().subplots_adjust(bottom=0.17, left=0.22)
    plots.savefig(args.output+"_fitting.pdf")

    
    #########################################################################   
         
    print("=================================")
    print("LLR-values over time")
    plots.pyplot_reset()
    #_plot_badges2(args)
    #transform = lambda c: list(c.apply(lambda v: numpy.exp(v)))
    transform = lambda c: list(c)
            
    x, y = _smoothing(data[TIMECOL], transform(data["SHfH0_LLR"]), args, params)
    pyplot.plot(list(x), y, label="basic", lw=3, ls="-", color=COLORS[1])
    #pyplot.plot(x[5::15], y[5::15], marker="o", markeredgecolor="none", markersize=5, color=COLORS[1], lw=0)
    
    x, y = _smoothing(data[TIMECOL], transform(data["BHfH0_LLR"]), args, params)
    pyplot.plot(list(x), y, label="robust", lw=3, ls="-", color=COLORS[2])
    #pyplot.plot(x[10::15], y[10::15], marker="*", markeredgecolor="none", markersize=5, color=COLORS[2], lw=0)

    plots.pyplot_parse_params2(grid=True)
    plots.pyplot_parse_params(params.copy())
    
    #pyplot.ylim((0, 30))
    #ymin, ymax = pyplot.ylim()
    #pyplot.yscale("log") 
    plot_legend(fontsize=20, loc=1)
    #_plot_pvalue_threshold(0.05, badge=args.badges[0])
    _plot_badges(args)
    pyplot.ylabel(r"test statistic, $LLR$", fontsize=25)
    _set_time_axis(params)    
    pyplot.tick_params(axis='both', which='major', labelsize=22)
    pyplot.gcf().subplots_adjust(bottom=0.17, left=0.18)
    plots.savefig(args.output+"_llr_over_time.pdf")    
    
    #########################################################################
    print("=================================")   
    print("Theoretical p-values")
    for bno, (b, l) in enumerate(zip(args.badges, args.labels)):                         
        print("Theoretical FHfH0_pval:", extract_closest_value(data, "FHfH0_pval", b))
        print("Theoretical SHfH0_pval:", extract_closest_value(data, "SHfH0_pval", b))
        print("Theoretical BHfH0_pval:", extract_closest_value(data, "BHfH0_pval", b))
        
    #########################################################################
    
    print("=================================")   
    print("How extreme is our LLR-statistic value") 
    llr_col = "%sHfH0_LLR" % model
     
    plots.pyplot_reset()
    virtual_badge_statistics = list(data[data["badge_no"]<0][llr_col])
    if len(virtual_badge_statistics)<=0:
        print("ERROR: it seems that there are no cases not overlapping with any badge!")
    else:
        print(" virtual_badge_statistics=", str(virtual_badge_statistics)[:300])
        #plots.densities_plot({"data": virtual_badge_statistics}, show=False, 
        #                     bw=params.get("bw", 0.05), shade=True, color="#2171b5", lw=2, cut=0)
        sns.kdeplot(numpy.array(virtual_badge_statistics, dtype=float), 
                    legend=False, bw=params.get("bw", 0.05),
                    shade=True, color="#2171b5", lw=2, cut=0)
                    
        for bno, (b, l) in enumerate(zip(args.badges, args.labels)):            
            llr = extract_closest_value(data, llr_col, b)             
            p1 = calc_empirical_pvalue(data, llr_col, b)
            ymin, ymax = pyplot.gca().get_ylim()        
            pyplot.axvline(x=llr, color="black", lw=2)
            pyplot.text(llr-0.003, ymin, l, 
                        rotation=90, color="black", va="bottom", ha="right", fontsize=20)
            pyplot.text(llr+0.003, ymin, 
                        '%s, %s' % (r"$LLR=%.3f$" % llr, _fmt_p(p1, r"p")), 
                        rotation=90, color="black", va="bottom", ha="left", fontsize=20)
                            
        plots.pyplot_parse_params2(legend=False)
        pyplot.xlabel(r"background test statistic, $LLR$", fontsize=25)
        pyplot.ylabel(r"density", fontsize=25)
        pyplot.tick_params(axis='both', which='major', labelsize=22)
        pyplot.gcf().subplots_adjust(bottom=0.15, left=0.15)
        pyplot.grid(True)
        xmin, xmax = pyplot.xlim()
        plots.pyplot_parse_params2(xmin=min(llr-1.0,xmin), xmax=max(xmax,llr+1.0))
        pyplot.gcf().subplots_adjust(bottom=0.17, left=0.18)
        plots.savefig(args.output+"_distribution_%s.pdf" % llr_col, reset=False)
    #########################################################################
        
    print("=================================")           
    print("Created plots: %s" % ", ".join(plots.get_saved_paths()))
    sys.exit(0)
    print("WARNING: The code below is not used in the final version")
    
    #########################################################################   
         
    print("=================================")
    print("p-values over time")
    plots.pyplot_reset()
    #_plot_badges2(args)
        
    x, y = _smoothing(data[TIMECOL], data["FHfH0_pval"], args, params)
    pyplot.plot(list(x), y, label="counts", lw=3,  ls="-", color=COLORS[0])
    #pyplot.plot(x[0::15], y[0::15], marker="h", markeredgecolor="none", markersize=5, color=COLORS[0], lw=0)
    
    x, y = _smoothing(data[TIMECOL], data["SHfH0_pval"], args, params)
    pyplot.plot(list(x), y, label="basic", lw=3, ls="-", color=COLORS[1])
    #pyplot.plot(x[5::15], y[5::15], marker="o", markeredgecolor="none", markersize=5, color=COLORS[1], lw=0)
    
    x, y = _smoothing(data[TIMECOL], data["BHfH0_pval"], args, params)
    pyplot.plot(list(x), y, label="robust", lw=3, ls="-", color=COLORS[2])
    #pyplot.plot(x[10::15], y[10::15], marker="*", markeredgecolor="none", markersize=5, color=COLORS[2], lw=0)

    plots.pyplot_parse_params2(ylog=True, grid=True)
    plots.pyplot_parse_params(params.copy())
    
    ymin, ymax = pyplot.ylim() 
    if ymin>=0.01:  plot_legend(fontsize=20, loc=1)
    else:           plot_legend(fontsize=20, loc=4)
    _plot_pvalue_threshold(0.05, badge=args.badges[0])
    _plot_badges(args)
    pyplot.ylabel(r"$p$-value", fontsize=25)
    _set_time_axis(params)    
    pyplot.tick_params(axis='both', which='major', labelsize=22)
    pyplot.gcf().subplots_adjust(bottom=0.15, left=0.15)
    plots.savefig(args.output+"_pvals_over_time.pdf")
    
    #########################################################################
    
    
    print("=================================")
    print("Fake badges vs alpha0/alpha1 ratios") 
    plots.pyplot_reset()
    
    df = data[data["badge_no"]<0]
    
    plots.pyplot_reset()
    pyplot.tick_params(axis='x', which='major', labelsize=22)
    pyplot.tick_params(axis='y', which='major', labelsize=22)
    pyplot.ylabel("p-value", fontsize=25)
    pyplot.xlabel(r"expected hazards ratio ($\hat{\alpha_1}/\hat{\alpha_0}$)", fontsize=25)
    pyplot.grid(True)
    pyplot.scatter(df["BHf_a1a0"], df["BHfH0_pval"], s=50, #s=df["BHf_lambda0"], 
                   color=COLORS[0], label="background", edgecolor="none")
    pyplot.ylim((0.0, 1.1))
    
    print(" %i cases with BHfH0_pval<0.05" % (sum(df["BHfH0_pval"]<0.05)))
    #print(df[df["BHfH0_pval"]<0.05].head())
    #print(" lambda0:")
    #print(df["BHf_lambda0"].head())

    for bno, (b, l) in enumerate(zip(args.badges, args.labels)):
        badge = find_badge_rows(data, b)
        p, ratio, l = badge["BH1H0_pval"].mean(), badge["BHf_a1a0"].mean(),  badge["BH1_lambda0"].mean()
        print(" badge id=%i: p=%s ratio=%s lambda=%s" % (bno, p, ratio, l))
        if ratio>100000:    print("WARNING: a1/a0 ratio=", ratio)
        else:   pyplot.scatter([ratio], [p], s=150, marker="h",#s=[l],
                                color="black", 
                                label=r"badge ($\hat{\lambda}=%g$)" % l, edgecolor="none")
        
    plot_legend(fontsize=20, loc=5)
    pyplot.tick_params(axis='both', which='major', labelsize=22)
    _plot_pvalue_threshold(0.05, verticalalignment="bottom", badge=args.badges[0])
    pyplot.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plots.savefig(args.output+"_pval_ratios.pdf")
    
    #########################################################################    

    
    