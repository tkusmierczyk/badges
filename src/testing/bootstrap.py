#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Helpers to calculate p-value from empirical distribution of the test statistic."""


import numpy as np
import sys

sys.path.append("../")
from testing.wilks_test import llr_from_pvalue

keep_valid = lambda vals: list(filter(lambda v: v is not None and not np.isnan(v) and v!=float("inf") and v!=-float("inf"), vals))


TIME_COLUMN = "wc"
BADGE_NO_COLUMN = "badge_no"


def find_badge_rows(data, t):
    distances = abs(data[TIME_COLUMN] - t)
    selector = distances == distances.min()
    closest = data[selector]
    return closest


def extract_closest_value(df, value_column, t):
    closest = find_badge_rows(df, t)
    value = float(closest[value_column].mean())
    return value


def calc_empirical_pvalue(df, statistic_column, badge_time, tail=1):
    """
    
        Args:
            tail  1=right tail/-1=left tail
    """
    virtual_badge_statistics = df[df[BADGE_NO_COLUMN]<0][statistic_column]
    true_badge_statistic     = extract_closest_value(df, statistic_column, badge_time)           
    more_extreme = (lambda s: s>true_badge_statistic) if tail==1 else (lambda s: s<true_badge_statistic)         
    empirical_pvalue = sum(1.0 for s in virtual_badge_statistics if more_extreme(s)) / len(virtual_badge_statistics)    
    return empirical_pvalue


def unify_column_names(df):
    df.rename(columns={"Hf_fisher_p": "FHfH0_pval",
                       "H1_fisher_p": "FH1H0_pval",
                       "fisher_p": "H1_fisher_p"}, inplace=True)
    df.rename(columns={"H0_a0": "SH0_a0", "H0_a1": "SH0_a1", "H0_ll": "SH0_ll", 
                       "Hf_a0": "SHf_a0", "SHf_a1": "SHf_a1", "Hf_ll": "SHf_ll", "HfH0_pval": "SHfH0_pval", 
                       "H1_a0": "SH1_a0", "H1_a1": "SH1_a1", "H1_ll": "SH1_ll", "H1H0_pval": "SH1H0_pval",
                       "FH1_pval": "FH1H0_pval", "FHf_pval": "FHfH0_pval"}, inplace=True)             
    for src, dst in {"FHfH0_pval": "F_p", "SHfH0_pval": "S_p", "BHfH0_pval": "B_p"}.items():
        if src not in df.columns: continue
        df[dst] = df[src]                
    return df


def ensure_llr_values(df):
    if "BHfH0_LLR" not in df.columns:
        print("WARNING: BHfH0_LLR restored from pvalues!")
        df["BHfH0_LLR"] = df["BHfH0_pval"].apply(llr_from_pvalue)
    if "BH1H0_LLR" not in df.columns:
        print("WARNING: BH1H0_LLR restored from pvalues!")        
        df["BH1H0_LLR"] = df["BH1H0_pval"].apply(llr_from_pvalue)
    if "SHfH0_LLR" not in df.columns:
        print("WARNING: SHfH0_LLR restored from pvalues!")        
        df["SHfH0_LLR"] = df["SHfH0_pval"].apply(llr_from_pvalue)
    if "SH1H0_LLR" not in df.columns:        
        print("WARNING: SH1H0_LLR restored from pvalues!")
        df["SH1H0_LLR"] = df["SH1H0_pval"].apply(llr_from_pvalue)
    return df

