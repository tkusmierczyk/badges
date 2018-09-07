#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Calculates SMD scores using results (user lists for each of the windows) from 
    sliding_window_extract_users.py and a file with user characteristics."""
    
import pandas as pd
import argparse
import logging
import numpy as np
import sys

sys.path.append("../")
from aux import parsing 


def _setup(args):
    params = parsing.parse_dictionary(args.params)
    print("params: %s" % params)
    
    try:    params["randomize"] = args.randomize
    except: pass        
    try:    verbose = args.verbose
    except: verbose = False    
    try:    debug = args.debug
    except: debug = False    
    
    np.random.seed(123)
    level = logging.DEBUG if debug else logging.INFO            
    
    fmt = '[%(process)4d][%(asctime)s][%(levelname)-5s][%(module)s:%(lineno)d/%(funcName)s] %(message)s'
    logging.basicConfig(level=level, format=fmt)
    
    if verbose:
        logging.getLogger().addHandler(logging.StreamHandler())
    
    return params


def smd(l0, l1):
    sd0, sd1 = np.std(l0), np.std(l1)
    sdpooled = np.sqrt(0.5*(sd0*sd0 + sd1*sd1))
    return (np.mean(l1)-np.mean(l0)) / sdpooled
    

def extract_smd_values(users_data, control_df, badge_users, feature, M=1000):
    smd_feature, smd_na = [], []
    f0 = users_data[users_data["userid"].isin(badge_users)][feature]
    control_users = list(control_df["users"])    
    #np.random.shuffle(control_users)
    userid2feature = dict(zip(users_data["userid"], users_data[feature]))
    for control in control_users[:M]:
        users = set(map(int, control.split(",")))
        #f1 = users_data[feature][users_data["userid"].isin(users)]
        f1 = pd.Series([userid2feature[userid] for userid in users if userid in userid2feature])
        smd_feature.append((smd(f0[~f0.isnull()], f1[~f1.isnull()])))
        smd_na.append((smd(np.asarray(f0.isnull(), dtype=int), np.asarray(f1.isnull(), dtype=int))))
    smd_feature, smd_na = np.array(smd_feature), np.array(smd_na)
    return smd_feature, smd_na


USER_FEATURES = "../../data/badges/user_features.tsv"

def main(argv):
    parser = argparse.ArgumentParser(description="""Calculates SMD scores using results 
    (user lists for each of the windows) 
    from sliding_window_extract_users.py 
    and a file with user characteristics (%s).""" % USER_FEATURES)
    
    parser.add_argument("-i", "--input", dest='input', 
                        help="a list of input TSV files with samples (three cols: id, time, type).",
                        required=True)    
    parser.add_argument("-p", "--params", dest='params', 
                        help="comma-separated params: option=value", 
                        required=False, default="")    

    parser.add_argument("-v", "--verbose", dest='verbose', 
                        help="verbose", 
                        required=False, default=False, action="store_true")
    parser.add_argument("-d", "--debug", dest='debug', 
                        help="debug", 
                        required=False, default=False, action="store_true")
    parser.add_argument("-c", "--cpu", dest='cpu', 
                        help="num processes to use", 
                        required=False, type=int, default=1)           
        
            
    args = parser.parse_args(argv)    
    _setup(args)
    
    ###################################################################################################################

    print("@TODO: loading user features from the file given as an argument")
    users_data =  pd.read_csv(USER_FEATURES, sep="\t")
    
    #########################################################################    
    
    logging.info("Reading data from %s" % args.input)
    df = pd.read_csv(args.input, sep="\t")
    logging.info(df.head())
    logging.info(" %i rows loaded" % len(df["time"].unique()))

    #########################################################################
    
    logging.info("Calculating SMD scores.")

    badge_row = df.iloc[abs(df["badge_time"]-df["time"]).idxmin()]
    assert badge_row["badge_no"]==0, "row found to be the closest to the badge is wrong"
    logging.info(" badge row = %s" % badge_row)
    badge_users = list(map(int, badge_row["users"].split(",")))
    
    fmt = lambda v: "?" if np.isnan(v) else ("%.2f" % v) 
    control_df = df[df["badge_no"]<0]
    for feature in ['reputation', 'age', 'views', 'upvotes', 'downvotes']:         
        smd_feature, smd_na = extract_smd_values(users_data, control_df, badge_users, feature)

        print("%s & %s & %s  \\\\" % (feature, 
                                      fmt(np.mean(abs(smd_feature))), 
                                      fmt(np.std(abs(smd_feature)) )))
        if sum(~np.isnan(smd_na))==0: continue
        print("%s-NA & %s & %s  \\\\" % (feature, 
                                         fmt(np.mean(abs(smd_na))), 
                                         fmt(np.std(abs(smd_na)) )))
        
    #########################################################################

if __name__=="__main__":
    main(sys.argv[1: ])    
    
    