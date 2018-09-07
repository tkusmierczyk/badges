#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")

import pandas as pd
from analytics.config import ZERO_TIME


def convert(t):
    try:
        return (pd.Timestamp(ZERO_TIME) + pd.Timedelta(days=float(t)))
    except:
        return ((pd.Timestamp(t)-pd.Timestamp(ZERO_TIME)).total_seconds() / (24.0*3600))


if __name__=="__main__":
    #Converts time from realtime to float number and vice versa
    
    try:
        arg = sys.argv[1]
    except:
        print("One argument expected: time string or time float value (number of days).")
        sys.exit(-1)
        
    print(convert(arg))