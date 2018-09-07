#!/bin/bash

#Input and output directory
DIR="../../data/badges/"

#Sliding window size
WIN="60"

#Badge introduction time - Tag Wiki Edits
TAGEDIT_BADGE="756"

#Badge introduction time - bounties
BOUNTY_BADGE="703.5"

#############################################################
echo "CALCULATING SLIDING WINDOW TEST STATISTICS"

CPUS=3 #How many cores to use
PARAMS="-p fitting_mode=3 -c $CPUS"

python sliding_window.py $PARAMS -i $DIR/tagedits/tageditor.tsv -o $DIR/tagedits/tageditor_w${WIN}_fitting.tsv -z 690 -e 900 -w $WIN -m 5 -s 1 -b $TAGEDIT_BADGE
python sliding_window.py $PARAMS -i $DIR/bounties/bounty1.tsv -o $DIR/bounties/bounty1_w${WIN}_fitting.tsv -z 500 -e 1500 -w $WIN -m 5 -s 1 -b $BOUNTY_BADGE 
python sliding_window.py $PARAMS -i $DIR/bounties/bounty2.tsv -o $DIR/bounties/bounty2_w${WIN}_fitting.tsv -z 500 -e 1500 -w $WIN -m 5 -s 1 -b $BOUNTY_BADGE 

#############################################################
echo "EXTRACTION OF USERS APPEARING IN EACH SLIDING WINDOW"
python sliding_window_extract_users.py $PARAMS -i $DIR/tagedits/tageditor.tsv -o $DIR/tagedits/tageditor_w${WIN}_users.tsv -z 690 -e 900 -w $WIN -m 5 -s 1 -b $TAGEDIT_BADGE
python sliding_window_extract_users.py $PARAMS -i $DIR/bounties/bounty1.tsv -o $DIR/bounties/bounty1_w${WIN}_users.tsv -z 500 -e 1500 -w $WIN -m 5 -s 1 -b $BOUNTY_BADGE 
python sliding_window_extract_users.py $PARAMS -i $DIR/bounties/bounty2.tsv -o $DIR/bounties/bounty2_w${WIN}_users.tsv -z 500 -e 1500 -w $WIN -m 5 -s 1 -b $BOUNTY_BADGE 

#############################################################
echo "PLOTTING SLIDING WINDOW RESULTS"
python sliding_window_plot.py  -z 720 -e 840 -i $DIR/tagedits/tageditor_w${WIN}_fitting.tsv  -b $TAGEDIT_BADGE -l "Tag Editor intro" -p "title=,legend=t,xlabel=sliding window center [days],ylabel=p-value,xmin=720,xmax=840,ymax=0.0015,smoothing=5,smoothing_mode=1,bw=0.1"
python sliding_window_plot.py -i $DIR/bounties/bounty1_w${WIN}_fitting.tsv -b $BOUNTY_BADGE -l "Promoter intro" -p "title=,legend=t,xlabel=sliding window center [days],ylabel=p-value,xmax=1000,xmin=650,ymax=0.005,bw=0.6,smoothing=5,smoothing_mode=1"
python sliding_window_plot.py -i $DIR/bounties/bounty2_w${WIN}_fitting.tsv -b $BOUNTY_BADGE -l "Investor intro" -p "title=,legend=t,xlabel=sliding window center [days],ylabel=p-value,xmax=1000,xmin=650,ymax=0.001,bw=0.2,smoothing=5,smoothing_mode=1"

#############################################################
echo "VALIDATING SMD SCORES"
python sliding_window_extract_users_smd.py -i $DIR/tagedits/tageditor_w${WIN}_users.tsv
python sliding_window_extract_users_smd.py -i $DIR/bounties/bounty1_w${WIN}_users.tsv
python sliding_window_extract_users_smd.py -i $DIR/bounties/bounty2_w${WIN}_users.tsv


