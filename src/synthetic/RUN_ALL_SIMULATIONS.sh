#!/bin/sh

DIR="../../data/synthetic_results/"
mkdir -p $DIR

CPU=3 #How many cores to use

LAMBDA_FITTING_MODE=",fitting_mode=4" #grid search for lambda0, common lambda for all robust models fitting
#LAMBDA_FITTING_MODE=",fitting_mode=4,independent_lambdas=1" #grid search for lambda0, separate lambdas for all robust models fitting
#LAMBDA_FITTING_MODE="" #use lambda0 from generated data

######################################

echo "SLIDING WINDOW DATA GENERATION"
python sliding_window.py -o "$DIR/synthetic" -l -s 1 -w 60 -c $CPU -p "T=360,mode=2$LAMBDA_FITTING_MODE"
python sliding_window.py -o "$DIR/synthetic_t001switch" -l -s 1 -w 60 -c $CPU -p "T=360,trend=0.001,mode=2$LAMBDA_FITTING_MODE" 

######################################

echo "SLIDING WINDOW DATA PLOTTING"
#python sliding_window_plot.py -i "$DIR/synthetic.tsv" #plot for all params
python sliding_window_plot.py -i "$DIR/synthetic.tsv" -p "l0=10" -minr 1.0
python sliding_window_plot.py -i "$DIR/synthetic_t001switch.tsv" -p "l0=10" -minr 1.0
python sliding_window_plot.py -i "$DIR/synthetic_t001switch.tsv" -p "l0=1000" -minr 1.0

#survival plots used in the publication
python sliding_window_plot_survival.py -i "$DIR/synthetic_t001switch.tsv" -p "l0=10" -minr 1.0

######################################

