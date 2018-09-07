Code used in the following paper:

**T. Kusmierczyk, M. Gomez-Rodriguez: [On the Causal Effect of Badges.](https://arxiv.org/abs/1707.08160) WWW 2018.** [(SLIDES)](https://www.slideshare.net/TomaszKusmierczyk/on-the-causal-effect-of-digital-badges)


-----------------------------------------------------------------------------------------------------------------

### Framework for passive causal inference   
**We developed a novel causal inference framework designed for cases where the goal is to validate an effect of the change applied to a group of users at a certain time point.**

Our framework avoids modeling the mechanisms underlying individual user actions and instead adopts a data-driven approach based on survival analysis 
and statistical hypothesis testing. At the heart of our approach there are two technical innovations: 

1. a robust survival-based hypothesis testing procedure, inspired by the discrete choice literature on latent variable models, 
which allows us to account for the utility heterogeneity,
1. a *bootstrap difference-in-differences* method, inspired by the economics literature on natural experiments, which allows us to control for 
the random fluctuations in users'{} behavior over time.


### Application: first-time digital badges

A wide variety of online platforms use digital badges to encourage users to take certain types of desirable actions. However, despite their growing popularity, their causal effect on users' behavior is not well understood. This is partly due to the lack of counterfactual data and the myriad of complex factors that influence users' behavior over time. As a consequence, their design and deployment lacks general principles.

We focus on first-time badges, which are awarded after a user takes a particular type of action for the first time, and study their causal effect by harnessing the delayed introduction of several badges in a popular Q&A website. In doing so, we introduce a novel causal inference framework for badges whose main technical innovations are a robust survival-based hypothesis testing procedure, which controls for the utility heterogeneity across users, and a bootstrap difference-in-differences method, which controls for the random fluctuations in users' behavior over time.


### Data

In our experiments, we used [StackOverflow](https://stackoverflow.com/) data [archive](https://archive.org/details/stackexchange) that was processed to extract the following files:

  * ``data/badges/tagedits/tageditor.tsv`` - data related to TagEditor badge
  * ``data/badges/bounties/bounty1.tsv`` - data related to the bounty badge of type1
  * ``data/badges/bounties/bounty2.tsv`` - data related to the bounty badge of type2
  * ``data/badges/user_features.tsv`` - users' characteristics necessary to calculate SMDs
  
Files are TAB separated and every included event consist of user id, time and event type (either eligiblity time, action time or data censoring). 
  

### Code 

* Real data experiments - ``src/experiments``:
  * ``RUN_ALL_EXPERIMENTS.sh`` - Runs main experiments for TagEditor badge and bounties and plots the results. 
  * ``sliding_window.py`` - Calculates p-values from various models (Fisher=counts, Simple=simple survival, Robust='Bayesian' with priors) on real world data using sliding window. Outputs a TSV file containing the test statistics for each time point.
  * ``sliding_window_plot.py`` - Visualises outputs of the ``sliding_window.py``.
  * ``sliding_window_extract_users.py`` - Extracts users appearing in each of the (sliding) window.
  * ``sliding_window_extract_users_smd.py`` - Calculates SMD scores using results (user lists for each of the windows) and users' characteristics (``data/badges/user_features.tsv``).

* Synthetic data experiments - ``src/synthetic``:
  * ``RUN_ALL_SIMULATIONS.sh`` - Runs main experiments on synthetic data and plots the results. 
  * ``sliding_window.py`` - Calculates p-values from various models  (Fisher=counts, Simple=simple survival, Robust='Bayesian' with priors) using sliding window over synthetically generated data. Outputs a TSV file containing the test statistics for each time point.
  * ``sliding_window_plot.py`` - Analysis (including empirical p-value calculation) and plotting of output data from ``sliding_window.py``
  * ``sliding_window_plot_survival.py`` - Plotting intensity values extracted from output data calculated in ``sliding_window.py``
  * ``thinning.py`` - Sampling from non-homogenous Poisson temporal point processes with known maximum.

* Modeling and fitting of non-homogenous Poisson temporal point processes - ``src/processes``:
  * ``process.py`` - Model of a survival process with one hazard rate until switching point and another after that.
  * ``process_factory.py`` - Factory that produces processes (``process.py``)
  * ``trend_process.py`` - Model of a survival process with hazard rate a1+trend\*time until switching point and a2+trend\*time after that.
  * ``trend_process_factory.py`` - Factory that produces processes (``trend_process.py``)
  * ``fit_simple.py`` - Fits the simple (intensities shared between users) survival model in two cases. First, assuming intensity to be constant (H0) and second, allowing for intensity change around certain time point (H1).
  * ``fit_bayes.py`` - Base functions for fitting of the robust (bayesian with priors) survival model.
  * ``fit_bayes1.py`` - Fits the robust (Bayesian with priors) survival model under H0.
  * ``fit_bayes2.py`` - Fits the robust (Bayesian with priors) survival model under H1.

* Statistical tests - ``src/testing``:
  * ``fisher_test.py`` - Calculation of p-values from Fisher exact test on counts (before vs after change).
  * ``wilks_test.py`` - Calculation of p-values from log-likelihood ratio test of nested models using Chi2 distribution (Wilks theorem).
  * ``bootstrap.py`` - Helpers to calculate p-value from empirical distribution of the test statistic.

* Auxiliary files used for plotting, arguments parsing and data processing - ``src/analytics``, ``src/aux``


### Execution and multiprocessing

Two main scripts to reproduce our results are ``src/experiments/RUN_ALL_EXPERIMENTS.sh`` and ``src/synthetic/RUN_ALL_EXPERIMENTS.sh``. We support multiprocessing, e.g., every time window can be executed in a separate process. The number of used cores can be specified inside the scripts.


