# Reliable Predictive Inference in Time-Series Settings

An important factor to guarantee a responsible use of data-driven systems is that we should be able to communicate their uncertainty to decision makers. This can be accomplished by constructing prediction sets, which provide an intuitive measure of the limits of predictive performance.

This package contains a Python implementation of Rolling Risk Control (Rolling RC) [1] methodology for constructing distribution-free prediction sets that provably control a general risk in an online setting. 

# Risk Control for Online Learning Models [1]

Rolling RC is a method that reliably reports the uncertainty of a target variable response in an online time-series setting and provably attains the user-specified risk level over long-time intervals.

[1] Shai Feldman, Liran Ringel Stephen Bates, Yaniv Romano, [“Risk Control for Online Learning Models”]() 2022.

## Getting Started

This package is self-contained and implemented in python.

Part of the code is a taken from the [oqr](https://github.com/Shai128/oqr) and [mqr](https://github.com/Shai128/mqr) packages. 


### Prerequisites

* python
* numpy
* scipy
* scikit-learn
* pytorch
* pandas

### Installing

The development version is available here on github:
```bash
git clone https://github.com/shai128/rrc.git
```

## Usage


### Rolling RC

Demonstration of the proposed method in an online depth prediction task is provided in [display_depth_results.ipynb](display_depth_results.ipynb)

Analysis of the proposed method on tabular dataset and comparison to competitive methods and can be found in [single_response_plots.ipynb](single_response_plots.ipynb).

## Reproducible Research

The code available under /reproducible_experiments/ in the repository replicates the experimental results in [1].

### Publicly Available Datasets

* [KITTI](https://www.cvlibs.net/datasets/kitti/): The KITTI Vision Benchmark Suite.

* [Power](https://archive.ics.uci.edu/ml/datasets/Power+consumption+of+Tetouan+city): Power consumption of Tetouan city Data Set.

* [Energy](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction): Appliances energy prediction Data Set.

* [Traffic](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume): Metro Interstate Traffic Volume Data Set.

* [Wind](https://www.kaggle.com/datasets/l3llff/wind-power): Wind Power in Germany.

* [Power](https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries/blob/main/data_prices/Prices_2016_2019_extract.csv): French electricity prices [2].


[2] Margaux Zaffran, Aymeric Dieuleveut, Olivier Féron, Yannig Goude, Julie Josse, [“Adaptive Conformal Predictions for Time Series.”](https://arxiv.org/abs/2202.07282) 2022.
