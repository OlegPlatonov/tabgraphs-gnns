# GNNs for TabGraphs

This directory contains the code for running various GNNs (and the graph-agnostic ResNet baseline) on TabGraphs datasets.

To use this code, you first need to download the TabGraphs datasets, uncompress them,
and put them in the `data` directory.
Then, you need to install the necessary dependencies from the `environment.yml` file.

To condust an experiment, run the `train.py` file.
See the `get_args` function in the beginning of the file for the possible command line arguments.

To reproduce our experiments, see the `scripts/gnn_experiments.sh` and `scripts/gnn_plr_experiments.sh` files.
Each line in these files corresponds to one experiment.
You probably do not want to run them all, since it takes a lot of time, and we have already done that for you.
But you can consult these files for examples of how to use our code and to see which hyperparameters we have tried.

To see the aggregated results of our experiments in neat tables, check the `notebooks/results.py` notebook.
