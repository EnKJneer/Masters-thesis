import glob
import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

if __name__ == "__main__":
    """ Constants """
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 2
    future_values = 2


    dataSets_list = [hdata.Combined_Plate, hdata.Combined_Plate_Single]
    model_rf = mrf.get_reference()
    experiment_results = hexp.run_experiment(dataSets_list, True, False, [model_rf],
                        NUMBEROFEPOCHS, NUMBEROFMODELS, past_values, future_values,n_drop_values=20,
                        plot_types=['datapath', 'heatmap', 'prediction_overview'])

    # Zugriff auf Ergebnisse
    print(f"Experiment gespeichert in: {experiment_results['results_dir']}")
    print(f"Anzahl Plots erstellt: {sum(len(paths) for paths in experiment_results['plot_paths'].values())}")