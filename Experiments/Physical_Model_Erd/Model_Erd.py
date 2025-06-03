import numpy as np
import pandas as pd
import torch
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn
import Models.model_physical as mphys

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 2

    window_size = 1
    past_values = 0
    future_values = 0

    #Combined_Gear,Combined_KL
    dataClass_1 = hdata.Combined_Plate_TrainVal
    dataClass_1.window_size = window_size
    dataClass_1.past_values = past_values
    dataClass_1.future_values = future_values

    dataSets_list = [dataClass_1]

    dataSets = [hdata.Combined_Plate]

    model_erd = mphys.PhysicalModelErd(learning_rate=1)
    model_erd_one_axis = mphys.PhysicalModelErdSingleAxis(0.01, 0.01, 0.01, 0, 0, learning_rate=1)
    models = [model_erd, model_erd_one_axis]

    # Run the experiment
    experiment_results = hexp.run_experiment(dataSets_list, True, False, [model_erd, model_erd_one_axis],
                        NUMBEROFEPOCHS, NUMBEROFMODELS, past_values, future_values,n_drop_values=25,
                        plot_types=['heatmap', 'prediction_overview', 'geometry_mae'])

    # Zugriff auf Ergebnisse
    print(f"Experiment gespeichert in: {experiment_results['results_dir']}")
    print(f"Anzahl Plots erstellt: {sum(len(paths) for paths in experiment_results['plot_paths'].values())}")