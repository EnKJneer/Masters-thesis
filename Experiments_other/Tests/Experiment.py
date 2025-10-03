import copy
import datetime
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_experiment as hexp
import Models.model_random_forest as mrf
import Models.model_neural_net as mnn


if __name__ == '__main__':

    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 1000
    NUMBEROFMODELS = 1

    window_size = 1
    past_values = 0
    future_values = 0

    dataSet = hdata.DataClass_ST_Plate_Notch
    #dataSet.header = ["pos_sp", "pos_x", "pos_y", "pos_z", "v_sp", "v_x", "v_y", "v_z", "a_x", "a_y", "a_z", "a_sp", "f_x_sim", "f_y_sim", "f_z_sim", "f_sp_sim", "materialremoved_sim"]
    #dataSet.folder = '..\\..\\DataSets_CMX_Plate_Notch_Gear_Reference/Data'
    #dataSet.folder = '..\\..\\Data'
    #dataSet.testing_data_paths = [  'DMC60H_AL2007T4_Gear_SF_3.csv','DMC60H_AL2007T4_Plate_SF_3.csv',
    #                                'DMC60H_S235JR_Gear_SF_3.csv','DMC60H_S235JR_Plate_Normal_3.csv']
    #dataSet.testing_data_paths =[  'CMX600V_AL2007T4_Validierung_Normal_2.csv','DMC60H_AL2007T4_Plate_Normal_3.csv',
    #                               'CMX600V_S235JR_Validierung_Normal_2.csv','DMC60H_S235JR_Plate_Normal_3.csv']
    dataSet.header =["v_sp", "v_x", "a_x", "f_x", "materialremoved_sim"]

    dataclass2 = copy.copy(dataSet)
    #dataclass2.name = 'mit z'
    #dataclass2.add_sign_hold = True
    #dataclass2.target_channels = ['curr_y']
    # dataclass2 = hdata.Combined_Plate_TrainVal
    dataClasses = [dataclass2]
    for dataclass in dataClasses:
        dataclass.add_padding = True

    model_rf = mrf.RandomForestModel(n_estimators= 167, min_samples_split= 6,
                    min_samples_leaf= 2)

    model_rnn = mnn.RNN(learning_rate= 0.1, n_hidden_size= 71, n_hidden_layers= 1,
                    activation= 'ELU', optimizer_type= 'quasi_newton')

    models = [model_rnn]

    # Run the experiment
    hexp.run_experiment(dataClasses, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['heatmap', 'prediction_overview'], experiment_name=dataSet.name)
