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
import Models.model_physical as mphy
if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 10  # Bei RF mit festem random state nicht sinvoll

    dataSet = hdata.DataClass_ST_Plate_Notch

    dataSet.folder = '..\\..\\Data'#'..\\..\\DataSets\\Data'
    dataSet.training_data_paths =  ['DMC_S235JR_Plate_Normal_1.csv', 'DMC_S235JR_Plate_Normal_2.csv',
                                                    'DMC_S235JR_Plate_SF_1.csv', 'DMC_S235JR_Plate_Depth_1.csv',
                                                    'DMC_S235JR_Plate_SF_2.csv', 'DMC_S235JR_Plate_Depth_2.csv',
                                                    'DMC_S235JR_Plate_SF_3.csv', 'DMC_S235JR_Plate_Depth_3.csv']
    dataSet.validation_data_paths = ['DMC_S235JR_Notch_Normal_1.csv', 'DMC_S235JR_Notch_Normal_2.csv', 'DMC_S235JR_Notch_Normal_3.csv',
                                              'DMC_S235JR_Notch_Depth_1.csv', 'DMC_S235JR_Notch_Depth_2.csv', 'DMC_S235JR_Notch_Depth_3.csv']
    dataSet.testing_data_paths = [  'DMC_AL2007T4_Gear_Normal_3.csv','DMC_AL2007T4_Plate_Normal_3.csv',
                                    'DMC_S235JR_Gear_Normal_3.csv','DMC_S235JR_Plate_Normal_3.csv']

    dataSet.header = ["v_sp", "v_x", "v_y", "v_z", "a_x", "a_y", "a_z", "a_sp", "f_x", "f_y", "f_z",
                      "materialremoved_sim"]

    dataclass = copy.copy(dataSet)
    model = mphy.FrictionModel()

    # Run the experiment
    hexp.run_experiment([dataclass], models=[model],
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['prediction_overview', 'model_heatmap'], experiment_name='SimpleFriction')