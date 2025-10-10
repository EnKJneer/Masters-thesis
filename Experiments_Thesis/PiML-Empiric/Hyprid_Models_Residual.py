import numpy as np
import pandas as pd
import torch
from numpy.f2py.auxfuncs import throw_error
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_base as mb
import Models.model_neural_net as mnn
import Models.model_physical as mphys
import Models.model_random_forest as mrf
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 100
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 0
    future_values = 0

    dataSets = [hdata.DataClass_ST_Plate_Notch]

    model_phys = mphys.EmpiricLinearModel()

    model_rf = mrf.RandomForestModel(n_estimators= 100, max_depth=100, min_samples_split= 2,
                    min_samples_leaf= 4)

    model_rnn = mnn.RNN(learning_rate= 0.1, n_hidden_size= 71, n_hidden_layers= 1,
                    activation= 'ELU', optimizer_type= 'quasi_newton')

    model_pirnn = mnn.PiRNN(learning_rate= 0.1, n_hidden_size= 71, n_hidden_layers= 1,
                      activation= 'ELU', optimizer_type= 'quasi_newton')

    model_hybrid_rnn = mnn.HybridModelResidual(physical_model=model_phys, ml_model=model_pirnn, name = 'Hybrid_RNN')
    model_hybrid_rf = mnn.HybridModelResidual(physical_model=model_phys, ml_model=model_rf, name = 'Hybrid_Random_Forest')

    use_rf = False
    if use_rf:
        models = [model_phys, model_rf, model_hybrid_rf] # , model_nn, model_hybrid_nn
        postfix = 'RF'
    else:
        models = [model_phys, model_rnn, model_hybrid_rnn]
        postfix = 'RNN'

    # Run the experiment
    hexp.run_experiment(dataSets, models=models, NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        experiment_name=model_phys.name+'_Residual_'+postfix)