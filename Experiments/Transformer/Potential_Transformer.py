import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 10

    window_size = 10
    past_values = 2
    future_values = 2

    dataSets = [hdata.Combined_Plate]

    input_size = None
    model_attention = mnn.NetAttention(input_size=input_size,output_size=1)
    model_transformer = mnn.NetTransformer(input_size=input_size, output_size=1)
    models = [model_attention, model_transformer]

    # Run the experiment
    hexp.run_experiment(dataSets, use_nn_reference=True, use_rf_reference=False, models=models, NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS, window_size=window_size, past_values=past_values, future_values=future_values)