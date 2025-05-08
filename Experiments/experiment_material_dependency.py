from torch import nn

import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hopt
import Models.model_neural_net as mnn

NUMBEROFTRIALS = 250
NUMBEROFEPOCHS = 800
NUMBEROFMODELS = 20

window_size = 10
past_values = 2
future_values  = 2

# load Hyperparameter
folder_path = '../Models/Hyperparameter/NeuralNet_curr_x'
model_params = hopt.GetOptimalParameter(folder_path, plot=False)
print(model_params)

data_params = []
folder_path = '../DataSets/DataFiltered'

training_base_names = ['AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF', 'AL_2007_T4_Plate_SF']
validation_base_names = ['AL_2007_T4_Gear_Normal', 'AL_2007_T4_Plate_Depth']
test_base_names = ['AL_2007_T4_Plate_Normal']
data_params.append(hdata.create_DataClasses_from_base_names(folder_path, training_base_names, validation_base_names, test_base_names, 'only - Aluminium'))

training_base_names = ['S235JR_Gear_Depth', 'S235JR_Gear_SF', 'S235JR_Plate_SF']
validation_base_names = ['S235JR_Gear_Normal', 'S235JR_Plate_Depth']
test_base_names = ['S235JR_Plate_Normal']
data_params.append(hdata.create_DataClasses_from_base_names(folder_path, training_base_names, validation_base_names, test_base_names, 'only - Steel'))

# Create Dataset - known
training_base_names = ['AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF', 'AL_2007_T4_Plate_SF', 'S235JR_Gear_Depth', 'S235JR_Gear_SF', 'S235JR_Plate_SF']
validation_base_names = ['AL_2007_T4_Gear_Normal', 'AL_2007_T4_Plate_Depth', 'S235JR_Gear_Normal', 'S235JR_Plate_Depth']
test_base_names = ['AL_2007_T4_Plate_Normal']
data_params.append(hdata.create_DataClasses_from_base_names(folder_path, training_base_names, validation_base_names, test_base_names, 'material known - Aluminium'))

training_base_names = ['AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF', 'AL_2007_T4_Plate_SF', 'S235JR_Gear_Depth', 'S235JR_Gear_SF', 'S235JR_Plate_SF']
validation_base_names = ['AL_2007_T4_Gear_Normal', 'AL_2007_T4_Plate_Depth', 'S235JR_Gear_Normal', 'S235JR_Plate_Depth']
test_base_names = ['S235JR_Plate_Normal']
data_params.append(hdata.create_DataClasses_from_base_names(folder_path, training_base_names, validation_base_names, test_base_names, 'material known - Steel'))

# Create Dataset - unknown
training_base_names = ['S235JR_Gear_Depth', 'S235JR_Gear_SF', 'S235JR_Plate_SF']
validation_base_names = ['S235JR_Gear_Normal', 'S235JR_Plate_Depth']
test_base_names = ['AL_2007_T4_Plate_Normal']
data_params.append(hdata.create_DataClasses_from_base_names(folder_path, training_base_names, validation_base_names, test_base_names, 'material unknown - Aluminum'))

training_base_names = ['AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF', 'AL_2007_T4_Plate_SF']
validation_base_names = ['AL_2007_T4_Gear_Normal', 'AL_2007_T4_Plate_Depth']
test_base_names = ['S235JR_Plate_Normal']
data_params.append(hdata.create_DataClasses_from_base_names(folder_path, training_base_names, validation_base_names, test_base_names, 'material unknown - Steel'))

results = []
names = []
for data_param in data_params:
    names.append(data_param.name)
    # load and prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = hdata.load_data(data_param, past_values=past_values,
                                                                     future_values=future_values, window_size=window_size)
    axis = 'curr_x'
    input_size = X_train.shape[1]
    output_size = y_train[axis].T.shape[0] if len(y_train[axis].shape) > 1 else 1  # handle single output case
    # Initialize the model
    model = mnn.Net(input_size=input_size, output_size=output_size,
                    n_neurons=model_params['n_neurons'], n_layers=model_params['n_layers'], activation=nn.ReLU)

    predictions = []
    losses = []
    for i in range(0, NUMBEROFMODELS):
        # Train the model
        val_error = model.train_model(
            X_train, y_train[axis], X_val, y_val[axis],
            learning_rate=model_params['learning_rate'], n_epochs=NUMBEROFEPOCHS, patience=5
        )
        loss, predictions = model.test_model(X_test, y_test[axis])
        losses.append(loss)
    results.append(losses)
    hplot.plot_prediction_vs_true(data_param.name + ' ' + axis, predictions.T, y_test[axis])

hplot.plot_bar_std(names, results, 'DataSet', 'mse', 'Influence of material known')