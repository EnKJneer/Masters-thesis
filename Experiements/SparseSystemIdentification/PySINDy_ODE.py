import copy
import os
import numpy as np
import pandas as pd
import pysindy as ps
from matplotlib import pyplot as plt
from scipy.signal import lfilter, firwin, butter, filtfilt
from sklearn.metrics import mean_absolute_error
# Assuming Helper.handling_data is a custom module you have
import Helper.handling_data as hdata

# Define the Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the Signum function
def signum(x):
    return np.sign(x)

# Function to filter the data
def apply_fir_filter(data, fir_coeff):
    return lfilter(fir_coeff, 1.0, data)

# Butterworth filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_butterworth_filter(data, cutoff, fs):
    b, a = butter_lowpass(cutoff, fs, order=5)
    y = filtfilt(b, a, data)
    return y

# Define a function that uses z_x as an inhomogeneous term
def zx_function(x, z_x):
    return z_x  # Use z_x directly as an inhomogeneous term

if __name__ == "__main__":
    path = '../../DataSets/Data'
    file = 'S235JR_Plate_Normal_2.csv'
    file_path = os.path.join(path, file)
    df = pd.read_csv(file_path)

    # Apply Butterworth filter to each column
    columns_to_filter = ["curr_x", "pos_x", "pos_y", "v_x", "v_y", "f_x_sim", "f_y_sim"]
    #columns_to_filter = ["curr_x", "pos_x", "v_x", "f_x_sim"]
    fs = 50
    cutoff = 5
    df_butterworth = df.copy()
    for column in columns_to_filter:
        df_butterworth[column] = apply_butterworth_filter(df[column], cutoff, fs)

    df = df_butterworth.iloc[300:700]
    df['z_x'] = np.sign(df['v_x'])

    # Select the relevant columns and convert them to a numpy array
    data = df[columns_to_filter].values

    print("Shape of data:", data.shape)
    plt.plot(data[:, 0])
    plt.show()

    # Sampling frequency
    fs = 50  # Sampling frequency in Hz
    N = data.shape[0]
    # Total time duration
    total_time = (N - 1) / fs
    # Create time vector
    time = np.linspace(0, total_time, N).squeeze()
    print("Length of time vector:", len(time))

    # Create custom libraries for the sigmoid and signum functions
    sigmoid_library = ps.CustomLibrary(library_functions=[sigmoid])
    signum_library = ps.CustomLibrary(library_functions=[signum])

    # Create a custom library for z_x
    zx_library = ps.CustomLibrary(library_functions=[lambda x: zx_function(x, df["z_x"].values)])

    # Combine the custom libraries with a polynomial library
    basis_functions = ps.PolynomialLibrary(degree=1) #+ signum_library

    # Create a SINDy model with the specified basis functions
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=0.1),
        feature_library=basis_functions,
        differentiation_method=ps.SmoothedFiniteDifference()
    )

    # Fit the model to your data
    model.fit(data, t= time)

    # Print the identified equations
    model.print()

    # Ensure the initial conditions are a NumPy array
    initial_condition = data[0, :]

    # Generate predictions
    predicted_data = model.simulate(initial_condition, time)

    # Calculate the MAE for curr_x
    mae = mean_absolute_error(data[:, 0], predicted_data[:, 0])
    print("MAE for curr_x:", mae)

    # Plot curr_x and the prediction for curr_x
    plt.figure(figsize=(12, 6))
    plt.plot(time, data[:, 0], label='Actual curr_x')
    plt.plot(time, predicted_data[:, 0], label='Predicted curr_x', linestyle='--')
    plt.title('Comparison of Actual and Predicted curr_x')
    plt.xlabel('Time')
    plt.ylabel('curr_x')
    plt.legend()
    plt.grid(True)
    plt.show()
