import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn

# Beispiel für diskrete Laplace-Transformation
def apply_fft(df):
    """
    Wendet die Fourier-Transformation auf jede Spalte eines DataFrames an
    und gibt Betrag und Phase zurück.
    """
    fft_result = df.apply(lambda column: np.fft.fft(column))
    magnitude = fft_result.apply(lambda column: np.abs(column))
    phase = fft_result.apply(lambda column: np.angle(column))
    return pd.DataFrame(np.concatenate([magnitude, phase], axis=1))

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 500
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 0
    future_values = 0

    dataClass = hdata.Combined_Plate_TrainVal
    dataClass.target_channels = ['curr_x']
    dataClass.past_values = past_values
    dataClass.future_values = future_values
    dataClass.window_size = window_size

    dataClasses = [dataClass] #, hdata.Combined_Plate_TrainVal_CONTDEV

    #model_simple = mphys.NaiveModelSimple()
    model = mnn.get_reference()
    model.output_size = 1
    models = [model]

    # Ihre Daten laden
    X_train, X_val, X_test, y_train, y_val, y_test = dataClass.load_data()

    # Oder für einzelne Datensätze:
    for i, (X, y) in enumerate(zip(X_test[:3], y_test[:3])):  # Nur erste 3
        print(f"\n=== Datensatz {i + 1} ===")
        X_df = apply_fft(X)
        #y_df = apply_fft(y)
        model.train_model(X_df, y, X_df, y)
        loss, y_pred = model.test_model(X_df, y)
        plt.plot(y_pred, label = 'y_pred')
        plt.plot(y.values, label = 'y_gt')
        plt.title(dataClass.testing_data_paths[i])
        plt.legend()
        plt.show()
        print(f"Loss: {loss}")

