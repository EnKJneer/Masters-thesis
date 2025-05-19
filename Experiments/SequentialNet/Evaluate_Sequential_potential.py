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
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf
import matplotlib.pyplot as plt
import seaborn as sns

""" Functions """

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 10

    window_size = 10
    past_values = 2
    future_values = 2

    dataSets = hdata.dataSets_list_Plate

    # torch.cuda.is_available()
    # Load data with gear method --> Needed to get input size
    X_train, X_val, X_test, y_train, y_val, y_test = hdata.load_data(dataSets[0],
                                                                                       past_values=past_values,
                                                                                       future_values=future_values,
                                                                                       window_size=window_size)

    input_size = X_train.shape[1]
    model_nn = mnn.get_reference_net(input_size)
    model_rnn = mnn.SequentialNet(input_size, 1, input_size, 1)
    """Save Meta information"""
    # Define the meta information structure
    meta_information = {
        "DataSets": [],
        "Models": {
            "Neural_Net_Ensemble": {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                "hyperparameters": {
                    "learning_rate": model_nn.learning_rate,
                    "n_hidden_size": model_nn.n_hidden_size,
                    "n_hidden_layers": model_nn.n_hidden_layers,
                }
            },
            "Sequential_Neural_Net_Ensemble": {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                "hyperparameters": {
                    "learning_rate": model_rnn.learning_rate,
                    "n_hidden_size": model_rnn.n_hidden_size,
                    "n_hidden_layers": model_rnn.n_hidden_layers,
                }
            }
        },
        "Data_Preprocessing": {
            "window_size": window_size,
            "past_values": past_values,
            "future_values": future_values
        }
    }

    for data_params in dataSets:
        data_info = {
            "name": data_params.name,
            "folder": data_params.folder,
            "training_validation_data": data_params.training_validation_datas,
            "testing_data_paths": data_params.testing_data_paths,
            "target_channels": data_params.target_channels,
            "percentage_used": data_params.percentage_used
        }
        meta_information["DataSets"].append(data_info)


    # Save the meta information to a JSON file
    with open('Data/info.json', 'w') as json_file:
        json.dump(meta_information, json_file, indent=4)

    """ Prediction """
    results = []
    for i,data in enumerate(dataSets):
        df, name = hdata.get_test_data_as_pd(data, past_values=past_values, future_values=future_values,
                                               window_size=window_size)
        file_path = f'Data/{data.name}.csv'
        hdata.save_data(df, file_path)
        df, name = hdata.get_test_data_as_pd(data, past_values=past_values, future_values=future_values,
                                               window_size=window_size)
        file_path = f'Data/{data.name}.csv'
        hdata.save_data(df, file_path)

        print(f"\n===== Verarbeitung: {data.name} =====")
        # Daten laden
        X_train, X_val, X_test, y_train, y_val, y_test = hdata.load_data(
            data, past_values, future_values, window_size, keep_separate=False) # is faster
        X_train_batched, X_val_batched, X_test_batched, y_train_batched, y_val_batched, y_test_batched = hdata.load_data(
            data, past_values, future_values, window_size, keep_separate=True)
        # Modellvergleich auf neuen Daten
        nn_preds, rnn_preds = [], []
        for _ in range(NUMBEROFMODELS):
            model_rnn = mnn.SequentialNet(input_size, 1, input_size, 1) # reset model
            model_rnn.train_model(X_train_batched, y_train_batched, X_val_batched, y_val_batched, NUMBEROFEPOCHS, patience=5)
            _, pred_rnn = model_rnn.test_model(X_test, y_test)
            rnn_preds.append(pred_rnn.flatten())

            model_nn = mnn.get_reference_net(input_size) # reset model
            model_nn.train_model(X_train, y_train, X_val, y_val, NUMBEROFEPOCHS, patience=5)
            _, pred_nn = model_nn.test_model(X_test, y_test)
            nn_preds.append(pred_nn.flatten())

        # Fehlerberechnung
        n_drop_values = 10
        mse_nn, std_nn = hdata.calculate_mse_and_std(nn_preds, y_test.values, n_drop_values)
        mse_rnn, std_rnn = hdata.calculate_mse_and_std(rnn_preds, y_test.values, n_drop_values)

        # Ergebnisse speichern
        results.extend([
            [data.name, model_nn.name, mse_nn, std_nn],
            [data.name, model_rnn.name, mse_rnn, std_rnn],
        ])
        # Save Results in csv
        header = [model_nn.name, model_rnn.name]
        data_list = [nn_preds, rnn_preds]
        df = pd.DataFrame()
        df['y_ground_truth'] = y_test["curr_x"]
        for i, col in enumerate(header):
            df[col] = np.mean(data_list[i], axis=0)
        hdata.add_pd_to_csv(file_path, df, header)

        plt.figure(figsize=(12, 6))
        plt.plot(df['y_ground_truth'], label='Ground Truth', color='black', linewidth=2)

        for col in header:
            plt.plot(df[col], linestyle='--', label=f'{col}', alpha=0.6)

        plt.title(f'{data.name}: Modellvergleich NN vs. RNN')
        plt.xlabel('Zeit')
        plt.ylabel('Strom in A')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Export als CSV
    # Ordner anlegen
    os.makedirs("Data", exist_ok=True)

    df = pd.DataFrame(results, columns=["DataSet", "Model", "MSE", "StdDev"])
    models = df['Model'].unique()
    n_models = len(models)
    datasets = sorted(df['DataSet'].unique())

    num_models = len(models)
    bar_width = 0.15
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        df_model = df[df['Model'] == model]
        df_model = df_model.set_index("DataSet").reindex(datasets).reset_index()

        y = df_model["MSE"].values
        yerr = df_model["StdDev"].values

        x_pos = x + i * bar_width
        bars = ax.bar(x_pos, y, width=bar_width, label=model, yerr=yerr, capsize=4)

        # Text über jedem Balken
        for k, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # x-Position: Mitte des Balkens
                height + 0.01 * max(y),  # y-Position: etwas über Balken
                f"{height:.2f}",  # Formatierter MSE-Wert
                ha='center', va='bottom', fontsize=8
            )

    ax.set_title(f"NN vs RNN")
    ax.set_xlabel("DataSets")
    ax.set_ylabel("MSE")
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Prozentuale Verbesserung berechnen und speichern
    improvement_results = []
    for dataset in datasets:
        # MSE-Werte für beide Modelle abrufen
        mse_nn = df[(df['DataSet'] == dataset) & (df['Model'] == models[0])]['MSE'].values[0]
        mse_rnn = df[(df['DataSet'] == dataset) & (df['Model'] == models[1])]['MSE'].values[0]

        # Prozentuale Verbesserung berechnen
        improvement = (mse_nn - mse_rnn) / mse_nn * 100
        improvement_results.append((dataset, models[1], improvement))

    print("\n Modellvergleichsergebnisse:")
    with open('Data/model_comparison_results.txt', 'w') as f:
        f.write("DataSet                 | Model          | MSE        | StdDev\n")
        f.write("-" * 75 + "\n")
        for row in results:
            print(f"{row[0]:<25} | {row[1]:<15} | MSE: {row[2]:.6f} | StdDev: {row[3]:.6f}")
            f.write(f"{row[0]:<25} | {row[1]:<15} | {row[2]:<6} | {row[3]:.6f}\n")

        f.write("\nProzentuale Verbesserung:\n")
        f.write("DataSet                 | Model           | Improvement(%) \n")
        print("DataSet                 | Model           | Improvement(%) \n")
        f.write("-" * 50 + "\n")
        for row in improvement_results:
            print(f"{row[0]:<25} | {row[1]:<15} | {row[2]:.2f}\n")
            f.write(f"{row[0]:<25} | {row[1]:<15} | {row[2]:.2f}\n")


        x_pos = x + i * bar_width
        bars = ax.bar(x_pos, y, width=bar_width, label=model, yerr=yerr, capsize=4)

        # Text über jedem Balken
        for k, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # x-Position: Mitte des Balkens
                height + 0.01 * max(y),  # y-Position: etwas über Balken
                f"{height:.2f}",  # Formatierter MSE-Wert
                ha='center', va='bottom', fontsize=8
            )

    ax.set_title(f"NN vs RNN")
    ax.set_xlabel("DataSets")
    ax.set_ylabel("MSE")
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Prozentuale Verbesserung berechnen und speichern
    improvement_results = []
    for dataset in datasets:
        # MSE-Werte für beide Modelle abrufen
        mse_nn = df[(df['DataSet'] == dataset) & (df['Model'] == models[0])]['MSE'].values[0]
        mse_rnn = df[(df['DataSet'] == dataset) & (df['Model'] == models[1])]['MSE'].values[0]

        # Prozentuale Verbesserung berechnen
        improvement = (mse_nn - mse_rnn) / mse_nn * 100
        improvement_results.append((dataset, models[1], improvement))

    print("\n Modellvergleichsergebnisse:")
    with open('Data/model_comparison_results.txt', 'w') as f:
        f.write("DataSet                 | Model          | MSE        | StdDev\n")
        f.write("-" * 75 + "\n")
        for row in results:
            print(f"{row[0]:<25} | {row[1]:<15} | MSE: {row[2]:.6f} | StdDev: {row[3]:.6f}")
            f.write(f"{row[0]:<25} | {row[1]:<15} | {row[2]:<6} | {row[3]:.6f}\n")

        f.write("\nProzentuale Verbesserung:\n")
        f.write("DataSet                 | Model           | Improvement(%) \n")
        print("DataSet                 | Model           | Improvement(%) \n")
        f.write("-" * 50 + "\n")
        for row in improvement_results:
            print(f"{row[0]:<25} | {row[1]:<15} | {row[2]:.2f}\n")
            f.write(f"{row[0]:<25} | {row[1]:<15} | {row[2]:.2f}\n")
