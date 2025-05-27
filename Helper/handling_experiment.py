import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import Helper.handling_data as hdata
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf

def run_experiment(dataSets, use_nn_reference, use_rf_reference, models, NUMBEROFEPOCHS=800, NUMBEROFMODELS=10, window_size=10, past_values=2, future_values=2, batched_data=False, n_drop_values=20, patience=5):
    if type(dataSets) is not list:
        dataSets = [dataSets]
    if type(models) is not list:
        models = [models]

    # Create directory for results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_dir = os.path.join("Results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Define the meta information structure
    meta_information = {
        "DataSets": [],
        "Models": [],
        "Data_Preprocessing": {
            "window_size": window_size,
            "past_values": past_values,
            "future_values": future_values,
            "batched_data": batched_data,
            "n_drop_values": n_drop_values,
            "NUMBEROFEPOCHS": NUMBEROFEPOCHS,
        }
    }

    # Define reference models
    reference_models = []
    if use_nn_reference:
        model_nn = mnn.get_reference()
        reference_models.append(model_nn)
    if use_rf_reference:
        model_rf = mrf.get_reference()
        reference_models.append(model_rf)

    # Add reference models to meta information
    for model in reference_models:
        model_info = {
            model.name+"_Ensemble": {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                **model.get_documentation()
            },
        }
        meta_information["Models"].append(model_info)

    # Add models to meta information
    for model in models:
        model_info = {
            model.name+"_Ensemble": {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                **model.get_documentation()
            },
        }
        meta_information["Models"].append(model_info)

    # Add datasets to meta information
    for data_params in dataSets:
        data_info = {
            **data_params.get_documentation()
        }
        meta_information["DataSets"].append(data_info)

    # Save the meta information to a JSON file
    documentation = meta_information

    """ Check if data_params.testing_data_paths is the same """

    """ Prediction """
    results = []
    for i, data in enumerate(dataSets):
        print(f"\n===== Verarbeitung: {data.name} =====")
        # Daten laden
        X_train, X_val, X_test, y_train, y_val, y_test = data.load_data(
            past_values, future_values, window_size, keep_separate=batched_data)
        if isinstance(X_test, list):
            df_list_results = [pd.DataFrame() for x in range(len(X_test))]
            header_list = [[] for x in range(len(X_test))]
        else:
            df_list_results = [pd.DataFrame()]
            header_list = []

        # Train and test reference models
        for model in reference_models:
            # Modellvergleich auf neuen Daten
            nn_preds = [[] for _ in range(len(X_test))] if isinstance(X_test, list) else []

            for _ in range(NUMBEROFMODELS):
                model.train_model(X_train, y_train, X_val, y_val, NUMBEROFEPOCHS, patience=patience)
                if isinstance(X_test, list):
                    for i, (x, y) in enumerate(zip(X_test, y_test)):
                        mse, pred_nn = model.test_model(x, y)
                        nn_preds[i].append(pred_nn.flatten())
                        print(f"{model.name}: Test MSE: {mse}")
                else:
                    mse, pred_nn = model.test_model(X_test, y_test)
                    nn_preds.append(pred_nn.flatten())
                    print(f"{model.name}: Test MSE: {mse}")
            # Fehlerberechnung
            for i, path in enumerate(data.testing_data_paths):
                name = model.name + "_" + path.replace('.csv', '')

                mse_nn, std_nn = hdata.calculate_mae_and_std(nn_preds[i], y_test[i].values if isinstance(y_test[i],
                                                                                                   pd.DataFrame) else y_test[i],
                                                             n_drop_values)
                # Ergebnisse speichern
                df_list_results[i][name] = np.mean(nn_preds[i], axis=0)
                results.extend([
                    [data.name + "_" + path.replace('.csv', ''), model.name, mse_nn, std_nn],
                ])
                header_list[i].append(name)

        if len(reference_models) > 0:
            criterion = reference_models[0].criterion
        else:
            criterion = None
        # Train and test models
        for model in models:
            # Modellvergleich auf neuen Daten
            nn_preds = [[] for _ in range(len(X_test))] if isinstance(X_test, list) else []

            for _ in range(NUMBEROFMODELS):
                model.train_model(X_train, y_train, X_val, y_val, n_epochs = NUMBEROFEPOCHS, patience=patience)
                if isinstance(X_test, list):
                    for i, (x, y) in enumerate(zip(X_test, y_test)):
                        mse, pred_nn = model.test_model(x, y)
                        print(f"{model.name}: Test RMSE: {mse}")
                        nn_preds[i].append(pred_nn.flatten())
                else:
                    mse, pred_nn = model.test_model(X_test, y_test)
                    print(f"{model.name}: Test RMSE: {mse}")
                    nn_preds.append(pred_nn.flatten())

            # Fehlerberechnung
            for i, path in enumerate(data.testing_data_paths):
                name = model.name + "_" + path.replace('.csv', '')

                mse_nn, std_nn = hdata.calculate_mae_and_std(nn_preds[i], y_test[i].values if isinstance(y_test[i],
                                                                                                   pd.DataFrame) else y_test[i],
                                                             n_drop_values)
                # TODO: DataSet und DataPath getrennt speichern. Bar Plot für jedes DataPath erstellen
                # Ergebnisse speichern
                df_list_results[i][name] = np.mean(nn_preds[i], axis=0)
                results.extend([
                    [data.name, model.name, mse_nn, std_nn],
                ]) # + "_" + path.replace('.csv', '')
                header_list[i].append(name)

        """ Dokumentation """
        df_list_data, names = hdata.get_test_data_as_pd(data, past_values=past_values, future_values=future_values,
                                               window_size=window_size)
        data_dir = os.path.join(results_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        file_paths = [os.path.join(data_dir, f'{data.name}_{name}.csv') for name in names]
        hdata.save_data(df_list_data, file_paths)

        if n_drop_values == 0:
            n_drop_values = 1 # prevent [-0]
        # Save Results in csv
        for y, df, header, path in zip(y_test, df_list_results, header_list, data.testing_data_paths):
            name = path.replace('.csv', '')
            df['y_ground_truth'] = y["curr_x"] if isinstance(y, pd.DataFrame) else y

            # Plot speichern
            plots_dir = os.path.join(results_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            plt.figure(figsize=(12, 6))
            plt.plot(df['y_ground_truth'].iloc[:-n_drop_values], label='Ground Truth', color='black', linewidth=2)

            for col in header:
                label = col.replace('_'+name,'')
                plt.plot(df[col].iloc[:-n_drop_values], linestyle='--', label=f'{label}', alpha=0.6)

            plt.title(f'{data.name}_{name}: Modellvergleich')
            plt.xlabel('Zeit')
            plt.ylabel('Strom in A')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, f'{data.name}_{name}_comparison.png')
            plt.savefig(plot_path)
            plt.close()

        hdata.add_pd_to_csv(df_list_data, file_paths, header_list)

    # Export als CSV

    df = pd.DataFrame(results, columns=["DataSet", "Model", "MSE", "StdDev"])
    models = df['Model'].unique()
    datasets = sorted(df['DataSet'].unique())
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(10, 5))
        df_dataset = df[df['DataSet'] == dataset]

        num_models = len(models)
        bar_width = 0.15
        x = np.arange(num_models)

        for i, model in enumerate(models):
            df_model = df_dataset[df_dataset['Model'] == model]
            y = df_model["MSE"].values
            yerr = df_model["StdDev"].values

            x_pos = x[i]
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

        ax.set_title(f"Vergleich Modelle für {dataset}")
        ax.set_xlabel("Modelle")
        ax.set_ylabel("MAE")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f'model_comparison_{dataset}.png')
        plt.savefig(plot_path)
        plt.close()

    """ 
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

    ax.set_title(f"Vergleich Modelle")
    ax.set_xlabel("DataSets")
    ax.set_ylabel("MAE")
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'model_comparison.png')
    plt.savefig(plot_path)
    plt.close()"""

    # Prozentuale Verbesserung berechnen und speichern
    improvement_results = []
    for dataset in datasets:
        for model in models[1:]:
            # MSE-Werte für beide Modelle abrufen
            mse_nn = df[(df['DataSet'] == dataset) & (df['Model'] == models[0])]['MSE'].values[0]
            mse_rnn = df[(df['DataSet'] == dataset) & (df['Model'] == model)]['MSE'].values[0]

            # Prozentuale Verbesserung berechnen
            improvement = (mse_nn - mse_rnn) / mse_nn * 100
            improvement_results.append((dataset, model, improvement))

    print("\n Modellvergleichsergebnisse:")
    with open(os.path.join(results_dir, 'Results.txt'), 'w') as f:
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

    documentation["Results"] = {
        "Model_Comparison": results,
        "Improvement": improvement_results
    }

    with open(os.path.join(results_dir, 'documentation.json'), 'w') as json_file:
        json.dump(documentation, json_file, indent=4)

    print("\n Ergebnisse wurden in 'documentation.json' und 'Results.txt' gespeichert.")