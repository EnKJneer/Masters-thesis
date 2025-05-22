import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import Helper.handling_data as hdata
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf

def run_experiment(dataSets, use_nn_reference, use_rf_reference, models, NUMBEROFEPOCHS=800, NUMBEROFMODELS=10, window_size=10, past_values=2, future_values=2, batched_data=False, n_drop_values=10):
    if type(dataSets) is not list:
        dataSets = [dataSets]

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
        }
    }

    # Define reference models
    reference_models = []
    if use_nn_reference:
        model_nn = mnn.get_reference_net()
        reference_models.append(model_nn)
    if use_rf_reference:
        model_rf = mrf.get_reference_model()
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
            "name": data_params.name,
            "folder": data_params.folder,
            "training_validation_data": data_params.training_validation_datas,
            "testing_data_paths": data_params.testing_data_paths,
            "target_channels": data_params.target_channels,
            "percentage_used": data_params.percentage_used
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
        X_train, X_val, X_test, y_train, y_val, y_test = hdata.load_data(
            data, past_values, future_values, window_size, keep_separate=batched_data)
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
                model.train_model(X_train, y_train, X_val, y_val, NUMBEROFEPOCHS, patience=5)
                if isinstance(X_test, list):
                    for i, (x, y) in enumerate(zip(X_test, y_test)):
                        _, pred_nn = model.test_model(x, y)
                        nn_preds[i].append(pred_nn.flatten())
                else:
                    _, pred_nn = model.test_model(X_test, y_test)
                    nn_preds.append(pred_nn.flatten())

            # Fehlerberechnung
            for i, path in enumerate(data.testing_data_paths):
                name = model.name + "_" + path.replace('.csv', '')

                mse_nn, std_nn = hdata.calculate_mse_and_std(nn_preds[i], y_test[i].values if isinstance(y_test[i],
                                                                                                   pd.DataFrame) else y_test[i],
                                                             n_drop_values)
                # Ergebnisse speichern
                df_list_results[i][name] = np.mean(nn_preds[i], axis=0)
                results.extend([
                    [data.name + "_" + path.replace('.csv', ''), model.name, mse_nn, std_nn],
                ])
                header_list[i].append(name)

        # Train and test models
        for model in models:
            # Modellvergleich auf neuen Daten
            nn_preds = [[] for _ in range(len(X_test))] if isinstance(X_test, list) else []

            for _ in range(NUMBEROFMODELS):
                model.train_model(X_train, y_train, X_val, y_val, NUMBEROFEPOCHS, patience=5)
                if isinstance(X_test, list):
                    for i, (x, y) in enumerate(zip(X_test, y_test)):
                        _, pred_nn = model.test_model(x, y)
                        nn_preds[i].append(pred_nn.flatten())
                else:
                    _, pred_nn = model.test_model(X_test, y_test)
                    nn_preds.append(pred_nn.flatten())

            # Fehlerberechnung
            for i, path in enumerate(data.testing_data_paths):
                name = model.name + "_" + path.replace('.csv', '')

                mse_nn, std_nn = hdata.calculate_mse_and_std(nn_preds[i], y_test[i].values if isinstance(y_test[i],
                                                                                                   pd.DataFrame) else y_test[i],
                                                             n_drop_values)
                # Ergebnisse speichern
                df_list_results[i][name] = np.mean(nn_preds[i], axis=0)
                results.extend([
                    [data.name + "_" + path.replace('.csv', ''), model.name, mse_nn, std_nn],
                ])
                header_list[i].append(name)

        """ Dokumentation """
        df_list_data, names = hdata.get_test_data_as_pd(data, past_values=past_values, future_values=future_values,
                                               window_size=window_size)
        data_dir = os.path.join(results_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        file_paths = [os.path.join(data_dir, f'{data.name}_{name}.csv') for name in names]
        hdata.save_data(df_list_data, file_paths)

        # Save Results in csv
        for y, df, header, path in zip(y_test, df_list_results, header_list, data.testing_data_paths):
            name = path.replace('.csv', '')
            df['y_ground_truth'] = y["curr_x"] if isinstance(y, pd.DataFrame) else y

            # Plot speichern
            plots_dir = os.path.join(results_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            plt.figure(figsize=(12, 6))
            plt.plot(df['y_ground_truth'], label='Ground Truth', color='black', linewidth=2)

            for col in header:
                label = col.replace('_'+name,'')
                plt.plot(df[col], linestyle='--', label=f'{label}', alpha=0.6)

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

    ax.set_title(f"Vergleich Modelle")
    ax.set_xlabel("DataSets")
    ax.set_ylabel("MSE")
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'model_comparison.png')
    plt.savefig(plot_path)
    plt.close()

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


class Experiment:
    """
    A class to conduct machine learning experiments using various datasets and models.

    This class facilitates the execution of machine learning experiments by allowing the user
    to specify datasets, models, and various parameters for training and testing. It supports
    the use of reference models (neural network and random forest) and custom models. The
    results of the experiments, including performance metrics and visualizations, are saved
    in a structured directory.

    Attributes:
        dataSets (list): A list of datasets to be used in the experiment.
        use_nn_reference (bool): Flag to indicate whether to use a neural network reference model.
        use_rf_reference (bool): Flag to indicate whether to use a random forest reference model.
        models (list): A list of models to be used in the experiment.
        number_of_epochs (int): Number of epochs for training the models. Default is 800.
        ensemble_size (int): Size of the ensemble for each model. Default is 10.
        window_size (int): Size of the window for data preprocessing. Default is 10.
        past_values (int): Number of past values to consider for predictions. Default is 2.
        future_values (int): Number of future values to predict. Default is 2.
        batched_data (bool): Flag to indicate whether data should be batched. Default is False.
        n_drop_values (int): Number of values to drop during error calculation. Default is 10.
        data_loader (function): Function to load and preprocess the data. Default is None.
        documentation (dict): Dictionary containing meta information and results of the experiment.

    Methods:
        run():
            Executes the experiment by training and testing the specified models on the datasets.
            It creates a directory for results, defines meta information, trains and tests models,
            calculates performance metrics, and saves the results and visualizations.
    """
    def __init__(self, dataSets, use_nn_reference, use_rf_reference, models, number_of_epochs=800, ensemble_size=10, window_size=10, past_values=2, future_values=2, batched_data=False, n_drop_values=10, data_loader=hdata.load_data):
        """
        Initializes the Experiment class with the specified parameters.

        Args:
            dataSets: A dataset or list of datasets to be used in the experiment.
            use_nn_reference (bool): Flag to indicate whether to use a neural network reference model.
            use_rf_reference (bool): Flag to indicate whether to use a random forest reference model.
            models: A model or list of models to be used in the experiment.
            number_of_epochs (int): Number of epochs for training the models. Default is 800.
            ensemble_size (int): Size of the ensemble for each model. Default is 10.
            window_size (int): Size of the window for data preprocessing. Default is 10.
            past_values (int): Number of past values to consider for predictions. Default is 2.
            future_values (int): Number of future values to predict. Default is 2.
            batched_data (bool): Flag to indicate whether data should be batched. Default is False.
            n_drop_values (int): Number of values to drop during error calculation. Default is 10.
            data_loader (function): Function to load and preprocess the data. Default is None.
        """
        if type(dataSets) is not list:
            self.dataSets = [dataSets]
        else:
            self.dataSets = dataSets
        self.use_nn_reference = use_nn_reference
        self.use_rf_reference = use_rf_reference
        if type(models) is not list:
            self.models = [models]
        else:
            self.models = models
        self.number_of_epochs = number_of_epochs
        self.ensemble_size = ensemble_size
        self.window_size = window_size
        self.past_values = past_values
        self.future_values = future_values
        self.batched_data = batched_data
        self.n_drop_values = n_drop_values
        self.data_loader = data_loader

    def run(self):
        """
        Executes the experiment by training and testing the specified models on the datasets.
        It creates a directory for results, defines meta information, trains and tests models,
        calculates performance metrics, and saves the results and visualizations.
        """
        # Create directory for results
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        results_dir = os.path.join("Results", timestamp)
        os.makedirs(results_dir, exist_ok=True)

        # Define the meta information structure
        meta_information = {
            "DataSets": [],
            "Models": [],
            "Data_Preprocessing": {
                "window_size": self.window_size,
                "past_values": self.past_values,
                "future_values": self.future_values,
                "batched_data": self.batched_data,
                "n_drop_values": self.n_drop_values,
            }
        }

        # Define reference models
        reference_models = []
        if self.use_nn_reference:
            model_nn = mnn.get_reference_net()
            reference_models.append(model_nn)
        if self.use_rf_reference:
            model_rf = mrf.get_reference_model()
            reference_models.append(model_rf)

        # Add reference models to meta information
        for model in reference_models:
            model_info = {
                model.name + "_Ensemble": {
                    "NUMBEROFMODELS": self.ensemble_size,
                    **model.get_documentation()
                },
            }
            meta_information["Models"].append(model_info)

        # Add models to meta information
        for model in self.models:
            model_info = {
                model.name + "_Ensemble": {
                    "NUMBEROFMODELS": self.ensemble_size,
                    **model.get_documentation()
                },
            }
            meta_information["Models"].append(model_info)

        # Add datasets to meta information
        for data_params in self.dataSets:
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
        documentation = meta_information

        """ Check if data_params.testing_data_paths is the same """

        """ Prediction """
        results = []
        for i, data in enumerate(self.dataSets):
            print(f"\n===== Verarbeitung: {data.name} =====")
            # Daten laden
            X_train, X_val, X_test, y_train, y_val, y_test = hdata.load_data(
                data, self.past_values, self.future_values, self.window_size, keep_separate=self.batched_data)
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

                for _ in range(self.ensemble_size):
                    model.train_model(X_train, y_train, X_val, y_val, self.number_of_epochs, patience=5)
                    if isinstance(X_test, list):
                        for i, (x, y) in enumerate(zip(X_test, y_test)):
                            _, pred_nn = model.test_model(x, y)
                            nn_preds[i].append(pred_nn.flatten())
                    else:
                        _, pred_nn = model.test_model(X_test, y_test)
                        nn_preds.append(pred_nn.flatten())

                # Fehlerberechnung
                for i, path in enumerate(data.testing_data_paths):
                    name = model.name + "_" + path.replace('.csv', '')

                    mse_nn, std_nn = hdata.calculate_mse_and_std(nn_preds[i], y_test[i].values if isinstance(y_test[i],
                                                                                                       pd.DataFrame) else y_test[i],
                                                                 self.n_drop_values)
                    # Ergebnisse speichern
                    df_list_results[i][name] = np.mean(nn_preds[i], axis=0)
                    results.extend([
                        [data.name + "_" + path.replace('.csv', ''), model.name, mse_nn, std_nn],
                    ])
                    header_list[i].append(name)

            # Train and test models
            for model in self.models:
                # Modellvergleich auf neuen Daten
                nn_preds = [[] for _ in range(len(X_test))] if isinstance(X_test, list) else []

                for _ in range(self.ensemble_size):
                    model.train_model(X_train, y_train, X_val, y_val, self.number_of_epochs, patience=5)
                    if isinstance(X_test, list):
                        for i, (x, y) in enumerate(zip(X_test, y_test)):
                            _, pred_nn = model.test_model(x, y)
                            nn_preds[i].append(pred_nn.flatten())
                    else:
                        _, pred_nn = model.test_model(X_test, y_test)
                        nn_preds.append(pred_nn.flatten())

                # Fehlerberechnung
                for i, path in enumerate(data.testing_data_paths):
                    name = model.name + "_" + path.replace('.csv', '')

                    mse_nn, std_nn = hdata.calculate_mse_and_std(nn_preds[i], y_test[i].values if isinstance(y_test[i],
                                                                                                       pd.DataFrame) else y_test[i],
                                                                 self.n_drop_values)
                    # Ergebnisse speichern
                    df_list_results[i][name] = np.mean(nn_preds[i], axis=0)
                    results.extend([
                        [data.name + "_" + path.replace('.csv', ''), model.name, mse_nn, std_nn],
                    ])
                    header_list[i].append(name)

            """ Dokumentation """
            df_list_data, names = hdata.get_test_data_as_pd(data, past_values=self.past_values, future_values=self.future_values,
                                                   window_size=self.window_size)
            data_dir = os.path.join(results_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            file_paths = [os.path.join(data_dir, f'{data.name}_{name}.csv') for name in names]
            hdata.save_data(df_list_data, file_paths)

            # Save Results in csv
            for y, df, header, path in zip(y_test, df_list_results, header_list, data.testing_data_paths):
                name = path.replace('.csv', '')
                df['y_ground_truth'] = y["curr_x"] if isinstance(y, pd.DataFrame) else y

                # Plot speichern
                plots_dir = os.path.join(results_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)

                plt.figure(figsize=(12, 6))
                plt.plot(df['y_ground_truth'], label='Ground Truth', color='black', linewidth=2)

                for col in header:
                    label = col.replace('_'+name,'')
                    plt.plot(df[col], linestyle='--', label=f'{label}', alpha=0.6)

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

        ax.set_title(f"Vergleich Modelle")
        ax.set_xlabel("DataSets")
        ax.set_ylabel("MSE")
        ax.set_xticks(x + bar_width * (n_models - 1) / 2)
        ax.set_xticklabels(datasets)
        ax.legend()
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, 'model_comparison.png')
        plt.savefig(plot_path)
        plt.close()

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

        self.documentation = documentation