import copy
import json
import os

import pandas as pd

import Helper.handling_hyperopt as hyperopt
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf

HEADER = ["DataSet", "DataPath", "Model", "MAE", "StdDev", "MAE_Ensemble", "Predictions", "GroundTruth", "RawData"]
SAMPLINGRATE = 50
AXIS = 'x'

if __name__ == '__main__':
    # Liste für alle DataFrames mit MAE/StdDev
    all_mae_std_dfs = []

    model_prefixes = [
        'Reference_Random_Forest_RandomSampler',
        'Reference_Random_Forest_GridSampler',
        'Reference_Random_Forest_TPESampler'
    ]

    new_names ={
        'Reference_Random_Forest_RandomSampler': '\nRandom Forest\nZufalls-Sampler',
        'Reference_Random_Forest_GridSampler': '\nRandom Forest\nRaster-Sampler',
        'Reference_Random_Forest_TPESampler': '\nRandom Forest\nTPE-Sampler',
    }

    for file in os.listdir('Predictions'):
        if file.endswith('.csv'):
            df = pd.read_csv(f"Predictions/{file}")
            # MAE und StdDev für die Sampler berechnen
            mae_std_df = hplot.calculate_nmae_and_std(df, file, model_prefixes)
            all_mae_std_dfs.append(mae_std_df)

    # Alle DataFrames zu einem einzigen DataFrame kombinieren
    combined_mae_std_df = pd.concat(all_mae_std_dfs, ignore_index=True)

    # HeatmapPlotter initialisieren
    plotter = hplot.HeatmapPlotter(output_dir='Plots_Thesis')

    # Heatmap erstellen
    plot_paths = plotter.create_plots(
        df=combined_mae_std_df,
        title='\nRandom Forest',
        filename_postfix='RF'
    )

    print(f"Heatmaps wurden erstellt: {plot_paths}")

    # ModelHeatmapPlotter initialisieren
    plotter = hplot.ModelHeatmapPlotter(output_dir='Plots_Thesis')

    # Heatmaps für jedes Modell erstellen
    plot_paths = plotter.create_plots(df=combined_mae_std_df, new_names=new_names)

    print(f"Heatmaps wurden erstellt: {plot_paths}")
