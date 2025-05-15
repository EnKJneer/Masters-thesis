import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def load_data(directory, max_points=None):
    """Load data from CSV files in the given directory with an optional maximum number of data points."""
    dataframes = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)

            # Limit the number of data points if max_points is specified
            if max_points is not None:
                df = df.head(max_points)

            dataframes[filename] = df
    return dataframes

def calculate_losses(dataframes, models = None):
    """Calculate the mean squared error between true and predicted values for each file."""
    losses = {}
    for filename, df in dataframes.items():
        true_col = 'curr_x'
        if models is None:
            pred_cols = [col for col in df.columns if col.startswith('curr_x_pred_')]
        else:
            pred_cols = []
            for col in models:
                pred_cols.append('curr_x_pred_' + col)

        # Drop rows where any prediction column has NaN values
        df = df.dropna(subset=[true_col] + pred_cols)

        # Calculate losses for each prediction column
        file_losses = {}
        for pred_col in pred_cols:
            loss = mean_squared_error(df[true_col], df[pred_col])
            file_losses[pred_col] = loss

        losses[filename] = file_losses
    return losses

def plot_bar_charts(losses):
    """Plot the losses as separate bar charts for each file."""
    for filename, file_losses in losses.items():
        fig, ax = plt.subplots(figsize=(20, 14))  # Increase figure size
        models = list(file_losses.keys())
        loss_values = list(file_losses.values())

        bars = ax.bar(models, loss_values)
        ax.set_ylabel('Mean Squared Error')
        ax.set_title(f'Loss Comparison for {filename}')

        # Adding the value labels on top of the bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval *1.01, round(yval, 5), ha='center', va='bottom')  # Rotate text

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout(pad=3.0)  # Increase padding to accommodate text
        plt.show()

def plot_time_series(dataframes, models = None):
    """Plot the true values and predictions as time series for each file, scaling y-axis to curr_x."""
    for filename, df in dataframes.items():
        true_col = 'curr_x'
        if models is None:
            pred_cols = [col for col in df.columns if col.startswith('curr_x_pred_')]
        else:
            pred_cols = []
            for col in models:
                pred_cols.append('curr_x_pred_' + col)

        plt.figure(figsize=(14, 8))
        plt.plot(df.index, df[true_col], label='True Values (curr_x)', linewidth=2)

        for pred_col in pred_cols:
            plt.plot(df.index, df[pred_col], label=f'Predicted ({pred_col})', linestyle='--')

        # Scale y-axis to curr_x
        plt.ylim(max(df[true_col].min()*1.2, -2), df[true_col].max()*1.2)

        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.title(f'Time Series Plot for {filename}')
        plt.legend()
        plt.tight_layout(pad=3.0)  # Increase padding to accommodate text
        plt.show()

def main(directory, max_points=None):
    """Main function to process files and plot losses and time series."""
    dataframes = load_data(directory, max_points)
    # tabPFN : ['tabPFN_ml_vector', 'tabPFN_online_fill_history', 'tabPFN_online_memory']
    # referenz: ['NN_en', "RF_mini_en", "RF_en", "NN_quanti_en", "NN_riemann_en"]
    models = ["NN_quanti_en", "NN_riemann_en", "RF_en", 'tabPFN_online_memory']
    losses = calculate_losses(dataframes, models)
    plot_bar_charts(losses)
    plot_time_series(dataframes, models)

if __name__ == "__main__":
    directory_path = 'Data'  # Replace with your directory path
    max_points = None # 2100  # Set the maximum number of data points to load
    main(directory_path, max_points)