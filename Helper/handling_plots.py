# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:58:39 2024

@author: Jonas Kyrion

SKRIPT DESCRIPTION:
    
"""
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

kit_deep_blue = "#144466"
kit_red = "#B2372C"
kit_green = "#009682"
kit_yellow = "#EEB70D"
kit_color_scale = [kit_deep_blue, kit_green, kit_yellow, kit_red]

def plot_datafram_columns_as_bars(df, xlabel='Labels', ylabel='Mean Squared Error', title='Model Comparison'):
    """
    Plots a comparison of models for each row based on Mean Squared Error (MSE) values using Seaborn.

    Parameters:
    df (pandas DataFrame): DataFrame containing MSE values with models as columns and labels as rows.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    title (str): Title of the plot.

    Returns:
    None
    """
    #sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Define the width of each bar
    bar_width = 0.2

    # Calculate the positions for each group of bars
    positions = np.arange(len(df.columns))

    # Plot each group of bars
    for i, label in enumerate(df.index):
        plt.bar(positions + i * bar_width, df.loc[label], width=bar_width, label=label)

    # Add text annotations for each bar
    for i, label in enumerate(df.index):
        for j, value in enumerate(df.loc[label]):
            plt.text(j + i * bar_width, value, f'{value:.3f}', ha='center', va='bottom', color='black')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(positions + bar_width * (len(df.index) - 1) / 2, df.columns, rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(title='Labels')
    plt.show()
    
def plot_prediction_vs_true(title, pred_parameters1, true_parameters1, ref_parameters = None):
    """
    Plots the predicted values against the true values.
    
    Parameters
    ----------
    title : str
        The title of the plot.
    pred_parameters1 : array-like
        The predicted values.
    true_parameters1 : array-like
        The true values.
    
    Returns
    -------
    None
    """
    x_val = np.linspace(1,len(true_parameters1),len(true_parameters1))
    x_val = x_val/50
    plt.xlabel("Time in s")
    plt.ylabel("Current in A")
    
    # Plot the predicted values
    plt.title(title)
    plt.scatter(x_val,pred_parameters1, s=1, label='Predicted')
    plt.scatter(x_val,true_parameters1, s=1, label='True')
    
    # plot ref
    if ref_parameters is not None:
        plt.scatter(x_val,ref_parameters, s=1, label='Reference')

    # Place the legend inside the plot
    plt.legend(loc='best')

    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_prediction_mean_vs_true(title, pred_parameters, true_parameters, ref_parameters = None, n = 2, s = 1, 
                                 label1='Mean', label2='Mean ref.'):
    """
    Plots the mean of predicted values against the true values with optional reference parameters.
 
    Parameters
    ----------
    title : str
        The title of the plot.
    pred_parameters : array-like
        The predicted values.
    true_parameters : array-like
        The true values.
    ref_parameters : array-like, optional
        The reference parameters for comparison. The default is None.
    n : int, optional
        The number of standard deviations to plot. The default is 2.
    s : int, optional
        The size of the scatter points. The default is 1.
 
    Returns
    -------
    None
    """
    # Convert pred_parameters to a NumPy array
    pred_parameters = np.array(pred_parameters)

    # Calculate the mean and standard deviation of pred_parameters
    mean = np.mean(pred_parameters, axis=0)
    std = np.std(pred_parameters, axis=0)

    # x axis
    x_val = np.linspace(1,len(mean),len(mean))
    x_val = x_val/50
    plt.xlabel("Time in s")
    plt.ylabel("Current in A")
    
    # Plot the true values
    plt.scatter(x_val, true_parameters, s=s, label='True')

    # Plot the mean of pred_parameters
    plt.scatter(x_val, mean, s=s, label=label1)

    # Plot the mean +/- n*std of pred_parameters
    plt.fill_between(x_val, (mean - n * std).squeeze(), (mean + n * std).squeeze(), alpha=0.4, label=label1 + f' +/- {n}*sigma')
    
    if ref_parameters is not None:
        ref_parameters = np.array(ref_parameters)
        mean_ref = np.mean(ref_parameters, axis=0)
        std_ref = np.std(ref_parameters, axis=0)
        # resize
        # Calculate the number of elements to drop from arr2
        num_elements_to_drop = len(mean_ref) - len(mean)        
        if num_elements_to_drop > 0:
            # Drop the endpoints of arr2
            mean_ref = mean_ref[:-num_elements_to_drop]
            std_ref = std_ref[:-num_elements_to_drop]

        plt.scatter(x_val, mean_ref, s=s, label=label2)
        plt.fill_between(x_val, (mean_ref - n * std_ref).squeeze(), (mean_ref + n * std_ref).squeeze(), alpha=0.4, label=label2 + f' +/- {n}*sigma')

        # Set the y range to the maximum element of y data
        plt.ylim(np.min(mean, mean_ref), np.max(mean, mean_ref)*2)
    else:
        # Set the y range to the maximum element of y data
        plt.ylim(np.min(mean), np.max(mean)*1.5)

    # Place the legend inside the plot
    plt.legend(loc='best')

    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_bar(xData, yData, xlabel, ylabel, title):
    """
    Plots a bar chart.

    Parameters
    ----------
    xData : array-like
        The x-axis data.
    yData : array-like
        The y-axis data.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    title : str
        The title of the plot.

    Returns
    -------
    None
    """
    # Balkendiagramm erstellen
    plt.bar(xData, yData)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # Werte über Balken anzeigen
    for i, value in enumerate(yData):
        plt.text(xData[i], value + 0.02, f'{value:.3f}', ha='center')
    
    # x-Achse anpassen
    plt.xticks(xData, xData, rotation=45)
    plt.tight_layout()
    
    plt.show()
    
def plot_bar_std(names, results, xlabel, ylabel, title):
    """
    Plot the mean of losses as a bar plot with standard deviation as error bars.

    Parameters:
    - names: List of names for each dataset.
    - results: List of lists containing loss values for each dataset.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - title: Title of the plot.
    """
    # Calculate mean and standard deviation for each dataset
    means = [np.mean(loss) for loss in results]
    stds = [np.std(loss) for loss in results]

    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, means, yerr=stds, capsize=5, color=kit_color_scale[:len(names)], alpha=0.7)

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=15, ha='center')

    # Adjust layout to prevent x-axis labels from being cut off
    plt.tight_layout()

    # Display mean values as text on the bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, means[i] + stds[i] + 0.01, f'{means[i]:.2f}', ha='center', va='bottom')

    # Display the plot
    plt.show()

def plot_ode_input_vector(title, acceleration, velocity, force, MRR):

    x_val = np.linspace(1,len(acceleration[0]),len(acceleration[0]))
    x_val = x_val/50
    plt.xlabel("Time in s")
    plt.ylabel("acceleration in m/s²")
    
    # Plot the predicted values
    plt.title(title + " acceleration")
    plt.scatter(x_val,acceleration[0], s=1, label='acc_x')
    plt.scatter(x_val,acceleration[1], s=1, label='acc_y')
    plt.scatter(x_val,acceleration[2], s=1, label='acc_z')
    plt.scatter(x_val,acceleration[3], s=1, label='acc_sp')

    plt.legend()
    plt.figure(dpi=1200)
    plt.show()
    
    # Plot the predicted values
    plt.title(title + " velocity")
    plt.scatter(x_val,velocity[0], s=1, label='vel_x')
    plt.scatter(x_val,velocity[1], s=1, label='vel_y')
    plt.scatter(x_val,velocity[2], s=1, label='vel_z')
    plt.scatter(x_val,velocity[3], s=1, label='vel_sp')

    plt.legend()
    plt.figure(dpi=1200)
    plt.show()
    
    # Plot the predicted values
    plt.title(title + " force")
    plt.scatter(x_val,force[0], s=1, label='force_x')
    plt.scatter(x_val,force[1], s=1, label='force_y')
    plt.scatter(x_val,force[2], s=1, label='force_z')
    plt.scatter(x_val,force[3], s=1, label='force_sp')

    plt.legend()
    plt.figure(dpi=1200)
    plt.show()
    
    # Plot the predicted values
    plt.title(title + " MRR")
    plt.scatter(x_val,MRR[0], s=1, label='MRR')

    plt.legend()
    plt.tight_layout()
    plt.figure(dpi=1200)
    plt.show()