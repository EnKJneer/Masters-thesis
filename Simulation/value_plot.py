import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
matplotlib.use('Agg')


def plot(interval: list, predict_curr: pd.DataFrame, true_curr: pd.DataFrame) -> None:    
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    
    # Erster Plot fuer x
    axs[0].plot(true_curr.index, true_curr['curr_x'], marker='o', label='True Curr x')
    axs[0].plot(predict_curr.index, predict_curr['predi_x'], marker='x', label='predic Curr x')
    axs[0].set_xlabel('Zeit')
    axs[0].set_ylabel('Strom')
    axs[0].legend()
    
    for i in interval[0]:
        axs[0].axvspan(i[0],i[1],color='red',alpha=0.3)

    
    # Erster Plot fuer y
    axs[1].plot(true_curr.index, true_curr['curr_y'], marker='o', label='True Curr y')
    axs[1].plot(predict_curr.index, predict_curr['predi_y'], marker='x', label='predic Curr y')
    axs[1].set_xlabel('Zeit')
    axs[1].set_ylabel('Strom')
    axs[1].legend()

    for i in interval[1]:
        axs[1].axvspan(i[0],i[1],color='red',alpha=0.3)
    
    # Erster Plot fuer y
    axs[2].plot(true_curr.index, true_curr['curr_z'], marker='o', label='True Curr z')
    axs[2].plot(predict_curr.index, predict_curr['predi_z'], marker='x', label='predic Curr z')
    axs[2].set_xlabel('Zeit')
    axs[2].set_ylabel('Strom')
    axs[2].legend()
    
    for i in interval[2]:
        axs[2].axvspan(i[0],i[1],color='red',alpha=0.3)
    
        # Erster Plot fuer y
    axs[3].plot(true_curr.index, true_curr['curr_sp'], marker='o', label='True Curr sp')
    axs[3].plot(predict_curr.index, predict_curr['predi_sp'], marker='x', label='predic Curr sp')
    axs[3].set_xlabel('Zeit')
    axs[3].set_ylabel('Strom')
    axs[3].legend()
    
    for i in interval[3]:
        axs[3].axvspan(i[0],i[1],color='red',alpha=0.3)

    plt.show()

def save_plot_to_path(true: pd.DataFrame, predi: pd.DataFrame, interval: list,
                      length_timeline: int, time_stemp: int, axis_monitoring_enabled) -> None:
    
    true_name = ['curr_x', 'curr_y', 'curr_z', 'curr_sp']
    predi_name = ['predi_x', 'predi_y', 'predi_z', 'predi_sp']
    achs_name = ['x','y','z','sp']

    for i in range(len(true_name)):

        if axis_monitoring_enabled[i]:
            plt.ioff()

            # Erstellen Plot aus gegebene Daten
            plt.figure(figsize=(10,5))
            plt.plot(true[true_name[i]], label=f'True {true_name[i]}')
            plt.plot(predi[predi_name[i]], label=f'{predi_name[i]}')

            #Definieren x-Achse Länge durch Anzahl der gesamte Datenpunkt
            plt.xlim(0, length_timeline)
            plt.ylim(-3,3)

            plt.xlabel('Zeit t')
            plt.ylabel('Wert')
            plt.title(f'{achs_name[i]}-Achse')
            plt.legend()

            #Interval hinzufügen

            for j in interval[i]:
                plt.axvspan(j[0],j[1],color='red',alpha=0.3)

            ordner_pfad = f'plots/{achs_name[i]}'
            bild_pfad = os.path.join(ordner_pfad, f'{achs_name[i]}_wert_{time_stemp}.png')
            plt.savefig(bild_pfad)

            plt.close()