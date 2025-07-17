import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

def fit_armax_model(X, y, order=(1, 0, 1)):
    # Hier wird ein einfaches ARMAX-Modell angepasst
    # X wird als exogene Variable verwendet
    model = ARIMA(y, order=order, exog=X)
    model_fit = model.fit()
    return model_fit

def analyze_parameter_consistency(armax_params_all_1, armax_params_all_2, file_labels, param_names_armax_1, param_names_armax_2):
    lin_df_1 = pd.DataFrame(armax_params_all_1, columns=param_names_armax_1, index=file_labels)
    lin_df_2 = pd.DataFrame(armax_params_all_2, columns=param_names_armax_2, index=file_labels)

    print("=== PARAMETER-KONSISTENZ-ANALYSE ===")
    print("Theoretisch sollten alle Parameter über alle Datensätze konstant sein.\n")

    def calculate_consistency_metrics(df, param_names, model_name):
        print(f"\n--- {model_name} Parameter ---")
        consistency_results = []
        for param in param_names:
            values = df[param].values
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            range_val = max_val - min_val
            cv = np.abs(std_val / mean_val) if mean_val != 0 else np.inf
            rsd = cv * 100
            normalized_range = range_val / np.abs(mean_val) if mean_val != 0 else np.inf
            consistency_results.append({
                'Parameter': param,
                'Mean': mean_val,
                'Std': std_val,
                'Min': min_val,
                'Max': max_val,
                'Range': range_val,
                'CV': cv,
                'RSD (%)': rsd,
                'Norm_Range': normalized_range
            })
            print(f"{param:>8}: Mean={mean_val:8.4f}, Std={std_val:8.4f}, RSD={rsd:6.2f}%, Range={range_val:8.4f}")
        return pd.DataFrame(consistency_results)

    armax_consistency_1 = calculate_consistency_metrics(lin_df_1, param_names_armax_1, "ARMAX Model 1")
    armax_consistency_2 = calculate_consistency_metrics(lin_df_2, param_names_armax_2, "ARMAX Model 2")

    print("\n=== KONSISTENZ-RANKING (nach RSD) ===")
    print("Niedrige RSD = hohe Konsistenz")
    print("\nARMAX Model 1 Parameter (beste bis schlechteste Konsistenz):")
    armax_sorted_1 = armax_consistency_1.sort_values('RSD (%)')
    for idx, row in armax_sorted_1.iterrows():
        print(f"{row['Parameter']:>8}: {row['RSD (%)']:6.2f}% RSD")

    print("\nARMAX Model 2 Parameter (beste bis schlechteste Konsistenz):")
    armax_sorted_2 = armax_consistency_2.sort_values('RSD (%)')
    for idx, row in armax_sorted_2.iterrows():
        print(f"{row['Parameter']:>8}: {row['RSD (%)']:6.2f}% RSD")

    # Visualisierung der Parameter-Konsistenz
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    x_pos = np.arange(len(param_names_armax_1))
    bars1 = axes[0, 0].bar(x_pos, armax_consistency_1['RSD (%)'], alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Parameter')
    axes[0, 0].set_ylabel('RSD (%)')
    axes[0, 0].set_title('ARMAX Model 1 Parameter - Relative Standardabweichung\n(Niedrig = Konsistent)')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(param_names_armax_1, rotation=45)
    axes[0, 0].grid(True, alpha=0.3)

    for bar, rsd in zip(bars1, armax_consistency_1['RSD (%)']):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height, f'{rsd:.1f}%', ha='center', va='bottom', fontsize=9)

    x_pos = np.arange(len(param_names_armax_2))
    bars2 = axes[0, 1].bar(x_pos, armax_consistency_2['RSD (%)'], alpha=0.7, color='lightcoral')
    axes[0, 1].set_xlabel('Parameter')
    axes[0, 1].set_ylabel('RSD (%)')
    axes[0, 1].set_title('ARMAX Model 2 Parameter - Relative Standardabweichung\n(Niedrig = Konsistent)')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(param_names_armax_2, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    for bar, rsd in zip(bars2, armax_consistency_2['RSD (%)']):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height, f'{rsd:.1f}%', ha='center', va='bottom', fontsize=9)

    for i, param in enumerate(param_names_armax_1):
        axes[1, 0].plot(range(len(file_labels)), lin_df_1[param].values, 'o-', label=param, alpha=0.7, markersize=4)
    axes[1, 0].set_xlabel('Datensatz')
    axes[1, 0].set_ylabel('Parameterwert')
    axes[1, 0].set_title('ARMAX Model 1 Parameter - Verlauf über Datensätze')
    axes[1, 0].set_xticks(range(len(file_labels)))
    axes[1, 0].set_xticklabels([f"D{i + 1}" for i in range(len(file_labels))], rotation=45)
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)

    for i, param in enumerate(param_names_armax_2):
        axes[1, 1].plot(range(len(file_labels)), lin_df_2[param].values, 'o-', label=param, alpha=0.7, markersize=4)
    axes[1, 1].set_xlabel('Datensatz')
    axes[1, 1].set_ylabel('Parameterwert')
    axes[1, 1].set_title('ARMAX Model 2 Parameter - Verlauf über Datensätze')
    axes[1, 1].set_xticks(range(len(file_labels)))
    axes[1, 1].set_xticklabels([f"D{i + 1}" for i in range(len(file_labels))], rotation=45)
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'armax_consistency_1': armax_consistency_1,
        'armax_consistency_2': armax_consistency_2,
        'lin_df_1': lin_df_1,
        'lin_df_2': lin_df_2
    }

if __name__ == "__main__":
    path_data = '../../DataSets/Data'
    all_results = []
    files = os.listdir(path_data)

    for file in files:
        if '_1.csv' in file and not 'Blowhole' in file and not 'Ano' in file and not '_2_1' in file:
            print(file)
            data = pd.read_csv(f'{path_data}/{file}')
            n = data[data['materialremoved_sim'] > 0].index.min()
            data = data.iloc[:n, :]
            X_1 = data[["v_x", "a_x"]].values
            X_2 = data[["v_x", "a_x", "v_y", "a_y", "v_z", "a_z", "v_sp", "a_sp"]].values
            y = data['curr_x'].values
            initial_curr_x = y[0]
            initial_v_x = data["v_x"].values[0]
            y = y - initial_curr_x * np.ones(len(y))

            model_1 = fit_armax_model(X_1, y, order=(1, 0, 1))
            model_2 = fit_armax_model(X_2, y, order=(1, 0, 1))

            y_pred_armax_1 = model_1.predict(exog=X_1)
            y_pred_armax_2 = model_2.predict(exog=X_2)

            plt.figure(figsize=(12, 6))
            plt.plot(data.index.values * 0.02, y, label='Ground Truth')
            plt.plot(data.index.values * 0.02, y_pred_armax_1, label='ARMAX Model 1')
            plt.plot(data.index.values * 0.02, y_pred_armax_2, label='ARMAX Model 2')
            plt.xlabel('Time')
            plt.ylabel('Current')
            plt.legend()
            plt.title(file)
            plt.show()

            all_results.append({
                'file': file,
                'mae_armax_1': mean_absolute_error(y, y_pred_armax_1),
                'mae_armax_2': mean_absolute_error(y, y_pred_armax_2),
                'params_armax_1': model_1.params,
                'params_armax_2': model_2.params,
                'initial_curr_x': initial_curr_x,
                'initial_v_x': initial_v_x,
            })

    armax_params_all_1 = []
    armax_params_all_2 = []
    file_labels = []

    for result in all_results:
        armax_params_all_1.append(result['params_armax_1'])
        armax_params_all_2.append(result['params_armax_2'])
        file_labels.append(result['file'])

    # Dynamisch Parameternamen aus den Modellen extrahieren
    param_names_armax_1 = all_results[0]['params_armax_1'].tolist()
    param_names_armax_2 = all_results[0]['params_armax_2'].tolist()

    armax_params_all_1 = np.array(armax_params_all_1)
    armax_params_all_2 = np.array(armax_params_all_2)

    consistency_results = analyze_parameter_consistency(
        armax_params_all_1, armax_params_all_2, file_labels, param_names_armax_1, param_names_armax_2
    )
