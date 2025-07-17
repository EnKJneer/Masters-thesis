import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def fit_linear_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_

def analyze_parameter_consistency(linear_params_all_1, linear_params_all_2, file_labels, param_names_linear_1, param_names_linear_2):
    """
    Analysiert die Konsistenz jedes Parameters über alle Datensätze
    """
    # DataFrames erstellen
    lin_df_1 = pd.DataFrame(linear_params_all_1, columns=param_names_linear_1, index=file_labels)
    lin_df_2 = pd.DataFrame(linear_params_all_2, columns=param_names_linear_2, index=file_labels)

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

    lin_consistency_1 = calculate_consistency_metrics(lin_df_1, param_names_linear_1, "Linear Model 1")
    lin_consistency_2 = calculate_consistency_metrics(lin_df_2, param_names_linear_2, "Linear Model 2")

    print("\n=== KONSISTENZ-RANKING (nach RSD) ===")
    print("Niedrige RSD = hohe Konsistenz")

    print("\nLinear Model 1 Parameter (beste bis schlechteste Konsistenz):")
    lin_sorted_1 = lin_consistency_1.sort_values('RSD (%)')
    for idx, row in lin_sorted_1.iterrows():
        print(f"{row['Parameter']:>8}: {row['RSD (%)']:6.2f}% RSD")

    print("\nLinear Model 2 Parameter (beste bis schlechteste Konsistenz):")
    lin_sorted_2 = lin_consistency_2.sort_values('RSD (%)')
    for idx, row in lin_sorted_2.iterrows():
        print(f"{row['Parameter']:>8}: {row['RSD (%)']:6.2f}% RSD")

    # 3. Visualisierung der Parameter-Konsistenz
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 3a. RSD Balkendiagramm für Modell 1
    ax1 = axes[0, 0]
    x_pos = np.arange(len(param_names_linear_1))
    bars1 = ax1.bar(x_pos, lin_consistency_1['RSD (%)'], alpha=0.7, color='skyblue')
    ax1.set_xlabel('Parameter')
    ax1.set_ylabel('RSD (%)')
    ax1.set_title('Linear Model 1 Parameter - Relative Standardabweichung\n(Niedrig = Konsistent)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(param_names_linear_1, rotation=45)
    ax1.grid(True, alpha=0.3)
    for bar, rsd in zip(bars1, lin_consistency_1['RSD (%)']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{rsd:.1f}%', ha='center', va='bottom', fontsize=9)

    # 3b. RSD Balkendiagramm für Modell 2
    ax2 = axes[0, 1]
    x_pos = np.arange(len(param_names_linear_2))
    bars2 = ax2.bar(x_pos, lin_consistency_2['RSD (%)'], alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Parameter')
    ax2.set_ylabel('RSD (%)')
    ax2.set_title('Linear Model 2 Parameter - Relative Standardabweichung\n(Niedrig = Konsistent)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(param_names_linear_2, rotation=45)
    ax2.grid(True, alpha=0.3)
    for bar, rsd in zip(bars2, lin_consistency_2['RSD (%)']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{rsd:.1f}%', ha='center', va='bottom', fontsize=9)

    # 3c. Parameterwerte über Datensätze für Modell 1
    ax3 = axes[1, 0]
    for i, param in enumerate(param_names_linear_1):
        ax3.plot(range(len(file_labels)), lin_df_1[param].values,
                 'o-', label=param, alpha=0.7, markersize=4)
    ax3.set_xlabel('Datensatz')
    ax3.set_ylabel('Parameterwert')
    ax3.set_title('Linear Model 1 Parameter - Verlauf über Datensätze')
    ax3.set_xticks(range(len(file_labels)))
    ax3.set_xticklabels([f"D{i + 1}" for i in range(len(file_labels))], rotation=45)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)

    # 3d. Parameterwerte über Datensätze für Modell 2
    ax4 = axes[1, 1]
    for i, param in enumerate(param_names_linear_2):
        ax4.plot(range(len(file_labels)), lin_df_2[param].values,
                 'o-', label=param, alpha=0.7, markersize=4)
    ax4.set_xlabel('Datensatz')
    ax4.set_ylabel('Parameterwert')
    ax4.set_title('Linear Model 2 Parameter - Verlauf über Datensätze')
    ax4.set_xticks(range(len(file_labels)))
    ax4.set_xticklabels([f"D{i + 1}" for i in range(len(file_labels))], rotation=45)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 4. Detaillierte Boxplots für jeden Parameter
    n_lin_1 = len(param_names_linear_1)
    n_lin_2 = len(param_names_linear_2)

    # Linear Model 1 Parameter Boxplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, param in enumerate(param_names_linear_1):
        ax = axes[i]
        values = lin_df_1[param].values
        box = ax.boxplot(values, patch_artist=True)
        box['boxes'][0].set_facecolor('lightblue')
        x_pos = np.ones(len(values)) + np.random.normal(0, 0.02, len(values))
        ax.scatter(x_pos, values, alpha=0.7, color='red', s=30, zorder=3)
        mean_val = np.mean(values)
        std_val = np.std(values)
        rsd = std_val / np.abs(mean_val) * 100 if mean_val != 0 else 0
        ax.axhline(y=mean_val, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
        ax.set_title(f'{param}\nRSD: {rsd:.1f}%')
        ax.set_ylabel('Wert')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for i in range(n_lin_1, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Linear Model 1 Parameter - Einzelverteilungen', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Linear Model 2 Parameter Boxplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, param in enumerate(param_names_linear_2):
        ax = axes[i]
        values = lin_df_2[param].values
        box = ax.boxplot(values, patch_artist=True)
        box['boxes'][0].set_facecolor('lightcoral')
        x_pos = np.ones(len(values)) + np.random.normal(0, 0.02, len(values))
        ax.scatter(x_pos, values, alpha=0.7, color='darkred', s=30, zorder=3)
        mean_val = np.mean(values)
        std_val = np.std(values)
        rsd = std_val / np.abs(mean_val) * 100 if mean_val != 0 else 0
        ax.axhline(y=mean_val, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
        ax.set_title(f'{param}\nRSD: {rsd:.1f}%')
        ax.set_ylabel('Wert')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for i in range(n_lin_2, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Linear Model 2 Parameter - Einzelverteilungen', fontsize=16)
    plt.tight_layout()
    plt.show()

    # 5. Konsistenz-Zusammenfassung
    print("\n=== KONSISTENZ-ZUSAMMENFASSUNG ===")

    def categorize_consistency(rsd):
        if rsd < 5:
            return "Sehr konsistent"
        elif rsd < 15:
            return "Konsistent"
        elif rsd < 30:
            return "Mäßig konsistent"
        else:
            return "Inkonsistent"

    print("\nLinear Model 1 Parameter Konsistenz:")
    for idx, row in lin_consistency_1.iterrows():
        category = categorize_consistency(row['RSD (%)'])
        print(f"{row['Parameter']:>8}: {category:>15} (RSD: {row['RSD (%)']:5.1f}%)")

    print("\nLinear Model 2 Parameter Konsistenz:")
    for idx, row in lin_consistency_2.iterrows():
        category = categorize_consistency(row['RSD (%)'])
        print(f"{row['Parameter']:>8}: {category:>15} (RSD: {row['RSD (%)']:5.1f}%)")

    # 6. Ausreißer-Identifikation
    print("\n=== AUSREISSER-IDENTIFIKATION ===")

    def identify_outliers(df, param_names, model_name):
        print(f"\n{model_name} Parameter Ausreißer:")
        for param in param_names:
            values = df[param].values
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = []
            for i, val in enumerate(values):
                if val < lower_bound or val > upper_bound:
                    outliers.append((file_labels[i], val))
            if outliers:
                print(f"  {param}: {len(outliers)} Ausreißer")
                for file, val in outliers:
                    print(f"    {file}: {val:.4f}")
            else:
                print(f"  {param}: Keine Ausreißer")

    identify_outliers(lin_df_1, param_names_linear_1, "Linear Model 1")
    identify_outliers(lin_df_2, param_names_linear_2, "Linear Model 2")

    return {
        'lin_consistency_1': lin_consistency_1,
        'lin_consistency_2': lin_consistency_2,
        'lin_df_1': lin_df_1,
        'lin_df_2': lin_df_2
    }

if __name__ == "__main__":
    path_data = '../../DataSets/Data'
    all_results = []
    files = os.listdir(path_data)
    param_names_linear_1 = ['offset', 'theta_a_x', 'theta_v_x']  # Parameter für Modell 1
    param_names_linear_2 = ['offset', 'theta_a_x', 'theta_v_x', 'theta_a_y', 'theta_v_y', 'theta_a_z', 'theta_v_z', 'theta_a_sp', 'theta_v_sp']  # Parameter für Modell 2

    for file in files:
        if '_1.csv' in file and not 'Blowhole' in file and not 'Ano' in file and not '_2_1' in file:
            print(file)
            data = pd.read_csv(f'{path_data}/{file}')
            n = data[data['materialremoved_sim'] > 0].index.min()
            data = data.iloc[:n, :]

            X_1 = np.column_stack([np.ones(len(data)), data[["v_x", "a_x"]].values])
            X_2 = np.column_stack([np.ones(len(data)), data[["v_x", "a_x", "v_y", "a_y", "v_z", "a_z", "v_sp", "a_sp"]].values])
            y = data['curr_x'].values

            initial_curr_x = y[0]
            initial_v_x = data["v_x"].values[0]

            y = y - initial_curr_x * np.ones(len(y))

            params_linear_1 = fit_linear_model(X_1, y)
            params_linear_2 = fit_linear_model(X_2, y)

            y_pred_linear_1 = X_1 @ params_linear_1
            y_pred_linear_2 = X_2 @ params_linear_2

            plt.figure(figsize=(12, 6))
            plt.plot(data.index.values * 0.02, y, label='Ground Truth')
            plt.plot(data.index.values * 0.02, y_pred_linear_1, label='Linear Model 1')
            plt.plot(data.index.values * 0.02, y_pred_linear_2, label='Linear Model 2')
            plt.xlabel('Time')
            plt.ylabel('Current')
            plt.legend()
            plt.title(file)
            plt.show()

            all_results.append({
                'file': file,
                'mae_linear_1': mean_absolute_error(y, y_pred_linear_1),
                'mae_linear_2': mean_absolute_error(y, y_pred_linear_2),
                'params_linear_1': params_linear_1,
                'params_linear_2': params_linear_2,
                'initial_curr_x': initial_curr_x,
                'initial_v_x': initial_v_x,
            })

    linear_params_all_1 = []
    linear_params_all_2 = []
    file_labels = []

    for result in all_results:
        linear_params_all_1.append(result['params_linear_1'])
        linear_params_all_2.append(result['params_linear_2'])
        file_labels.append(result['file'])

    linear_params_all_1 = np.vstack(linear_params_all_1)
    linear_params_all_2 = np.vstack(linear_params_all_2)

    consistency_results = analyze_parameter_consistency(
        linear_params_all_1, linear_params_all_2, file_labels, param_names_linear_1, param_names_linear_2
    )

    # DataFrame aus den gesammelten Daten
    results_df = pd.DataFrame(all_results)

    # Plot: Anfangswerte von curr_x und v_x
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(results_df['file'], results_df['initial_curr_x'], marker='o', label='curr_x[0]')
    ax.plot(results_df['file'], results_df['initial_v_x'], marker='s', label='v_x[0]')
    ax.set_ylabel('Anfangswert')
    ax.set_title('Anfangswerte der Datenreihen')
    ax.set_xticklabels(results_df['file'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Balkenplot: MAE pro Datei
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.35
    x = np.arange(len(results_df))

    ax.bar(x - bar_width / 2, results_df['mae_linear_1'], bar_width, label='Linear 1')
    ax.bar(x + bar_width / 2, results_df['mae_linear_2'], bar_width, label='Linear 2')

    ax.set_xticks(x)
    ax.set_xticklabels(results_df['file'], rotation=45, ha='right')
    ax.set_ylabel('MAE')
    ax.set_title('Mittlere Abweichung (MAE) je Datei')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
