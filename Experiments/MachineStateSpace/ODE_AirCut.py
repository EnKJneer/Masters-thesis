import os

import numpy as np
import seaborn as sns  # für hübschere Plots
from jax import jacfwd
import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
from optimistix import LevenbergMarquardt, max_norm, least_squares
import lineax as lx
import Helper.handling_data as hdata

def analyze_parameter_consistency(controller_params_all, linear_params_all, file_labels,
                                  param_names_controller, param_names_linear):
    """
    Analysiert die Konsistenz jedes Parameters über alle Datensätze
    """

    # DataFrames erstellen
    ctrl_df = pd.DataFrame(controller_params_all, columns=param_names_controller, index=file_labels)
    lin_df = pd.DataFrame(linear_params_all, columns=param_names_linear, index=file_labels)

    print("=== PARAMETER-KONSISTENZ-ANALYSE ===")
    print("Theoretisch sollten alle Parameter über alle Datensätze konstant sein.\n")

    # 1. Konsistenz-Metriken berechnen
    def calculate_consistency_metrics(df, param_names, model_name):
        print(f"\n--- {model_name} Parameter ---")

        consistency_results = []

        for param in param_names:
            values = df[param].values

            # Grundstatistiken
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            range_val = max_val - min_val

            # Variationskoeffizient (CV) - Hauptmetrik für Konsistenz
            cv = np.abs(std_val / mean_val) if mean_val != 0 else np.inf

            # Relative Standardabweichung in %
            rsd = cv * 100

            # Normalisierte Spannweite
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

    # Konsistenz-Metriken berechnen
    ctrl_consistency = calculate_consistency_metrics(ctrl_df, param_names_controller, "Controller")
    lin_consistency = calculate_consistency_metrics(lin_df, param_names_linear, "Linear")

    # 2. Konsistenz-Ranking
    print("\n=== KONSISTENZ-RANKING (nach RSD) ===")
    print("Niedrige RSD = hohe Konsistenz")

    print("\nController Parameter (beste bis schlechteste Konsistenz):")
    ctrl_sorted = ctrl_consistency.sort_values('RSD (%)')
    for idx, row in ctrl_sorted.iterrows():
        print(f"{row['Parameter']:>8}: {row['RSD (%)']:6.2f}% RSD")

    print("\nLinear Parameter (beste bis schlechteste Konsistenz):")
    lin_sorted = lin_consistency.sort_values('RSD (%)')
    for idx, row in lin_sorted.iterrows():
        print(f"{row['Parameter']:>8}: {row['RSD (%)']:6.2f}% RSD")

    # 3. Visualisierung der Parameter-Konsistenz
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 3a. RSD Balkendiagramm
    ax1 = axes[0, 0]
    x_pos = np.arange(len(param_names_controller))
    bars1 = ax1.bar(x_pos, ctrl_consistency['RSD (%)'], alpha=0.7, color='skyblue')
    ax1.set_xlabel('Parameter')
    ax1.set_ylabel('RSD (%)')
    ax1.set_title('Controller Parameter - Relative Standardabweichung\n(Niedrig = Konsistent)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(param_names_controller, rotation=45)
    ax1.grid(True, alpha=0.3)

    # Werte auf Balken anzeigen
    for bar, rsd in zip(bars1, ctrl_consistency['RSD (%)']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{rsd:.1f}%', ha='center', va='bottom', fontsize=9)

    ax2 = axes[0, 1]
    x_pos = np.arange(len(param_names_linear))
    bars2 = ax2.bar(x_pos, lin_consistency['RSD (%)'], alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Parameter')
    ax2.set_ylabel('RSD (%)')
    ax2.set_title('Linear Parameter - Relative Standardabweichung\n(Niedrig = Konsistent)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(param_names_linear, rotation=45)
    ax2.grid(True, alpha=0.3)

    for bar, rsd in zip(bars2, lin_consistency['RSD (%)']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{rsd:.1f}%', ha='center', va='bottom', fontsize=9)

    # 3b. Parameterwerte über Datensätze
    ax3 = axes[1, 0]
    for i, param in enumerate(param_names_controller):
        ax3.plot(range(len(file_labels)), ctrl_df[param].values,
                 'o-', label=param, alpha=0.7, markersize=4)
    ax3.set_xlabel('Datensatz')
    ax3.set_ylabel('Parameterwert')
    ax3.set_title('Controller Parameter - Verlauf über Datensätze')
    ax3.set_xticks(range(len(file_labels)))
    ax3.set_xticklabels([f"D{i + 1}" for i in range(len(file_labels))], rotation=45)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    for i, param in enumerate(param_names_linear):
        ax4.plot(range(len(file_labels)), lin_df[param].values,
                 'o-', label=param, alpha=0.7, markersize=4)
    ax4.set_xlabel('Datensatz')
    ax4.set_ylabel('Parameterwert')
    ax4.set_title('Linear Parameter - Verlauf über Datensätze')
    ax4.set_xticks(range(len(file_labels)))
    ax4.set_xticklabels([f"D{i + 1}" for i in range(len(file_labels))], rotation=45)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 4. Detaillierte Boxplots für jeden Parameter
    n_ctrl = len(param_names_controller)
    n_lin = len(param_names_linear)

    # Controller Parameter Boxplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, param in enumerate(param_names_controller):
        ax = axes[i]
        values = ctrl_df[param].values

        # Boxplot
        box = ax.boxplot(values, patch_artist=True)
        box['boxes'][0].set_facecolor('lightblue')

        # Einzelne Punkte
        x_pos = np.ones(len(values)) + np.random.normal(0, 0.02, len(values))
        ax.scatter(x_pos, values, alpha=0.7, color='red', s=30, zorder=3)

        # Statistiken anzeigen
        mean_val = np.mean(values)
        std_val = np.std(values)
        rsd = std_val / np.abs(mean_val) * 100 if mean_val != 0 else 0

        ax.axhline(y=mean_val, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
        ax.set_title(f'{param}\nRSD: {rsd:.1f}%')
        ax.set_ylabel('Wert')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Leere Subplots ausblenden
    for i in range(n_ctrl, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Controller Parameter - Einzelverteilungen', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Linear Parameter Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, param in enumerate(param_names_linear):
        ax = axes[i]
        values = lin_df[param].values

        # Boxplot
        box = ax.boxplot(values, patch_artist=True)
        box['boxes'][0].set_facecolor('lightcoral')

        # Einzelne Punkte
        x_pos = np.ones(len(values)) + np.random.normal(0, 0.02, len(values))
        ax.scatter(x_pos, values, alpha=0.7, color='darkred', s=30, zorder=3)

        # Statistiken anzeigen
        mean_val = np.mean(values)
        std_val = np.std(values)
        rsd = std_val / np.abs(mean_val) * 100 if mean_val != 0 else 0

        ax.axhline(y=mean_val, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
        ax.set_title(f'{param}\nRSD: {rsd:.1f}%')
        ax.set_ylabel('Wert')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle('Linear Parameter - Einzelverteilungen', fontsize=16)
    plt.tight_layout()
    plt.show()

    # 5. Konsistenz-Zusammenfassung
    print("\n=== KONSISTENZ-ZUSAMMENFASSUNG ===")

    # Konsistenz-Kategorien definieren
    def categorize_consistency(rsd):
        if rsd < 5:
            return "Sehr konsistent"
        elif rsd < 15:
            return "Konsistent"
        elif rsd < 30:
            return "Mäßig konsistent"
        else:
            return "Inkonsistent"

    print("\nController Parameter Konsistenz:")
    for idx, row in ctrl_consistency.iterrows():
        category = categorize_consistency(row['RSD (%)'])
        print(f"{row['Parameter']:>8}: {category:>15} (RSD: {row['RSD (%)']:5.1f}%)")

    print("\nLinear Parameter Konsistenz:")
    for idx, row in lin_consistency.iterrows():
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

    identify_outliers(ctrl_df, param_names_controller, "Controller")
    identify_outliers(lin_df, param_names_linear, "Linear")

    return {
        'ctrl_consistency': ctrl_consistency,
        'lin_consistency': lin_consistency,
        'ctrl_df': ctrl_df,
        'lin_df': lin_df
    }

def model_ode(params, args):
    a, v, f, t = args

    # Cast zu JAX-kompatiblen Arrays
    a   = jnp.asarray(a)
    v   = jnp.asarray(v)
    f   = jnp.asarray(f)
    t   = jnp.asarray(t)

    # Parameter extrahieren
    #R0, C0, R10, L10, R1, C1, x1_0, R21, L21, R2, C2, x2_0, alpha, beta, x0_0 = params
    R0, C0, R10, L10, R1, C1, x0_0, x1_0, alpha = params

    # i10 berechnen
    i10 = v / R0 + a / C0

    # x0 berechnen, numerische Integration
    x0 = jnp.cumsum(v) + x0_0

    # x1 berechnen (analytische Lösung)
    exponential_term_1 = jnp.exp(-R10 / L10 * t)
    x1 = x0 + (x1_0 + R10 * i10) * exponential_term_1

    # v1 berechnen (Ableitung von x1)
    v1 = jnp.gradient(x1, t[1] - t[0])

    # i21 berechnen
    i21 = v1 / R1 + jnp.gradient(v1, t[1] - t[0]) / C1 + i10

    # x2 berechnen (analytische Lösung)
    #exponential_term_2 = jnp.exp(-R21 / L21 * t)
    #x2 = x1 + (x2_0 + R21 * i21) * exponential_term_2

    # v2 berechnen (Ableitung von x2)
    #v2 = jnp.gradient(x2, t[1] - t[0])

    # im berechnen
    #im = v2 / R2 + jnp.gradient(v2, t[1] - t[0]) / C2 + i21

    # tau berechnen (Modell-Ausgabe)
    tau_model = alpha * i21

    return tau_model

def model_linear(params, args):
    p1, p2, p3 = params
    a, v, f = args
    y = p1 * a + p2 * v + p3 #* f + p4
    return y

def residual_fn_ode(params, args):
    y_pred = model_ode(params, args[:-1])
    y_true = args[-1]

    # Parameter extrahieren
    R0, C0, R10, L10, R1, C1, x0_0, x1_0, alpha = params

    # Straffunktion für negative Parameter
    penalty = 0.0
    penalty += jnp.sum(jnp.where(R0 < 0, -R0 * 1e6, 0))
    penalty += jnp.sum(jnp.where(C0 < 0, -C0 * 1e6, 0))
    penalty += jnp.sum(jnp.where(R10 < 0, -R10 * 1e6, 0))
    penalty += jnp.sum(jnp.where(L10 < 0, -L10 * 1e6, 0))
    penalty += jnp.sum(jnp.where(R1 < 0, -R1 * 1e6, 0))
    penalty += jnp.sum(jnp.where(C1 < 0, -C1 * 1e6, 0))
    #penalty += jnp.sum(jnp.where(R21 < 0, -R21 * 1e6, 0))
    #penalty += jnp.sum(jnp.where(L21 < 0, -L21 * 1e6, 0))
    #penalty += jnp.sum(jnp.where(R2 < 0, -R2 * 1e6, 0))
    #penalty += jnp.sum(jnp.where(C2 < 0, -C2 * 1e6, 0))

    return (y_pred - y_true) + penalty

def residual_fn_linear(params, args):
    y_pred = model_linear(params, args[:-1])
    y_true = args[-1]
    p1, p2, p3 = params
    return y_pred - y_true + jnp.abs(p3)

def fit_model_fast(residual_fn, params0, args, name=""):
    """Optimized fitting with multiple acceleration techniques"""
    solver = LevenbergMarquardt(
        rtol=1e-4,
        atol=1e-4,
        norm=max_norm,
        linear_solver=lx.QR(),
        verbose=frozenset()
    )
    result = least_squares(
        residual_fn,
        solver,
        params0,
        args=args,
        max_steps=1000000,
        has_aux=False,
        throw=True
    )
    print(f"[{name}] Gefundene Parameter:", result.value)
    print(f"[{name}] MAE:", jnp.mean(jnp.abs(residual_fn(params0, args))))
    return result.value

def compute_mae(y_pred, y_true):
    return jnp.mean(jnp.abs(y_pred - y_true))

def compute_parameter_variance(residual_fn, params, args):
    J = jacfwd(residual_fn)(params, args)
    J = jnp.atleast_2d(J)
    JTJ = J.T @ J
    try:
        cov = jnp.linalg.inv(JTJ)
    except jnp.linalg.LinAlgError:
        cov = jnp.full((len(params), len(params)), jnp.nan)
    variances = jnp.diag(cov)
    return variances

if __name__ == "__main__":
    jax.config.update('jax_enable_x64', True)

    path_data = '..\..\DataSets\Data'

    all_results = []

    files = os.listdir(path_data)
    #files = ['AL_2007_T4_Plate_Normal_1.csv', 'AL_2007_T4_Plate_Depth_1.csv', 'AL_2007_T4_Plate_SF_1.csv']
    #files = ['AL_2007_T4_Plate_SF_1.csv', 'AL_2007_T4_Gear_SF_1.csv', 'S235JR_Plate_SF_1.csv', 'S235JR_Gear_SF_1.csv',
             #'AL_2007_T4_Plate_Depth_1.csv', 'AL_2007_T4_Gear_Depth_1.csv', 'S235JR_Plate_Depth_1.csv', 'S235JR_Gear_Depth_1.csv',
             #'S235JR_Plate_Normal_1.csv', 'S235JR_Gear_Normal_1.csv']
    for file in files:
        if '_1.csv' in file and not 'Blowhole' in file and not 'Ano' in file and not '_2_1' in file:
            print(file)
            # file = file.replace('.csv', '')
            data = pd.read_csv(f'{path_data}/{file}')
            n = data[data['materialremoved_sim'] > 0].index.min()
            data = data.iloc[:n, :]

            indx = 1
            #X_test.rolling(window=50, min_periods=1).mean()
            a = data["a_x"].values
            v = data["v_x"].values
            f_x = data["f_x_sim"].values

            t = data.index.values * 0.02
            y_gt = data['curr_x'].values

            initial_curr_x = y_gt[0]
            initial_v_x = v[0]

            print(f'curr_x Anfangswert: {y_gt[0]}')
            print(f'v_x Anfangswert: {v[0]}')

            y_gt = y_gt - initial_curr_x * np.ones(len(y_gt))

            # Initial parameter guesses
            params0_controller = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, -0.001])
            params0_linear = jnp.array([1.0, 1.0, 1.0])

            # Fit the models
            args_controller = (a, v, f_x, t, y_gt.squeeze())
            params_controller = fit_model_fast(residual_fn_ode, params0_controller, args_controller, "Controller")

            args_linear = (a, v, f_x, y_gt.squeeze())
            params_linear = fit_model_fast(residual_fn_linear, params0_linear, args_linear, "Linear")

            # Predictions
            y_pred_controller = model_ode(params_controller, args_controller[:-1])
            y_pred_linear = model_linear(params_linear, args_linear[:-1])

            # Plotting
            plt.figure(figsize=(12, 6))
            plt.plot(t, y_gt, label='Ground Truth')
            plt.plot(t, y_pred_controller, label='Ode Model')
            plt.plot(t, y_pred_linear, label='Linear Model')
            plt.xlabel('Time')
            plt.ylabel('Current')
            plt.legend()
            plt.title(file)
            plt.show()

            # Speichern der Ergebnisse
            all_results.append({
                'file': file,
                'mae_controller': float(compute_mae(y_pred_controller, y_gt)),
                'mae_linear': float(compute_mae(y_pred_linear, y_gt)),
                'params_controller': params_controller,
                'params_linear': params_linear,
                'initial_curr_x': initial_curr_x,
                'initial_v_x': initial_v_x,
            })

    # Listen für Parameter sammeln
    controller_params_all = []
    linear_params_all = []

    file_labels = []

    for result in all_results:
        controller_params_all.append(np.array(result['params_controller']))
        linear_params_all.append(np.array(result['params_linear']))
        file_labels.append(result['file'])

    # Arrays: [n_files, n_params]
    controller_params_all = np.vstack(controller_params_all)
    linear_params_all = np.vstack(linear_params_all)

    # Mittelwert und Standardabweichung berechnen
    mean_ctrl = controller_params_all.mean(axis=0)
    std_ctrl = controller_params_all.std(axis=0)

    mean_lin = linear_params_all.mean(axis=0)
    std_lin = linear_params_all.std(axis=0)

    param_names_controller = ['R0', 'C0', 'R10', 'L10', 'R1', 'C1', 'x0_0', 'x1_0', 'alpha']
    param_names_linear = ['p1', 'p2', 'p3']

    # Normieren: min-max über alle Parameter
    all_params = np.hstack([controller_params_all, linear_params_all])
    param_min = all_params.min(axis=0)
    param_max = all_params.max(axis=0)
    param_range = np.clip(param_max - param_min, 1e-8, None)  # gegen Division durch 0

    n = len(param_names_controller)
    controller_params_norm = controller_params_all #(controller_params_all - param_min[:n]) / param_range[:n]
    linear_params_norm = linear_params_all # (linear_params_all - param_min[n:]) / param_range[n:]

    mean_ctrl_norm = controller_params_norm.mean(axis=0)
    std_ctrl_norm = controller_params_norm.std(axis=0)
    mean_lin_norm = linear_params_norm.mean(axis=0)
    std_lin_norm = linear_params_norm.std(axis=0)

    # Plot: normierte Parameter
    fig, ax = plt.subplots(figsize=(10, 5))
    x1 = np.arange(len(mean_ctrl_norm))
    x2 = np.arange(len(mean_lin_norm)) + len(mean_ctrl_norm) + 1

    ax.bar(x1, mean_ctrl_norm, yerr=std_ctrl_norm, capsize=5, label='Physical Model (normiert)', alpha=0.7)
    ax.bar(x2, mean_lin_norm, yerr=std_lin_norm, capsize=5, label='Linear (normiert)', alpha=0.7)

    xticks = list(x1) + list(x2)
    xtick_labels = param_names_controller + param_names_linear
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45)
    ax.set_title('Normierte Parameter-Mittelwerte und -Standardabweichungen')
    #ax.set_ylabel('Normierter Wert (0–1)')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    # DataFrame aus den gesammelten Daten
    results_df = pd.DataFrame(all_results)

    # Balkenplot: MAE pro Datei
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.35
    x = np.arange(len(results_df))

    ax.bar(x - bar_width / 2, results_df['mae_controller'], bar_width, label='Physical Model')
    ax.bar(x + bar_width / 2, results_df['mae_linear'], bar_width, label='Linear')

    ax.set_xticks(x)
    ax.set_xticklabels(results_df['file'], rotation=45, ha='right')
    ax.set_ylabel('MAE')
    ax.set_title('Mittlere Abweichung (MAE) je Datei')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    # Verwendung der Funktion:
    consistency_results = analyze_parameter_consistency(
        controller_params_all, linear_params_all, file_labels,
        param_names_controller, param_names_linear
    )

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