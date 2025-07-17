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

def model_controller(params, args):
    K_l, K_p, K_i, F, offset = params
    x_e, a, v, f, t = args

    # Cast zu JAX-kompatiblen Arrays
    x_e = jnp.asarray(x_e)
    a   = jnp.asarray(a)
    v   = jnp.asarray(v)
    f   = jnp.asarray(f)
    t   = jnp.asarray(t)

    def body_fun(carry, i):
        v_e_past, y_i = carry
        v_e = K_l * x_e[i] - v[i]
        u = K_p * (v_e + v_e_past)
        y_diff = u - y_i
        u_a = K_i * y_diff
        y_i_next = y_i + (u_a - y_i ) * F
        return (v_e, y_i_next), y_i_next

    _, y = jax.lax.scan(body_fun, (0, 0), jnp.arange(len(t-1)))
    #y = jnp.insert(y, 0, 0)
    y = y - offset * jnp.ones_like(y)
    return y
def model_linear(params, args):
    p1, p2, p3, p4 = params
    x_e, a, v, f = args
    e = jnp.cumsum(x_e)
    y = p1 * a + p2 * v + p3 * f + p4
    return y

def residual_fn_controller(params, args):
    y_pred = model_controller(params, args[:-1])
    y_true = args[-1]
    return y_pred - y_true

def residual_fn_linear(params, args):
    y_pred = model_linear(params, args[:-1])
    y_true = args[-1]
    return y_pred - y_true

def fit_model_fast(residual_fn, params0, args, name=""):
    """Optimized fitting with multiple acceleration techniques"""
    solver = LevenbergMarquardt(
        rtol=1e-5,
        atol=1e-5,
        norm=max_norm,
        linear_solver=lx.QR(),
        verbose=frozenset()
    )
    result = least_squares(
        residual_fn,
        solver,
        params0,
        args=args,
        max_steps=100000,
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

    path_data = '..\..\DataSets\MergedData'

    all_results = []

    files = os.listdir(path_data)
    files = ['AL_2007_T4_Plate_Normal_1.csv', 'AL_2007_T4_Plate_Depth_1.csv', 'AL_2007_T4_Plate_SF_1.csv']
    files = ['AL_2007_T4_Plate_SF_1.csv', 'AL_2007_T4_Gear_SF_1.csv', 'S235JR_Plate_SF_1.csv', 'S235JR_Gear_SF_1.csv',
             'AL_2007_T4_Plate_Depth_1.csv', 'AL_2007_T4_Gear_Depth_1.csv', 'S235JR_Plate_Depth_1.csv', 'S235JR_Gear_Depth_1.csv',
             'S235JR_Plate_Normal_1.csv', 'S235JR_Gear_Normal_1.csv',
             ]
    for file in files:
        # file = file.replace('.csv', '')
        data = pd.read_csv(f'{path_data}/{file}')
        n = int(data[data['materialremoved_sim'] > 0].index.min() * 0.9)
        data = data.iloc[:n, :]

        indx = 1
        #X_test.rolling(window=50, min_periods=1).mean()
        a = data["a_x"].values
        v = data["v_x"].values
        f_x = data["f_x_sim"].values
        x_e = data["CONT_DEV_X"].values
        t = data.index.values * 0.02
        y_gt = data['curr_x'].values
        print(f'curr_x Anfangswert: {y_gt[0]}')
        print(f'v_x Anfangswert: {v[0]}')
        # Initial parameter guesses
        params0_controller = jnp.array([1.0, 1.0, 1.0, 1.0, 0])
        params0_linear = jnp.array([1.0, 1.0, 1.0, 1.0])

        # Fit the models
        args_controller = (x_e, a, v, f_x, t, y_gt.squeeze())
        params_controller = fit_model_fast(residual_fn_controller, params0_controller, args_controller, "Controller")

        args_linear = (x_e, a, v, f_x, y_gt.squeeze())
        params_linear = fit_model_fast(residual_fn_linear, params0_linear, args_linear, "Linear")

        # Predictions
        y_pred_controller = model_controller(params_controller, args_controller[:-1])
        y_pred_linear = model_linear(params_linear, args_linear[:-1])

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(t, y_gt, label='Ground Truth')
        plt.plot(t, y_pred_controller, label='Controller Model')
        plt.plot(t, y_pred_linear, label='Linear Model')
        plt.xlabel('Time')
        plt.ylabel('Current')
        plt.legend()
        plt.title(file)
        plt.show()

        initial_curr_x = y_gt[0]
        initial_v_x = v[0]

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

    param_names_controller = ['K_l', 'K_p', 'K_i', 'F', 'offset']
    param_names_linear = ['p1', 'p2', 'p3', 'p4']

    # Normieren: min-max über alle Parameter
    all_params = np.hstack([controller_params_all, linear_params_all])
    param_min = all_params.min(axis=0)
    param_max = all_params.max(axis=0)
    param_range = np.clip(param_max - param_min, 1e-8, None)  # gegen Division durch 0

    controller_params_norm = (controller_params_all - param_min[:5]) / param_range[:5]
    linear_params_norm = (linear_params_all - param_min[5:]) / param_range[5:]

    mean_ctrl_norm = controller_params_norm.mean(axis=0)
    std_ctrl_norm = controller_params_norm.std(axis=0)
    mean_lin_norm = linear_params_norm.mean(axis=0)
    std_lin_norm = linear_params_norm.std(axis=0)

    # Plot: normierte Parameter
    fig, ax = plt.subplots(figsize=(10, 5))
    x1 = np.arange(len(mean_ctrl_norm))
    x2 = np.arange(len(mean_lin_norm)) + len(mean_ctrl_norm) + 1

    ax.bar(x1, mean_ctrl_norm, yerr=std_ctrl_norm, capsize=5, label='Controller (normiert)', alpha=0.7)
    ax.bar(x2, mean_lin_norm, yerr=std_lin_norm, capsize=5, label='Linear (normiert)', alpha=0.7)

    xticks = list(x1) + list(x2)
    xtick_labels = param_names_controller + param_names_linear
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45)
    ax.set_title('Normierte Parameter-Mittelwerte und -Standardabweichungen')
    ax.set_ylabel('Normierter Wert (0–1)')
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

    ax.bar(x - bar_width / 2, results_df['mae_controller'], bar_width, label='Controller')
    ax.bar(x + bar_width / 2, results_df['mae_linear'], bar_width, label='Linear')

    ax.set_xticks(x)
    ax.set_xticklabels(results_df['file'], rotation=45, ha='right')
    ax.set_ylabel('MAE')
    ax.set_title('Mittlere Abweichung (MAE) je Datei')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

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