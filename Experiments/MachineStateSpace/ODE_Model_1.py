import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from optimistix import LevenbergMarquardt, max_norm, least_squares
import lineax as lx
from functools import partial
import Helper.handling_data as hdata

# Enable JIT compilation for all functions
@partial(jax.jit, static_argnums=(0,))
def rk4_step(f, y, t, dt, *args):
    """Single Runge-Kutta 4th order step - JIT compiled"""
    k1 = f(y, t, *args)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt, *args)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt, *args)
    k4 = f(y + dt * k3, t + dt, *args)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

@partial(jax.jit, static_argnums=(0,))
def rk4_scan_integrate(f, y0, t_array, *static_args):
    def scan_fn(y, t_curr):
        dt = t_array[1] - t_array[0]
        y_next = rk4_step(f, y, t_curr, dt, *static_args)
        return y_next, y_next
    _, y_values = jax.lax.scan(scan_fn, y0, t_array[1:])
    return jnp.concatenate([jnp.array([y0]), y_values])

def central_diff_4th_order(y, dt):
    # Annahme: y ist 1D, gleichmäßig verteilt
    dydt = jnp.zeros_like(y)
    dydt = dydt.at[2:-2].set(
        (y[:-4] - 8 * y[1:-3] + 8 * y[3:-1] - y[4:]) / (12 * dt)
    )
    # Ränder (1. Ordnung)
    dydt = dydt.at[0].set((y[1] - y[0]) / dt)
    dydt = dydt.at[1].set((y[2] - y[0]) / (2 * dt))
    dydt = dydt.at[-2].set((y[-1] - y[-3]) / (2 * dt))
    dydt = dydt.at[-1].set((y[-1] - y[-2]) / dt)
    return dydt

def residuals_nonlinear_model(params, args):
    """Residuen für das erweiterte nichtlineare Modell"""
    t, v, a, ip, tau_measured = args
    theta = params
    # Parameter extrahieren
    R0, C0, R10, L10, R1, C1, v1_0, R21, L21, R2, C2, v2_0, alpha, beta, gamma = theta

    # Straffunktion für negative Parameter
    penalty = 0.0
    penalty += jnp.sum(jnp.where(R0 < 0, -R0 * 1e6, 0))
    penalty += jnp.sum(jnp.where(C0 < 0, -C0 * 1e6, 0))
    penalty += jnp.sum(jnp.where(R10 < 0, -R10 * 1e6, 0))
    penalty += jnp.sum(jnp.where(L10 < 0, -L10 * 1e6, 0))
    penalty += jnp.sum(jnp.where(R1 < 0, -R1 * 1e6, 0))
    penalty += jnp.sum(jnp.where(C1 < 0, -C1 * 1e6, 0))
    penalty += jnp.sum(jnp.where(R21 < 0, -R21 * 1e6, 0))
    penalty += jnp.sum(jnp.where(L21 < 0, -L21 * 1e6, 0))
    penalty += jnp.sum(jnp.where(R2 < 0, -R2 * 1e6, 0))
    penalty += jnp.sum(jnp.where(C2 < 0, -C2 * 1e6, 0))

    # i10 berechnen
    i10 = v / R0 + a / C0 + ip* gamma

    # i10_dot berechnen (numerische Ableitung)
    #i10_dot = jnp.gradient(i10, t[1] - t[0])

    i10_dot = central_diff_4th_order(i10, t[1] - t[0])

    # v1 berechnen (analytische Lösung)
    exponential_term_1 = jnp.exp(-R10 / L10 * t)
    v1 = v + (v1_0 - R10 * R10 / L10 * i10) * exponential_term_1

    # v1_dot berechnen
    v1_dot = a + (R10 / L10) * (v - v1) + R10 * i10_dot

    # i21 berechnen
    i21 = v1 / R1 + v1_dot / C1 + i10

    # i21_dot berechnen (numerische Ableitung)
    #i21_dot = jnp.gradient(i21, t[1] - t[0])
    i21_dot = central_diff_4th_order(i21, t[1] - t[0])

    # v2 berechnen (analytische Lösung)
    exponential_term_2 = jnp.exp(-R21 / L21 * t)
    v2 = v1 + (v2_0 - R21 * R21 / L21 * i21) * exponential_term_2

    # v2_dot berechnen
    v2_dot = v1_dot + R21 * i21_dot + (R21 / L21) * (v1 - v2)

    # im berechnen (jetzt mit v2 und v2_dot)
    im = v2 / R2 + v2_dot / C2 + i21

    # tau berechnen (Modell-Ausgabe)
    tau_model = alpha * im + beta

    return (tau_model - tau_measured).reshape(-1) + penalty

@jax.jit
def residuals_linear_only(params, args):
    """Linear residuals - JIT compiled"""
    _, v, a, ip, tau_measured = args
    theta = params
    tau_model = theta[0] * v + theta[1] * a + theta[2] * ip + theta[3]
    return (tau_model - tau_measured).reshape(-1)

def fit_model_fast(residual_fn, params0, args, name="", use_analytical_jacobian=True):
    """Optimized fitting with multiple acceleration techniques"""
    solver = LevenbergMarquardt(
        rtol=1e-6,
        atol=1e-6,
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
    return result.value

def smart_parameter_initialization(v, a, ip, tau_measured):
    """Smart parameter initialization für das erweiterte Modell"""
    # Einfache lineare Regression für ersten Anhaltspunkt
    X = jnp.column_stack([v, a, ip, jnp.ones(len(v))])
    linear_params = jnp.linalg.lstsq(X, tau_measured, rcond=None)[0]

    # Parameter mit physikalisch sinnvollen Werten initialisieren
    params_init = jnp.array([
        2.0,  # R0 - Widerstand
        0.1,  # C0 - Kapazität
        0.5,  # R10 - Widerstand
        0.05,  # L10 - Induktivität
        1.0,  # R1 - Widerstand
        0.1,  # C1 - Kapazität
        0.0,  # v1_0 - Anfangswert für v1
        0.5,  # R21 - Widerstand (neue Stufe)
        0.05,  # L21 - Induktivität (neue Stufe)
        1.0,  # R2 - Widerstand (Output-Stufe)
        0.1,  # C2 - Kapazität (Output-Stufe)
        v[0],  # v2_0 - Anfangswert für v2
        -0.001,  # alpha - aus linearer Regression
        0.0,  # beta - Offset aus linearer Regression
        linear_params[2]  # gamma - neuer Parameter
    ])
    return params_init

def run_comparison_optimized(dataClass):
    """Optimized comparison mit dem korrekten Modell"""
    X_train, X_val, X_test, y_train, y_val, y_test_original = dataClass.load_data()
    ind = 1
    n = 25
    # Kleinere Datenmenge für schnellere Tests
    subset_factor = 2
    data = X_test[ind]
    v = jnp.array(data['v_x_1_current'][:-n:subset_factor])
    a = jnp.array(data['a_x_1_current'][:-n:subset_factor])
    ip = jnp.array(data['f_x_sim_1_current'][:-n:subset_factor])
    tau_measured = jnp.array(y_test_original[ind]['curr_x'][:-n:subset_factor])  # Hier nehmen wir curr_x als tau
    t = jnp.linspace(0, len(v) - 1, len(v))
    args = (t, v, a, ip, tau_measured)
    print(f"Optimierung mit {len(v)} Datenpunkten")

    # Smarte Parameter-Initialisierung
    params_init = smart_parameter_initialization(v, a, ip, tau_measured)
    print("Smarte Initialisierung:", params_init)

    # Modelle fitten
    import time
    # Nichtlineares Modell
    start_time = time.time()
    theta_nonlinear = fit_model_fast(
        residuals_nonlinear_model,
        params_init,
        args,
        name="Nichtlineares Modell",
        use_analytical_jacobian=False
    )
    nonlinear_time = time.time() - start_time

    # Lineares Modell
    start_time = time.time()
    theta_linear = fit_model_fast(
        residuals_linear_only,
        params_init[:4],
        args,
        name="Lineares Modell"
    )
    linear_time = time.time() - start_time
    print(f"Optimierungszeiten - Nichtlinear: {nonlinear_time:.2f}s, Linear: {linear_time:.2f}s")

    # Vollständige Daten für finale Auswertung
    v_full = jnp.array(X_test[ind]['v_x_1_current'][:-n])
    a_full = jnp.array(X_test[ind]['a_x_1_current'][:-n])
    ip_full = jnp.array(X_test[ind]['f_x_sim_1_current'][:-n])
    tau_measured_full = jnp.array(y_test_original[ind]['curr_x'][:-n])
    t_full = jnp.linspace(0, len(v_full) - 1, len(v_full))

    # Nichtlineares Modell auswerten
    R0, C0, R10, L10, R1, C1, v1_0, R21, L21, R2, C2, v2_0, alpha, beta, gamma = theta_nonlinear
    # i10 berechnen
    i10_full = v_full / R0 + a_full / C0 + ip_full * gamma
    # i10_dot berechnen
    #i10_dot_full = jnp.gradient(i10_full, t_full[1] - t_full[0])
    i10_dot_full = central_diff_4th_order(i10_full, t_full[1] - t_full[0])
    # v1 berechnen (analytische Lösung)
    exponential_term_1_full = jnp.exp(-R10 / L10 * t_full)
    v1_full = v_full + (v1_0 - R10 * R10 / L10 * i10_full) * exponential_term_1_full
    # v1_dot berechnen
    v1_dot_full = a_full + (R10 / L10) * (v_full - v1_full) + R10 * i10_dot_full
    # i21 berechnen
    i21_full = v1_full / R1 + v1_dot_full / C1 + i10_full
    # i21_dot berechnen
    #i21_dot_full = jnp.gradient(i21_full, t_full[1] - t_full[0])
    i21_dot_full = central_diff_4th_order(i21_full, t_full[1] - t_full[0])
    # v2 berechnen (analytische Lösung)
    exponential_term_2_full = jnp.exp(-R21 / L21 * t_full)
    v2_full = v1_full + (v2_0 - R21 * R21 / L21 * i21_full) * exponential_term_2_full
    # v2_dot berechnen
    v2_dot_full = v1_dot_full + R21 * i21_dot_full + (R21 / L21) * (v1_full - v2_full)
    # im berechnen (jetzt mit v2 und v2_dot)
    im_full = v2_full / R2 + v2_dot_full / C2 + i21_full
    # tau berechnen
    tau_model_nonlinear = alpha * im_full + beta
    loss_nonlinear = jnp.mean((tau_measured_full - tau_model_nonlinear) ** 2)

    # Lineares Modell auswerten
    tau_model_linear = theta_linear[0] * v_full + theta_linear[1] * a_full + theta_linear[2] * ip_full + theta_linear[3]
    loss_linear = jnp.mean((tau_measured_full - tau_model_linear) ** 2)

    # Plots erstellen
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Modellvergleich
    ax1.plot(t_full, tau_measured_full, label="Gemessenes τ", linewidth=2, color='black')
    ax1.plot(t_full, tau_model_nonlinear, '--', label=f"Nichtlineares Modell (Loss={loss_nonlinear:.4e})", color='red')
    ax1.plot(t_full, tau_model_linear, ':', label=f"Lineares Modell (Loss={loss_linear:.4e})", color='blue')
    ax1.set_xlabel("Zeit (Samples)")
    ax1.set_ylabel("τ")
    ax1.legend()
    ax1.set_title("Modellvergleich")
    ax1.grid(True)

    # Stufe 1: i10, v1 und v
    ax2.plot(t_full, i10_full, label="i10", color='green', linewidth=1.5)
    ax2.plot(t_full, v1_full, label="v1", color='orange', linewidth=1.5)
    ax2.plot(t_full, v_full, label="v (Input)", color='gray', alpha=0.7)
    ax2.set_xlabel("Zeit (Samples)")
    ax2.set_ylabel("Werte")
    ax2.legend()
    ax2.set_title("Stufe 1: i10, v1 und Eingangssignal v")
    ax2.grid(True)

    # Stufe 2: i21, v2 und v1_dot
    ax3.plot(t_full, i21_full, label="i21", color='purple', linewidth=1.5)
    ax3.plot(t_full, v2_full, label="v2", color='brown', linewidth=1.5)
    ax3.plot(t_full, v1_dot_full, label="v1_dot", color='pink', alpha=0.8)
    ax3.plot(t_full, v2_dot_full, label="v2_dot", color='cyan', alpha=0.8)
    ax3.set_xlabel("Zeit (Samples)")
    ax3.set_ylabel("Werte")
    ax3.legend()
    ax3.set_title("Stufe 2: i21, v2 und Ableitungen")
    ax3.grid(True)

    # Parameter-Darstellung
    param_names = ['R0', 'C0', 'R10', 'L10', 'R1', 'C1', 'v1_0', 'R21', 'L21', 'R2', 'C2', 'v2_0', 'alpha', 'beta', 'gamma']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow', 'lime', 'navy', 'maroon', 'teal']
    bars = ax4.bar(param_names, theta_nonlinear, color=colors)
    ax4.set_ylabel("Parameterwerte")
    ax4.set_title("Gefittete Parameter (Erweiterte Modell)")
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    # Werte auf Balken anzeigen (nur für größere Werte)
    for bar, value in zip(bars, theta_nonlinear):
        height = bar.get_height()
        if abs(height) > 0.001:  # Nur für sichtbare Werte
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                     fontsize=7, rotation=45)

    plt.tight_layout()
    plt.show()

    # Parameter ausgeben
    print("\nGefittete Parameter des erweiterten nichtlinearen Modells:")
    for i, name in enumerate(param_names):
        print(f"{name}: {theta_nonlinear[i]:.6f}")
    print(f"\nModell-Performance:")
    print(f"Nichtlineares Modell MSE: {loss_nonlinear:.6e}")
    print(f"Lineares Modell MSE: {loss_linear:.6e}")
    print(f"Verbesserung: {((loss_linear - loss_nonlinear) / loss_linear * 100):.2f}%")
    print(f"\nZwischenergebnisse (Mittelwerte):")
    print(f"i10: {jnp.mean(i10_full):.4f} ± {jnp.std(i10_full):.4f}")
    print(f"v1: {jnp.mean(v1_full):.4f} ± {jnp.std(v1_full):.4f}")
    print(f"i21: {jnp.mean(i21_full):.4f} ± {jnp.std(i21_full):.4f}")
    print(f"v2: {jnp.mean(v2_full):.4f} ± {jnp.std(v2_full):.4f}")
    print(f"im: {jnp.mean(im_full):.4f} ± {jnp.std(im_full):.4f}")

def run_comparison(dataClass):
    """Original function - calls optimized version"""
    return run_comparison_optimized(dataClass)

if __name__ == "__main__":
    # Enable JAX optimizations
    jax.config.update('jax_enable_x64', True)
    dataClass = hdata.Combined_Plate_TrainVal
    dataClass.window_size = 1
    dataClass.past_values = 0
    dataClass.future_values = 0
    dataClass.add_sign_hold = True
    dataClass.target_channels = ['curr_x']
    dataClass.header = ["pos_x", "v_x", "a_x", "f_x_sim"]
    run_comparison(dataClass)
