import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from optimistix import LevenbergMarquardt, max_norm, least_squares
import lineax as lx
import Helper.handling_data as hdata


def residuals_with_integrals(params, args):
    t, v, a, ip, im = args
    theta = params
    dt = jnp.mean(jnp.diff(t))

    integral = jnp.cumsum(theta[0] * v + theta[1] * a + theta[2] * ip) * dt
    double_integral = jnp.cumsum(jnp.cumsum(theta[6] * v + theta[7] * a + theta[8] * ip) * dt) * dt
    im_model = integral + double_integral + theta[3] * v + theta[4] * a + theta[5] * ip

    return (im_model - im).reshape(-1)


def residuals_linear_only(params, args):
    _, v, a, ip, im = args
    theta = params
    im_model = theta[0] * v + theta[1] * a + theta[2] * ip + theta[3]
    return (im_model - im).reshape(-1)


def fit_model(residual_fn, params0, args, name=""):
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
        max_steps=10000,
        has_aux=False,
        throw=True
    )
    print(f"[{name}] Gefundene Parameter:", result.value)
    return result.value


def run_comparison(dataClass):
    X_train, X_val, X_test, y_train, y_val, y_test_original = dataClass.load_data()
    ind = 3
    n = 25

    v = jnp.array(X_test[ind]['v_x_1_current'][:-n])
    a = jnp.array(X_test[ind]['a_x_1_current'][:-n])
    ip = jnp.array(X_test[ind]['f_x_sim_1_current'][:-n])
    im = jnp.array(y_test_original[ind]['curr_x'][:-n])
    t = jnp.linspace(0, len(v) - 1, len(v))

    args = (t, v, a, ip, im)

    # Fit Modell mit Integralen
    theta_integrals = fit_model(
        residuals_with_integrals,
        jnp.array([0.01] * 9),
        args,
        name="Mit Integralen"
    )

    dt = jnp.mean(jnp.diff(t))
    int1 = jnp.cumsum(theta_integrals[0] * v + theta_integrals[1] * a + theta_integrals[2] * ip) * dt
    int2 = jnp.cumsum(jnp.cumsum(theta_integrals[6] * v + theta_integrals[7] * a + theta_integrals[8] * ip) * dt) * dt
    im_model_integrals = int1 + int2 + theta_integrals[3] * v + theta_integrals[4] * a + theta_integrals[5] * ip
    loss_integrals = jnp.mean((im - im_model_integrals) ** 2)

    # Fit lineares Modell
    theta_linear = fit_model(
        residuals_linear_only,
        jnp.array([0.01] * 4),
        args,
        name="Nur Linear"
    )
    im_model_linear = theta_linear[0] * v + theta_linear[1] * a + theta_linear[2] * ip
    loss_linear = jnp.mean((im - im_model_linear) ** 2)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, im, label="Echte curr_x", linewidth=2)
    plt.plot(t, im_model_integrals, '--', label=f"Mit Integralen (Loss={loss_integrals:.4e})")
    plt.plot(t, im_model_linear, ':', label=f"Nur Linear (Loss={loss_linear:.4e})")
    plt.xlabel("Zeit (Samples)")
    plt.ylabel("curr_x")
    plt.legend()
    plt.title("Modellvergleich: Lineare Anteile vs. zusätzliche Integrale")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataClass = hdata.Combined_Plate_TrainVal
    dataClass.window_size = 1
    dataClass.past_values = 0
    dataClass.future_values = 0
    dataClass.add_sign_hold = True
    dataClass.target_channels = ['curr_x']
    dataClass.header = ["pos_x", "v_x", "a_x", "f_x_sim"]

    run_comparison(dataClass)
