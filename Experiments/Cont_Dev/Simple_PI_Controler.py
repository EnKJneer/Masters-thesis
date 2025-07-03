import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
from optimistix import LevenbergMarquardt, max_norm, least_squares
import lineax as lx
import Helper.handling_data as hdata

def model_controller(params, args):
    K, K_l, K_mw, F, p1, p2, p3, p4 = params
    x_e, a, v, f, t = args

    # ⬇️ Cast zu JAX-kompatiblen Arrays
    x_e = jnp.asarray(x_e)
    a   = jnp.asarray(a)
    v   = jnp.asarray(v)
    f   = jnp.asarray(f)
    t   = jnp.asarray(t)

    def body_fun(carry, i):
        v_e_past, i_e_past, y_i = carry
        M_l = p3 * f[i] # p1 * a[i] + p2 * v[i] + p3 * f[i] + p4
        #v_i = v_val + K_mw * (y_val - M_l) * dt
        #y_i_next = y_val + K * (x_e[i] * K_l - v_i)* dt # * jnp.exp(-F * t[i])
        v_e = K_l * x_e[i] - v[i]
        i_s = p1* (v_e_past + v_e)
        i_e = i_s - (y_i + M_l)
        u = i_e_past + i_e
        y_i_next = y_i + (K*u - y_i)
        return (v_e, i_e, y_i_next), y_i_next

    v_val = jnp.zeros_like(t)
    y_val = jnp.zeros_like(t)

    _, y = jax.lax.scan(body_fun, (v_val[0], p2, p4), jnp.arange(len(t)))

    return y
def model_linear(params, args):
    p1, p2, p3, p4 = params
    x_e, a, v, f = args
    e = jnp.cumsum(x_e)
    y = p1 * a + p2 * jnp.sign(v) + p3 * f + p4
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
    return result.value

if __name__ == "__main__":
    jax.config.update('jax_enable_x64', True)

    dataClass = hdata.Combined_Plate_TrainVal_CONTDEV
    dataClass.window_size = 1
    dataClass.past_values = 0
    dataClass.future_values = 0
    dataClass.add_sign_hold = True
    dataClass.target_channels = ['curr_x']
    X_train, X_val, X_test, y_train, y_val, y_test_original = dataClass.load_data()

    indx = 1
    n = 25
    a = X_test[indx]["a_x_1_current"].iloc[:-n].values
    v = X_test[indx]["v_x_1_current"].iloc[:-n].values
    f_x = X_test[indx]["f_x_sim_1_current"].iloc[:-n].values
    x_e = X_test[indx]["CONT_DEV_X_1_current"].iloc[:-n].values
    t = X_test[indx].index[:-n].values * 0.02
    y_gt = y_test_original[indx][dataClass.target_channels].iloc[:-n].values

    # Initial parameter guesses
    params0_controller = jnp.array([1.0, 10000.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
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
    plt.title('Model Predictions vs Ground Truth')
    plt.show()
