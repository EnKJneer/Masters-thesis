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

@jax.jit
def integrand_ode(z1_dot, t_curr, theta, t_data, v_data, a_data, ip_data):
    idx = jnp.argmin(jnp.abs(t_data - t_curr))
    v_interp = v_data[idx]
    a_interp = a_data[idx]
    ip_interp = ip_data[idx]
    i10 = theta[0] * v_interp + theta[1] * a_interp + theta[2] * ip_interp
    integrant = theta[4] * i10 + theta[3] * v_interp + theta[5] * a_interp
    return integrant * jnp.exp(theta[3] * t_curr)

def residuals_with_integrals(params, args):
    t, v, a, ip, im = args
    theta = params
    i10 = theta[0] * v + theta[1] * a + theta[2] * ip
    z1_dot_values = rk4_scan_integrate(
        integrand_ode, im[0], t, theta, t, v, a, ip
    )
    z1_dot_values = z1_dot_values  * jnp.exp(-theta[3] * t)
    z1_dot_dot = -theta[3] * z1_dot_values + theta[4] * i10 + theta[3] * v + theta[5] * a
    im_model = i10 + z1_dot_values + z1_dot_dot + theta[6]
    return (im_model - im).reshape(-1)


@jax.jit
def residuals_linear_only(params, args):
    """Linear residuals - JIT compiled"""
    _, v, a, ip, im = args
    theta = params
    im_model = theta[0] * v + theta[1] * a + theta[2] * ip + theta[3]
    return (im_model - im).reshape(-1)


def compute_analytical_jacobian(params, args):
    """Compute analytical Jacobian using JAX autodiff"""
    return jax.jacfwd(residuals_with_integrals)(params, args)


def fit_model_fast(residual_fn, params0, args, name="", use_analytical_jacobian=True):
    """Optimized fitting with multiple acceleration techniques"""

    # Strategy 2: More aggressive solver settings
    solver = LevenbergMarquardt(
        rtol=1e-4,  # Relaxed tolerance for speed
        atol=1e-4,  # Relaxed tolerance for speed
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


def smart_parameter_initialization(v, a, ip, im):
    """Smart parameter initialization based on data statistics"""
    # Simple linear regression for initial guess
    X = jnp.column_stack([v, a, ip, jnp.ones(len(v))])
    theta_init_linear = jnp.linalg.lstsq(X, im, rcond=None)[0]

    # Initialize parameters with physics-based reasoning
    params_init = jnp.array([
        theta_init_linear[0],  # theta[0]: v coefficient R0
        theta_init_linear[1],  # theta[1]: a coefficient C0
        theta_init_linear[2],  # theta[2]: ip coefficient
        0.1,  # theta[3]: decay rate L/R
        0.01,  # theta[4]: i10
        0.01,  # theta[5]: integrand a coeff
        0.01,  # theta[6]: integrand ip coeff
        theta_init_linear[3]  # theta[11]: offset
    ])

    return params_init


def run_comparison_optimized(dataClass):
    """Optimized comparison with multiple acceleration strategies"""
    X_train, X_val, X_test, y_train, y_val, y_test_original = dataClass.load_data()
    ind = 1
    n = 25

    # Strategy 3: Use smaller subset for faster testing
    subset_factor = 2  # Use every 2nd point
    v = jnp.array(X_test[ind]['v_x_1_current'][:-n:subset_factor])
    a = jnp.array(X_test[ind]['a_x_1_current'][:-n:subset_factor])
    ip = jnp.array(X_test[ind]['f_x_sim_1_current'][:-n:subset_factor])
    im = jnp.array(y_test_original[ind]['curr_x'][:-n:subset_factor])
    t = jnp.linspace(0, len(v) - 1, len(v))
    args = (t, v, a, ip, im)

    print(f"Optimizing with {len(v)} data points (reduced from {len(X_test[ind]['v_x_1_current'][:-n])})")

    # Strategy 4: Smart parameter initialization
    params_init = smart_parameter_initialization(v, a, ip, im)
    print("Smart initialization:", params_init)

    # Fit models with optimizations
    import time

    # Integral model
    start_time = time.time()
    theta_integrals = fit_model_fast(
        residuals_with_integrals,
        params_init,
        args,
        name="Mit Integralen (Optimiert)",
        use_analytical_jacobian=True
    )
    integral_time = time.time() - start_time

    # Linear model
    start_time = time.time()
    theta_linear = fit_model_fast(
        residuals_linear_only,
        params_init[:4],
        args,
        name="Nur Linear (Optimiert)"
    )
    linear_time = time.time() - start_time

    print(f"Optimization times - Integral: {integral_time:.2f}s, Linear: {linear_time:.2f}s")

    # Reconstruct models for full data (optional: for better plots)
    # Use full data for final evaluation
    v_full = jnp.array(X_test[ind]['v_x_1_current'][:-n])
    a_full = jnp.array(X_test[ind]['a_x_1_current'][:-n])
    ip_full = jnp.array(X_test[ind]['f_x_sim_1_current'][:-n])
    im_full = jnp.array(y_test_original[ind]['curr_x'][:-n])
    t_full = jnp.linspace(0, len(v_full) - 1, len(v_full))

    # Evaluate on full data
    i10_full = theta_integrals[0] * v_full + theta_integrals[1] * a_full + theta_integrals[2] * ip_full
    z1_dot_values_full = rk4_scan_integrate(
        partial(integrand_ode, theta=theta_integrals, t_data=t_full, v_data=v_full, a_data=a_full, ip_data=ip_full),
        im_full[0],
        t_full,
    )
    z1_dot_values_full = z1_dot_values_full * jnp.exp(-theta_integrals[3] * t_full)
    z1_dot_dot_full = -theta_integrals[3] * z1_dot_values_full + theta_integrals[4] * i10_full + theta_integrals[3] * v_full + theta_integrals[5] * a_full
    im_model_integrals = i10_full + z1_dot_values_full + z1_dot_dot_full + theta_integrals[6]
    loss_integrals = jnp.mean((im_full - im_model_integrals) ** 2)

    im_model_linear = theta_linear[0] * v_full + theta_linear[1] * a_full + theta_linear[2] * ip_full + theta_linear[3]
    loss_linear = jnp.mean((im_full - im_model_linear) ** 2)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot for model comparison
    ax1.plot(t_full, im_full, label="Echte curr_x", linewidth=2)
    ax1.plot(t_full, im_model_integrals, '--', label=f"RK4 Integrale (Loss={loss_integrals:.4e})")
    ax1.plot(t_full, im_model_linear, ':', label=f"Linear (Loss={loss_linear:.4e})")
    ax1.set_xlabel("Zeit (Samples)")
    ax1.set_ylabel("curr_x")
    ax1.legend()
    ax1.set_title(f"Optimierter Modellvergleich (Trainiert auf {len(v)} Punkten)")
    ax1.grid(True)

    # Plot for integrated components with different y-axes
    ax3 = ax2.twinx()
    ax2.plot(t_full, z1_dot_values_full, label="z1_dot (RK4)", color='b')
    ax3.plot(t_full, z1_dot_dot_full, label="z1_dot_dot", color='r')
    ax2.set_xlabel("Zeit (Samples)")
    ax2.set_ylabel("z1_dot Werte", color='b')
    ax3.set_ylabel("z1_dot_dot Werte", color='r')
    ax2.tick_params(axis='y', labelcolor='b')
    ax3.tick_params(axis='y', labelcolor='r')
    ax2.set_title("Integrierte Komponenten mit unterschiedlichen y-Achsen")
    ax2.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

# Keep the original function for compatibility
def run_comparison(dataClass):
    """Original function - calls optimized version"""
    return run_comparison_optimized(dataClass)


if __name__ == "__main__":
    # Enable JAX optimizations
    jax.config.update('jax_enable_x64', True)  # Use float64 for better precision

    dataClass = hdata.Combined_Plate_TrainVal
    dataClass.window_size = 1
    dataClass.past_values = 0
    dataClass.future_values = 0
    dataClass.add_sign_hold = True
    dataClass.target_channels = ['curr_x']
    dataClass.header = ["pos_x", "v_x", "a_x", "f_x_sim"]
    run_comparison(dataClass)