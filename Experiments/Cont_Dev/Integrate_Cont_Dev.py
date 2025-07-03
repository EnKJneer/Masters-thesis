import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
from optimistix import LevenbergMarquardt, max_norm, least_squares
import lineax as lx
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

def integrand_x0(x0, t_curr, v_data):
    idx = jnp.argmin(jnp.abs(t_curr - jnp.arange(len(v_data))))
    v_interp = v_data[idx]
    return v_interp

def run_comparison(dataClass):
    X_train, X_val, X_test, y_train, y_val, y_test_original = dataClass.load_data()
    indx = 1
    n = 25
    # Extracting cont_dev_x data and convert to JAX array
    cont_dev_x = jnp.array(X_test[indx]["CONT_DEV_X_1_current"].iloc[:-n])
    curr_x = jnp.array(y_test_original[indx]["curr_x"].iloc[:-n])
    v_x = jnp.array(X_test[indx]["v_x_1_current"].iloc[:-n])
    f_x = jnp.array(X_test[indx]["f_x_sim_1_current"].iloc[:-n])

    # Define the time array
    t = jnp.linspace(0, len(cont_dev_x) - 1, len(cont_dev_x))

    # Integrate cont_dev_x using RK4
    integrated_cont_dev_x = jnp.cumsum(cont_dev_x*1000 + v_x) #rk4_scan_integrate(partial(integrand_x0, v_data=cont_dev_x), jnp.array([cont_dev_x[0]]), t)
    pos_x = jnp.array(v_x/100)

    # Plot the results
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot cont_dev_x and integrated_cont_dev_x on the first axis
    #ax1.plot(t, -f_x, label='f_x', color='blue')
    ax1.plot(t, cont_dev_x*1000 - v_x, label='pos_x', color='blue')
    #ax1.plot(t, integrated_cont_dev_x, label='Integrated cont_dev_x', color='red')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('cont_dev_x', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # Create a second y-axis for curr_x
    ax2 = ax1.twinx()
    ax2.plot(t, curr_x, label='curr_x', color='green')
    ax2.set_ylabel('curr_x', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Integration of cont_dev_x and curr_x over time using RK4')
    plt.show()

if __name__ == "__main__":
    jax.config.update('jax_enable_x64', True)
    dataClass = hdata.Combined_Plate_TrainVal_CONTDEV
    dataClass.window_size = 1
    dataClass.past_values = 0
    dataClass.future_values = 0
    dataClass.add_sign_hold = True
    dataClass.target_channels = ['curr_x']
    run_comparison(dataClass)
