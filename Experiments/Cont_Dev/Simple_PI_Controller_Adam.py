import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import Helper.handling_data as hdata

def model_controller(params, args):
    K, K_l, K_mw, F, p1, p2, p3, p4 = params
    x_e, a, v, f, t = args

    def body_fun(carry, i):
        v_e_past, i_e_past, y_i = carry
        M_l = p3 * f[i]
        v_e = K_l * x_e[i] - v[i]
        i_s = p1 * (v_e_past + v_e)
        i_e = i_s - (y_i + M_l)
        u = i_e_past + i_e
        y_i_next = y_i + (K * u - y_i)
        return (v_e, i_e, y_i_next), y_i_next

    _, y = jax.lax.scan(body_fun, (jnp.zeros_like(t)[0], 0.4, 0.4), jnp.arange(len(t)))
    return y

def loss_fn(params, args):
    y_pred = model_controller(params, args[:-1])
    y_true = args[-1]
    return jnp.mean((y_pred - y_true) ** 2)

def fit_model_adam(params0, args, initial_learning_rate=0.001, num_steps=1000, patience=50, reduction_factor=0.5):
    optimizer = optax.adam(initial_learning_rate)
    opt_init, opt_update = optimizer
    opt_state = opt_init(params0)

    @jax.jit
    def step(params, opt_state, args, learning_rate):
        loss, grads = jax.value_and_grad(loss_fn)(params, args)
        updates, opt_state = opt_update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    params = params0
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    current_learning_rate = initial_learning_rate

    for step_num in range(num_steps):
        params, opt_state, loss = step(params, opt_state, args, current_learning_rate)
        losses.append(loss)

        # Check if the loss has improved
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Reduce learning rate if loss hasn't improved for 'patience' steps
        if patience_counter >= patience:
            current_learning_rate *= reduction_factor
            patience_counter = 0
            optimizer = optax.adam(current_learning_rate)
            opt_init, opt_update = optimizer
            opt_state = opt_init(params)
            print(f"Reduced learning rate to {current_learning_rate}")

    return params, losses

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
    a = jnp.array(X_test[indx]["a_x_1_current"].iloc[:-n].values)
    v = jnp.array(X_test[indx]["v_x_1_current"].iloc[:-n].values)
    f_x = jnp.array(X_test[indx]["f_x_sim_1_current"].iloc[:-n].values)
    x_e = jnp.array(X_test[indx]["CONT_DEV_X_1_current"].iloc[:-n].values)
    t = jnp.array(X_test[indx].index[:-n].values * 0.02)
    y_gt = jnp.array(y_test_original[indx][dataClass.target_channels].iloc[:-n].values)

    # Initial parameter guesses
    params0_controller = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Fit the model
    args_controller = (x_e, a, v, f_x, t, y_gt)
    params_controller, losses = fit_model_adam(params0_controller, args_controller)

    # Predictions
    y_pred_controller = model_controller(params_controller, args_controller[:-1])

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(t, y_gt, label='Ground Truth')
    plt.plot(t, y_pred_controller, label='Controller Model')
    plt.xlabel('Time')
    plt.ylabel('Current')
    plt.legend()
    plt.title('Model Predictions vs Ground Truth')
    plt.show()
