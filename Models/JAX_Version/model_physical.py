import jax.numpy as jnp
import jax
import numpy as np
import optax
import pandas as pd
from jax import lax
import lineax as lx
from optimistix import LevenbergMarquardt, max_norm, least_squares

import Models.model_base as mb
# Annahme: DT ist definiert (falls nicht, bitte anpassen)
DT = 0.02  # Beispielwert

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


class ODEModelJAX(mb.BaseModel):
    def __init__(self, name="ODE_Model_JAX",
                 R0=1.0, C0=1.0, R10=1.0, L10=1.0, R1=1.0, C1=1.0,
                 x0_0=0.0, x1_0=0.0, alpha=1.0,
                 dt=DT, target_channel='curr_x'):
        self.name = name
        self.R0 = R0
        self.C0 = C0
        self.R10 = R10
        self.L10 = L10
        self.R1 = R1
        self.C1 = C1
        self.x0_0 = x0_0
        self.x1_0 = x1_0
        self.alpha = alpha
        self.dt = dt
        self.target_channel = target_channel
        self.axis_map = {'curr_sp': 0, 'curr_x': 1, 'curr_y': 2, 'curr_z': 3}
        if target_channel not in self.axis_map:
            raise ValueError("Ungültiger target_channel.")
        self.axis = self.axis_map[target_channel]

    def get_input_vector_from_df(self, df_list):
        prefix_current = '_1_current'
        axes = ['sp', 'x', 'y', 'z']
        acceleration_list = []
        velocity_list = []
        force_list = []
        time_list = []

        for df in df_list:
            acceleration = []
            velocity = []
            force = []
            for axis in axes:
                acceleration.append(df[['a_' + axis + prefix_current]].values)
                velocity.append(df[['v_' + axis + prefix_current]].values)
                force.append(df[['f_' + axis + '_sim' + prefix_current]].values)

            # Zeit aus Index oder separater Spalte extrahieren
            if 'time' in df.columns:
                time = df['time'].values
            else:
                time = np.arange(len(df)) * self.dt

            acceleration_list.append(np.array(acceleration[self.axis].squeeze()))
            velocity_list.append(np.array(velocity[self.axis].squeeze()))
            force_list.append(np.array(force[self.axis].squeeze()))
            time_list.append(time)

        return acceleration_list, velocity_list, force_list, time_list

    def criterion(self, y_target, y_pred):
        if isinstance(y_target, list) and isinstance(y_pred, list):
            mse_list = [jnp.mean((yt - yp) ** 2) for yt, yp in zip(y_target, y_pred)]
            return jnp.mean(jnp.array(mse_list))
        else:
            if type(y_target) == pd.DataFrame:
                y_target = y_target.values
            return jnp.mean((y_target - y_pred) ** 2)

    @staticmethod
    def equation(params, X):
        a, v, f, t = X
        R0, C0, R10, L10, R1, C1, x0_0, x1_0, alpha = params

        # i10 berechnen
        i10 = v / R0 + a / C0

        # x0 berechnen, numerische Integration
        x0 = jnp.cumsum(v * (t[1] - t[0]) if len(t) > 1 else v * 0.01) + x0_0

        # x1 berechnen (analytische Lösung)
        exponential_term_1 = jnp.exp(-R10 / L10 * t)
        x1 = x0 + (x1_0 + R10 * i10) * exponential_term_1

        # v1 berechnen (Ableitung von x1)
        v1 = jnp.gradient(x1, t[1] - t[0] if len(t) > 1 else 0.01)

        # i21 berechnen
        i21 = v1 / R1 + jnp.gradient(v1, t[1] - t[0] if len(t) > 1 else 0.01) / C1 + i10

        # tau berechnen (Modell-Ausgabe)
        tau_model = alpha * i21

        return tau_model

    def predict(self, X):
        if type(X) is not list:
            X = [X]
        a, v, f, t = self.get_input_vector_from_df(X)

        # Flache Listen für fit_model_fast
        a = jnp.array(jnp.concatenate(a))
        v = jnp.array(jnp.concatenate(v))
        f = jnp.array(jnp.concatenate(f))
        t = jnp.array(jnp.concatenate(t))

        params = (self.R0, self.C0, self.R10, self.L10, self.R1, self.C1,
                  self.x0_0, self.x1_0, self.alpha)
        return self.equation(params, (a, v, f, t))

    def train_model_old(self, X_train, y_train, X_val, y_val, verbose=True, **kwargs):
        """Training mit LevenbergMarquardt wie im Original"""
        # --- Daten vorbereiten ---
        if not isinstance(X_train, list):
            X_train = [X_train]
            y_train = [y_train]

        a_raw, v_raw, f_raw, t_raw = self.get_input_vector_from_df(X_train)
        y = [jnp.array(y.squeeze()) for y in y_train]

        # Flache Listen für fit_model_fast
        a_flat = jnp.array(jnp.concatenate(a_raw))
        v_flat = jnp.array(jnp.concatenate(v_raw))
        f_flat = jnp.array(jnp.concatenate(f_raw))
        t_flat = jnp.array(jnp.concatenate(t_raw))
        y_flat = jnp.array(jnp.concatenate(y))

        def residual_fn_ode(params, args):

            y_pred = self.equation(params, args[:-1])
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

            return (y_pred - y_true) + penalty

        # Initiale Parameter
        params0 = jnp.array([
            self.R0, self.C0, self.R10, self.L10, self.R1, self.C1,
            self.x0_0, self.x1_0, self.alpha
        ])

        # Args für fit_model_fast
        args = (a_flat, v_flat, f_flat, t_flat, y_flat)

        # Training mit LevenbergMarquardt
        result_params = fit_model_fast(residual_fn_ode, params0, args, name="ODE_Model")

        # Beste Parameter setzen
        (self.R0, self.C0, self.R10, self.L10, self.R1, self.C1,
         self.x0_0, self.x1_0, self.alpha) = result_params.tolist()

        if verbose:
            print(f'\nFinal best params: R0={self.R0:.3f}, C0={self.C0:.3f}, R10={self.R10:.3f}, L10={self.L10:.3f}')
            print(
                f'R1={self.R1:.3f}, C1={self.C1:.3f}, x0_0={self.x0_0:.3f}, x1_0={self.x1_0:.3f}, alpha={self.alpha:.3f}')

        return result_params

    def train_model(self, X_train, y_train, X_val, y_val, verbose=True, **kwargs):
        """Training mit LevenbergMarquardt wie im Original"""
        # --- Daten vorbereiten ---
        if not isinstance(X_train, list):
            X_train = [X_train]
            y_train = [y_train]

        a, v, f, t = self.get_input_vector_from_df(X_train)
        y = [jnp.array(y.squeeze()) for y in y_train]

        def residual_fn_ode(params, args):
            a_list, v_list, f_list, t_list, y_list = args
            residuen =  []
            for a, v, f, t, y in zip(a_list, v_list, f_list, t_list, y_list):
                y_pred = self.equation(params, (a, v, f, t))
                y_true = y
                residuen.append(y_true - y_pred)

            residuen = jnp.concatenate(residuen)

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

            return residuen + penalty

        # Initiale Parameter
        params0 = jnp.array([
            self.R0, self.C0, self.R10, self.L10, self.R1, self.C1,
            self.x0_0, self.x1_0, self.alpha
        ])

        # Args für fit_model_fast
        args = (a, v, f, t, y)

        # Training mit LevenbergMarquardt
        result_params = fit_model_fast(residual_fn_ode, params0, args, name="ODE_Model")

        # Beste Parameter setzen
        (self.R0, self.C0, self.R10, self.L10, self.R1, self.C1,
         self.x0_0, self.x1_0, self.alpha) = result_params.tolist()

        if verbose:
            print(f'\nFinal best params: R0={self.R0:.3f}, C0={self.C0:.3f}, R10={self.R10:.3f}, L10={self.L10:.3f}')
            print(
                f'R1={self.R1:.3f}, C1={self.C1:.3f}, x0_0={self.x0_0:.3f}, x1_0={self.x1_0:.3f}, alpha={self.alpha:.3f}')

        return result_params

    def test_model(self, X, y_target):
        prediction = self.predict(X)
        loss = self.criterion(y_target, prediction)
        return loss, prediction

    def get_documentation(self):
        return {
            "description": "This is a JAX version of an ODE-based electrical circuit model with resistors, capacitors, and inductors.",
            "parameters": {
                "name": self.name,
                "R0": self.R0,
                "C0": self.C0,
                "R10": self.R10,
                "L10": self.L10,
                "R1": self.R1,
                "C1": self.C1,
                "x0_0": self.x0_0,
                "x1_0": self.x1_0,
                "alpha": self.alpha,
                "dt": self.dt,
                "target_channel": self.target_channel
            }
        }

class LinearModelJAX(mb.BaseModel):
    def __init__(self, name="Linear_Model_JAX",
                 p1=1.0, p2=1.0, p3=0.0,
                 dt=DT, target_channel='curr_x'):
        self.name = name
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.dt = dt
        self.target_channel = target_channel
        self.axis_map = {'curr_sp': 0, 'curr_x': 1, 'curr_y': 2, 'curr_z': 3}
        if target_channel not in self.axis_map:
            raise ValueError("Ungültiger target_channel.")
        self.axis = self.axis_map[target_channel]

    def get_input_vector_from_df(self, df_list):
        prefix_current = '_1_current'
        axes = ['sp', 'x', 'y', 'z']
        acceleration_list = []
        velocity_list = []
        force_list = []

        for df in df_list:
            acceleration = []
            velocity = []
            force = []
            for axis in axes:
                acceleration.append(df[['a_' + axis + prefix_current]].values)
                velocity.append(df[['v_' + axis + prefix_current]].values)
                force.append(df[['f_' + axis + '_sim' + prefix_current]].values)

            acceleration_list.append(np.array(acceleration[self.axis].squeeze()))
            velocity_list.append(np.array(velocity[self.axis].squeeze()))
            force_list.append(np.array(force[self.axis].squeeze()))

        return acceleration_list, velocity_list, force_list

    def criterion(self, y_target, y_pred):
        if isinstance(y_target, list) and isinstance(y_pred, list):
            mse_list = [jnp.mean((yt - yp) ** 2) for yt, yp in zip(y_target, y_pred)]
            return jnp.mean(jnp.array(mse_list))
        else:
            if type(y_target) == pd.DataFrame:
                y_target = y_target.values
            return jnp.mean((y_target - y_pred) ** 2)

    @staticmethod
    def equation(params, X):
        a_list, v_list, f_list = X
        p1, p2, p3 = params

        a = jnp.asarray(a_list)
        v = jnp.asarray(v_list)
        f = jnp.asarray(f_list)

        y = p1 * a + p2 * v + p3

        return y

    def predict(self, X):
        if type(X) is not list:
            X = [X]
        a, v, f = self.get_input_vector_from_df(X)
        a = jnp.concatenate(a)
        v = jnp.concatenate(v)
        f = jnp.concatenate(f)

        params = (self.p1, self.p2, self.p3)
        return np.array(self.equation(params, (a, v, f)))

    def train_model(self, X_train, y_train, X_val, y_val, verbose=True, **kwargs):
        """Training mit LevenbergMarquardt wie im Original"""
        # --- Daten vorbereiten ---
        if not isinstance(X_train, list):
            X_train = [X_train]
            y_train = [y_train]

        a, v, f = self.get_input_vector_from_df(X_train)
        y = [jnp.array(y.squeeze()) for y in y_train]

        # Flache Listen für fit_model_fast
        a_flat = jnp.concatenate(a)
        v_flat = jnp.concatenate(v)
        f_flat = jnp.concatenate(f)
        y_flat = jnp.concatenate(y)

        def residual_fn_linear(params, args):
            y_pred = self.equation(params, args[:-1])
            y_true = args[-1]
            p1, p2, p3 = params
            return y_pred - y_true + jnp.abs(p3)

        # Initiale Parameter
        params0 = jnp.array([self.p1, self.p2, self.p3])

        # Args für fit_model_fast
        args = (a_flat, v_flat, f_flat, y_flat)

        # Training mit LevenbergMarquardt
        result_params = fit_model_fast(residual_fn_linear, params0, args, name="Linear_Model")

        # Beste Parameter setzen
        self.p1, self.p2, self.p3 = result_params.tolist()

        if verbose:
            print(f'\nFinal best params: p1={self.p1:.3f}, p2={self.p2:.3f}, p3={self.p3:.3f}')

        return result_params

    def test_model(self, X, y_target):
        prediction = self.predict(X)
        loss = self.criterion(y_target, prediction)
        return loss, prediction

    def get_documentation(self):
        return {
            "description": "This is a JAX version of a simple linear model with three parameters.",
            "parameters": {
                "name": self.name,
                "p1": self.p1,
                "p2": self.p2,
                "p3": self.p3,
                "dt": self.dt,
                "target_channel": self.target_channel
            }
        }

class LuGreModelJAX(mb.BaseModel):
    def __init__(self, name="LuGre_Model_Jax",
                 a1=1e1, a2=1e1, a3=1, b=1e1,
                 sigma_0=1e1, sigma_1=1e1 , sigma_2=1e1,
                 f_s=1e-1, f_c=1e1, v_s=1e1,
                 dt=DT, target_channel='curr_x'):
        self.name = name
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.b = b
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.f_s = f_s
        self.f_c = f_c
        self.v_s = v_s
        self.dt = dt

        self.target_channel = target_channel
        self.axis_map = {'curr_sp': 0, 'curr_x': 1, 'curr_y': 2, 'curr_z': 3}
        if target_channel not in self.axis_map:
            raise ValueError("Ungültiger target_channel.")
        self.axis = self.axis_map[target_channel]

    def get_input_vector_from_df(self, df_list):
        prefix_current = '_1_current'
        axes = ['sp', 'x', 'y', 'z']

        acceleration_list = []
        velocity_list = []
        force_list = []
        MRR_list = []

        for df in df_list:
            acceleration = []
            velocity = []
            force = []

            for axis in axes:
                acceleration.append(df[['a_' + axis + prefix_current]].values)
                velocity.append(df[['v_' + axis + prefix_current]].values)
                force.append(df[['f_' + axis + '_sim' + prefix_current]].values)

            MRR = df['materialremoved_sim' + prefix_current].values

            acceleration_list.append(np.array(acceleration[self.axis].squeeze()))
            velocity_list.append(np.array(velocity[self.axis].squeeze()))
            force_list.append(np.array(force[self.axis].squeeze()))
            MRR_list.append(MRR)

        return acceleration_list, velocity_list, force_list, MRR_list

    def criterion(self, y_target, y_pred):
        # Wenn y_target und y_pred Listen sind, berechne den MSE für jedes Paar
        if isinstance(y_target, list) and isinstance(y_pred, list):
            # Berechne den MSE für jedes Paar von y_target und y_pred
            mse_list = [jnp.mean((yt - yp) ** 2) for yt, yp in zip(y_target, y_pred)]
            # Berechne den mittleren MSE über alle Paare
            return jnp.mean(jnp.array(mse_list))
        else:
            # Standard MSE Berechnung für Arrays
            if type(y_target) == pd.DataFrame:
                y_target = y_target.values
            return jnp.mean((y_target - y_pred) ** 2)

    @staticmethod
    def g_fn(v, f_s, f_c, v_s):
        f_c_s = f_c #
        eps = 1e-8
        return f_c_s + (f_s - f_c_s) * jnp.exp(-(v / 0.001) ** 2)  + eps #

    @staticmethod
    def step_fn(carry, inputs):
        z, params = carry
        a, v, f = inputs
        a1, a2, a3, b, sigma_0, sigma_1, sigma_2, f_s, f_c, v_s, dt = params
        dt = DT
        def dz_fn(z, v):
            g_v = LuGreModelJAX.g_fn(v, f_s, f_c, v_s)
            return v - (sigma_0 * jnp.abs(v) / g_v) * z

        # Runge-Kutta 4. Ordnung
        k1 = dz_fn(z, v)
        k2 = dz_fn(z + 0.5 * dt * k1, v)
        k3 = dz_fn(z + 0.5 * dt * k2, v)
        k4 = dz_fn(z + dt * k3, v)

        z_new = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        z_new = jnp.clip(z_new, -1e2, 1e2)

        #g_v = LuGreModelJAX.g_fn(v, f_s, f_c, v_s)
        f_friction = sigma_0 * z_new + sigma_1 * dz_fn(z_new, v) + sigma_2 * v

        y = a1 * a + a2 * f + f_friction + b
        return (z_new, params), y

    @staticmethod
    def equation(params, X):
        a_list, v_list, f_list = X
        y_output = []

        for a, v, f in zip(a_list, v_list, f_list):
            z = jnp.array(0.0)

            # Using scan to iterate over the velocity array
            _, y = lax.scan(LuGreModelJAX.step_fn, (z, params), (a, v, f))

            y_output.append(y)

        return y_output

    @staticmethod
    def staty_state_equation(params, X):
        a_list, v_list, f_list = X
        a1, a2, a3, b, sigma_0, sigma_1, sigma_2, f_s, f_c, v_s, dt = params
        y_output = []

        for a, v, f in zip(a_list, v_list, f_list):
            f_friction = f_c + (f_s -f_c) * jnp.exp(-(v / v_s) ** 2)
            y = a1 * a + a2 * f + jnp.sign(v) * f_friction + b
            y_output.append(y)

        return y_output

    def predict(self, X):
        if type(X) is not list:
            X = [X]
        a, v, f, MRR = self.get_input_vector_from_df(X)

        params = (self.a1, self.a2, self.a3, self.b,
                  self.sigma_0, self.sigma_1, self.sigma_2,
                  self.f_s, self.f_c, self.v_s, self.dt)

        return self.equation(params, (a, v, f))

    def train_model(self, X_train, y_train, X_val, y_val,
                    n_steps=3000, step_chunk=50, verbose=True,
                    patience=1, factor=0.5, min_lr=1e-6, **kwargs):
        # --- Daten vorbereiten ---
        if not isinstance(X_train, list):
            X_train = [X_train]
            y_train = [y_train]
        if not isinstance(X_val, list):
            X_val = [X_val]
            y_val = [y_val]

        a, v, f, MRR = self.get_input_vector_from_df(X_train)
        y = [jnp.array(y.squeeze()) for y in y_train]

        a_val, v_val, f_val, MRR_val = self.get_input_vector_from_df(X_val)
        y_val_jnp = [jnp.array(y.squeeze()) for y in y_val]

        def val_loss(params):
            y_pred = self.equation(params, (a_val, v_val, f_val))
            return np.mean(jnp.array([jnp.mean((y_i - y_pred_i) ** 2) for y_i, y_pred_i in zip(y_val_jnp, y_pred)]))

        def loss_fn(params):
            y_pred = self.equation(params, (a, v, f))
            loss = jnp.mean(jnp.array([jnp.mean((y_i - y_pred_i) ** 2) for y_i, y_pred_i in zip(y, y_pred)]))
            return loss

        # Initiale Parameter
        params = jnp.array([
            self.a1, self.a2, self.a3, self.b,
            self.sigma_0, self.sigma_1, self.sigma_2,
            self.f_s, self.f_c, self.v_s, self.dt
        ])
        learning_rate = 1
        best_val_loss = jnp.inf
        best_params = params
        patience_counter = 0
        total_steps = 0
        lr_reduction_count = 0

        while total_steps < n_steps:
            solver = optax.lbfgs(learning_rate=learning_rate)
            opt_state = solver.init(params)

            value_and_grad_fn = jax.value_and_grad(loss_fn)

            def inner_loop_body(carry, _):
                p, state = carry
                loss_val, grads = value_and_grad_fn(p)
                updates, state = solver.update(
                    grads, state, p, value=loss_val, grad=grads, value_fn=loss_fn
                )
                p = optax.apply_updates(p, updates)
                return (p, state), loss_val

            (params, opt_state), loss_history = jax.lax.scan(
                inner_loop_body, (params, opt_state), None, length=step_chunk
            )

            train_loss_now = loss_fn(params)
            val_loss_now = val_loss(params)

            if verbose:
                print(
                    f"Step {total_steps + step_chunk}: Train loss = {train_loss_now:.4e}, Val loss = {val_loss_now:.4e}")

            if val_loss_now < best_val_loss:
                best_val_loss = val_loss_now
                best_params = params
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    new_lr = max(learning_rate * factor, min_lr)
                    if verbose:
                        print(f"Reducing learning rate from {learning_rate:.4e} to {new_lr:.4e}")
                    learning_rate = new_lr
                    patience_counter = 0
                    lr_reduction_count += 1

                    # Early Stopping Bedingung
                    if lr_reduction_count >= 3:
                        if verbose:
                            print(
                                "Early stopping triggered due to no significant improvement after 3 learning rate reductions.")
                        break

            total_steps += step_chunk

        # Beste Parameter setzen
        (self.a1, self.a2, self.a3, self.b,
         self.sigma_0, self.sigma_1, self.sigma_2,
         self.f_s, self.f_c, self.v_s, self.dt) = best_params.tolist()

        print(
            f'\nFinal best params: a1={self.a1:.3f}, a2={self.a2:.3f}, a3={self.a3:.3f}, b={self.b:.3f}, σ0={self.sigma_0:.3f}, σ1={self.sigma_1:.3f}, σ2={self.sigma_2:.3f}')
        print(
            f'Friction params: f_s={self.f_s:.3f}, f_c={self.f_c:.3f}, v_s={self.v_s:.3f}, dt={self.dt:.3f}')
        print(f'Final validation loss: {best_val_loss:.4e}')

        return best_val_loss


    def test_model(self, X, y_target):
        prediction = self.predict(X)
        loss = self.criterion(y_target, prediction)
        return loss, prediction[0]

    def get_documentation(self):
        return {
            "description": "This is a JAX version of the LuGre friction model with differential step integration using `lax.scan`. ",
            "parameters": {
                "parameters": {
                    "name": self.name,
                    "a1": self.a1,
                    "a2": self.a2,
                    "a3": self.a3,
                    "b": self.b,
                    "sigma_0": self.sigma_0,
                    "sigma_1": self.sigma_1,
                    "sigma_2": self.sigma_2,
                    "f_s": self.f_s,
                    "f_c": self.f_c,
                    "v_s": self.v_s,
                    "dt": self.dt,
                    "target_channel": self.target_channel
                }
            }
        }

class LuGreModelJAX2TwoStage(mb.BaseModel):
    def __init__(self, name="LuGre_Model_Jax_TwoStage",
                 a1=1e1, a2=1e1, a3=1, b=1e1,
                 sigma_0=1e1, sigma_1=1e1, sigma_2=1e1,
                 f_s=1e-1, f_c=1e1, v_s=1e1,
                 dt=0.02, target_channel='curr_x',
                 velocity_threshold=1e-3):
        self.name = name
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.b = b
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.f_s = f_s
        self.f_c = f_c
        self.v_s = v_s
        self.dt = dt
        self.velocity_threshold = velocity_threshold

        self.target_channel = target_channel
        self.axis_map = {'curr_sp': 0, 'curr_x': 1, 'curr_y': 2, 'curr_z': 3}
        if target_channel not in self.axis_map:
            raise ValueError("Ungültiger target_channel.")
        self.axis = self.axis_map[target_channel]

    def get_input_vector_from_df(self, df_list):
        prefix_current = '_1_current'
        axes = ['sp', 'x', 'y', 'z']

        acceleration_list = []
        velocity_list = []
        force_list = []
        MRR_list = []

        for df in df_list:
            acceleration = []
            velocity = []
            force = []

            for axis in axes:
                acceleration.append(df[['a_' + axis + prefix_current]].values)
                velocity.append(df[['v_' + axis + prefix_current]].values)
                force.append(df[['f_' + axis + '_sim' + prefix_current]].values)

            MRR = df['materialremoved_sim' + prefix_current].values

            acceleration_list.append(np.array(acceleration[self.axis].squeeze()))
            velocity_list.append(np.array(velocity[self.axis].squeeze()))
            force_list.append(np.array(force[self.axis].squeeze()))
            MRR_list.append(MRR)

        return acceleration_list, velocity_list, force_list, MRR_list

    def filter_static_data(self, X, y):
        """Filtert Daten mit dv/dt ≈ 0 für statisches Training"""
        a_list, v_list, f_list, _ = self.get_input_vector_from_df(X if isinstance(X, list) else [X])
        y_list = y if isinstance(y, list) else [y]

        static_X = []
        static_y = []
        static_indices = []

        for i, (a, v, f, y_data) in enumerate(zip(a_list, v_list, f_list, y_list)):
            # Berechne dv/dt (Beschleunigung)
            dv_dt = np.abs(a)

            # Finde Indizes wo dv/dt ≈ 0
            static_mask = dv_dt < self.velocity_threshold

            if np.any(static_mask):
                static_indices.append(static_mask)
                # Speichere nur die statischen Teile
                static_X.append({
                    'a': a[static_mask],
                    'v': v[static_mask],
                    'f': f[static_mask]
                })
                static_y.append(y_data.squeeze()[static_mask] if hasattr(y_data, 'squeeze') else y_data[static_mask])

        return static_X, static_y, static_indices

    def criterion(self, y_target, y_pred):
        if isinstance(y_target, list) and isinstance(y_pred, list):
            mse_list = [jnp.mean((yt - yp) ** 2) for yt, yp in zip(y_target, y_pred)]
            return jnp.mean(jnp.array(mse_list))
        else:
            if type(y_target) == pd.DataFrame:
                y_target = y_target.values
            return jnp.mean((y_target - y_pred) ** 2)

    @staticmethod
    def g_fn(v, f_s, v_s, f_c):
        eps = 1e-12 # for numeric stability
        return f_c + (f_s - f_c) * jnp.exp(-(v / v_s) ** 2) + eps

    @staticmethod
    def step_fn(carry, inputs):
        z, params = carry
        a, v, f = inputs
        a1, a2, a3, b, sigma_0, sigma_1, sigma_2, v_s, f_s, f_c, dt = params

        def dz_fn(z, v):
            g_v = LuGreModelJAX2TwoStage.g_fn(v, v_s, f_s, f_c)
            return v - (sigma_0 * jnp.abs(v) / g_v) * z

        # Runge-Kutta 4. Ordnung
        k1 = dz_fn(z, v)
        k2 = dz_fn(z + 0.5 * dt * k1, v)
        k3 = dz_fn(z + 0.5 * dt * k2, v)
        k4 = dz_fn(z + dt * k3, v)

        z_new = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        z_new = jnp.clip(z_new, -1e2, 1e2)

        #g_v = LuGreModelJAX.g_fn(v, f_s, f_c, v_s)
        f_friction = sigma_0 * z_new + sigma_1 * dz_fn(z_new, v) + sigma_2 * v

        y = a1 * a + a2 * f + f_friction + b
        return (z_new, params), y

    @staticmethod
    def static_equation(params, X_static):
        """Statisches Modell für dv/dt ≈ 0"""
        a1, a2, b, f_s, f_c, v_s = params
        y_output = []

        for data in X_static:
            a, v, f = data['a'], data['v'], data['f']
            # Statische Reibung: f_friction = sign(v) * (f_c + (f_s - f_c) * exp(-(v/v_s)^2))
            g_v = LuGreModelJAX2TwoStage.g_fn(v, f_s, f_c, v_s)
            f_friction = jnp.sign(v) * g_v
            y = a1 * a + a2 * f + f_friction + b
            y_output.append(y)

        return y_output

    @staticmethod
    def equation(params, X):
        a_list, v_list, f_list = X
        y_output = []

        for a, v, f in zip(a_list, v_list, f_list):
            z = jnp.array(0.0)
            _, y = lax.scan(LuGreModelJAX2TwoStage.step_fn, (z, params), (a, v, f))
            y_output.append(y)

        return y_output

    def train_static_model(self, X_static, y_static, n_steps=2000, step_chunk=100, verbose=True):
        """Trainiert zunächst das statische Modell"""
        if verbose:
            print("=== Training Static Model ===")

        y_static_jnp = [jnp.array(y.squeeze()) for y in y_static]

        def static_loss_fn(params):
            y_pred = self.static_equation(params, X_static)
            loss = jnp.mean(jnp.array([jnp.mean((y_i - y_pred_i) ** 2) for y_i, y_pred_i in zip(y_static_jnp, y_pred)]))
            return loss

        # Nur statische Parameter optimieren: a1, a2, b, f_s, f_c, v_s
        static_params = jnp.array([self.a1, self.a2, self.b, self.v_s, self.f_s, self.f_c])

        learning_rate = 1
        total_steps = 0

        while total_steps < n_steps:
            solver = optax.lbfgs(learning_rate=learning_rate)
            opt_state = solver.init(static_params)

            value_and_grad_fn = jax.value_and_grad(static_loss_fn)

            def inner_loop_body(carry, _):
                p, state = carry
                loss_val, grads = value_and_grad_fn(p)
                updates, state = solver.update(
                    grads, state, p, value=loss_val, grad=grads, value_fn=static_loss_fn
                )
                p = optax.apply_updates(p, updates)
                return (p, state), loss_val

            (static_params, opt_state), loss_history = jax.lax.scan(
                inner_loop_body, (static_params, opt_state), None, length=step_chunk
            )

            if verbose:
                loss_val = static_loss_fn(static_params)
                print(f"Static training step {total_steps + step_chunk}: loss = {loss_val:.4e}")

            total_steps += step_chunk

        # Setze die gelernten statischen Parameter
        self.a1, self.a2, self.b, self.v_s, self.f_s, self.f_c  = static_params.tolist()

        if verbose:
            print(f"Static model trained. Final loss: {static_loss_fn(static_params):.4e}")
            print(f"Static parameters: a1={self.a1:.3f}, a2={self.a2:.3f}, b={self.b:.3f}")
            print(f"                   f_s={self.f_s:.3f}, f_c={self.f_c:.3f}, v_s={self.v_s:.3f}")

    def predict(self, X):
        if type(X) is not list:
            X = [X]
        a, v, f, MRR = self.get_input_vector_from_df(X)

        params = (self.a1, self.a2, self.a3, self.b,
                  self.sigma_0, self.sigma_1, self.sigma_2,
                  self.v_s, self.f_s, self.f_c, self.dt)

        return self.equation(params, (a, v, f))

    def train_model(self, X_train, y_train, X_val, y_val,
                    n_steps=2500, step_chunk=50, verbose=True,
                    patience=1, factor=0.5, min_lr=1e-6,
                    static_training_steps=500, static_step_chunk=100, **kwargs):
        """Zwei-stufiges Training: erst statisch, dann dynamisch"""

        # Schritt 1: Statisches Modell trainieren
        if verbose:
            print("=== Phase 1: Static Model Training ===")

        X_train_list = X_train if isinstance(X_train, list) else [X_train]
        y_train_list = y_train if isinstance(y_train, list) else [y_train]

        X_static, y_static, _ = self.filter_static_data(X_train_list, y_train_list)

        if len(X_static) > 0:
            self.train_static_model(X_static, y_static, n_steps=static_training_steps,
                                    step_chunk=static_step_chunk, verbose=verbose)
        else:
            if verbose:
                print("No static data found for pre-training")

        # Schritt 2: Vollständiges dynamisches Modell trainieren
        if verbose:
            print("\n=== Phase 2: Dynamic Model Training ===")

        # Bereite Daten für dynamisches Training vor
        if not isinstance(X_train, list):
            X_train = [X_train]
            y_train = [y_train]
        if not isinstance(X_val, list):
            X_val = [X_val]
            y_val = [y_val]

        a, v, f, MRR = self.get_input_vector_from_df(X_train)
        y = [jnp.array(y.squeeze()) for y in y_train]

        a_val, v_val, f_val, MRR_val = self.get_input_vector_from_df(X_val)
        y_val_jnp = [jnp.array(y.squeeze()) for y in y_val]

        def val_loss(dynamic_params):
            full_params = (*dynamic_params, self.f_s, self.f_c, self.dt)
            y_pred = self.equation(full_params, (a_val, v_val, f_val))
            return np.mean(jnp.array([jnp.mean((y_i - y_pred_i) ** 2) for y_i, y_pred_i in zip(y_val_jnp, y_pred)]))

        def loss_fn(dynamic_params):
            full_params = (*dynamic_params, self.f_s, self.f_c, self.dt)
            y_pred = self.equation(full_params, (a, v, f))
            loss = jnp.mean(jnp.array([jnp.mean((y_i - y_pred_i) ** 2) for y_i, y_pred_i in zip(y, y_pred)]))
            return loss

        dynamic_params = jnp.array(
            [self.a1, self.a2, self.a3, self.b, self.sigma_0, self.sigma_1, self.sigma_2, self.v_s])

        learning_rate = 0.1
        best_val_loss = jnp.inf
        best_dynamic_params = dynamic_params
        patience_counter = 0
        total_steps = 0
        lr_reduction_count = 0

        while total_steps < n_steps:
            solver = optax.adam(learning_rate=learning_rate)
            opt_state = solver.init(dynamic_params)

            value_and_grad_fn = jax.value_and_grad(loss_fn)

            def inner_loop_body(carry, _):
                p, state = carry
                loss_val, grads = value_and_grad_fn(p)
                updates, state = solver.update(grads, state)
                p = optax.apply_updates(p, updates)
                return (p, state), loss_val

            (dynamic_params, opt_state), loss_history = jax.lax.scan(
                inner_loop_body, (dynamic_params, opt_state), None, length=step_chunk
            )

            train_loss_now = loss_fn(dynamic_params)
            val_loss_now = val_loss(dynamic_params)

            if verbose:
                print(
                    f"Dynamic step {total_steps + step_chunk}: Train loss = {train_loss_now:.4e}, Val loss = {val_loss_now:.4e}")

            if val_loss_now < best_val_loss:
                best_val_loss = val_loss_now
                best_dynamic_params = dynamic_params
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    new_lr = max(learning_rate * factor, min_lr)
                    if verbose:
                        print(f"Reducing learning rate from {learning_rate:.4e} to {new_lr:.4e}")
                    learning_rate = new_lr
                    patience_counter = 0
                    lr_reduction_count += 1

                    # Early Stopping Bedingung
                    if lr_reduction_count >= 3:
                        if verbose:
                            print(
                                "Early stopping triggered due to no significant improvement after 3 learning rate reductions.")
                        break

            total_steps += step_chunk

        self.a1, self.a2, self.a3, self.b, self.sigma_0, self.sigma_1, self.sigma_2, self.v_s = best_dynamic_params.tolist()

        if verbose:
            print(
                f'\nFinal best params: a1={self.a1:.3f}, a2={self.a2:.3f}, a3={self.a3:.3f}, b={self.b:.3f}, σ0={self.sigma_0:.3f}, σ1={self.sigma_1:.3f}, σ2={self.sigma_2:.3f}')
            print(
                f'Friction params: f_s={self.f_s:.3f}, f_c={self.f_c:.3f}, v_s={self.v_s:.3f}, dt={self.dt:.3f}')
            print(f'Final validation loss: {best_val_loss:.4e}')

        return best_val_loss

    def test_model(self, X, y_target):
        prediction = self.predict(X)
        loss = self.criterion(y_target, prediction)
        return loss, prediction[0]

    def get_documentation(self):
        return {
            "description": "This is a JAX version of the LuGre friction model with differential step integration using `lax.scan`. "
                           "The training process is two-stage, involving static model training followed by dynamic model training.",
            "parameters": {
                "parameters": {
                    "name": self.name,
                    "a1": self.a1,
                    "a2": self.a2,
                    "a3": self.a3,
                    "b": self.b,
                    "sigma_0": self.sigma_0,
                    "sigma_1": self.sigma_1,
                    "sigma_2": self.sigma_2,
                    "f_s": self.f_s,
                    "f_c": self.f_c,
                    "v_s": self.v_s,
                    "dt": self.dt,
                    "target_channel": self.target_channel,
                    "velocity_threshold": self.velocity_threshold
                }
            }
        }