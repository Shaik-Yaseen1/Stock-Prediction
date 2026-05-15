"""Two-layer stacked LSTM (Equinox/JAX) for adjusted-close forecasting."""

from __future__ import annotations

import os
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import joblib
import numpy as np
import optax
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SEQUENCE_LENGTH = 60


class LSTMForecaster(eqx.Module):
    lstm1: eqx.nn.LSTMCell
    lstm2: eqx.nn.LSTMCell
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear

    def __call__(self, seq: jax.Array) -> jax.Array:
        """seq shape (SEQ_LEN,) univariate scaled series."""
        xs = seq[:, None]
        zeros1 = jnp.zeros(self.lstm1.hidden_size)
        h0 = (zeros1, zeros1)

        def step1(carry, x):
            new = self.lstm1(jnp.ravel(x), carry)
            return new, new[0]

        _, hs1 = jax.lax.scan(step1, h0, xs)

        zeros2 = jnp.zeros(self.lstm2.hidden_size)
        h02 = (zeros2, zeros2)

        def step2(carry, x):
            new = self.lstm2(x, carry)
            return new, new[0]

        _, hs2 = jax.lax.scan(step2, h02, hs1)
        last = hs2[-1]
        y = self.lin1(last)
        y = jax.nn.relu(y)
        y = self.lin2(y)
        return y.squeeze()


@eqx.filter_jit
def forward(model: LSTMForecaster, seq: jax.Array) -> jax.Array:
    return model(seq)


def _new_model(key: jax.Array) -> LSTMForecaster:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return LSTMForecaster(
        lstm1=eqx.nn.LSTMCell(1, 64, key=k1),
        lstm2=eqx.nn.LSTMCell(64, 32, key=k2),
        lin1=eqx.nn.Linear(32, 16, key=k3),
        lin2=eqx.nn.Linear(16, 1, key=k4),
    )


def series_to_xy(scaled: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(seq_len, len(scaled)):
        x.append(scaled[i - seq_len : i, 0])
        y.append(scaled[i, 0])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def train_lstm(
    closes: pd.Series,
    epochs: int = 25,
    batch_size: int = 32,
    *,
    rng_seed: int = 0,
) -> tuple[LSTMForecaster, MinMaxScaler, dict[str, float]]:
    data = closes.astype(np.float64).values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaled = scaler.fit_transform(data).astype(np.float32)

    if len(scaled) < SEQUENCE_LENGTH + 20:
        raise ValueError(
            f"Need at least {SEQUENCE_LENGTH + 20} trading days; got {len(scaled)}."
        )

    x_all, y_all = series_to_xy(scaled, SEQUENCE_LENGTH)
    split = int(len(x_all) * 0.85)
    x_train, x_val = x_all[:split], x_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]

    key = jax.random.key(rng_seed)
    model = _new_model(key)
    optim = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def batch_loss(m: LSTMForecaster, xb: jax.Array, yb: jax.Array) -> jax.Array:
        preds = jax.vmap(m)(xb)
        return jnp.mean((preds - yb) ** 2)

    @eqx.filter_jit
    def step(
        m: LSTMForecaster, st: Any, xb: jax.Array, yb: jax.Array
    ) -> tuple[LSTMForecaster, Any, jax.Array]:
        def loss_fn(mm: LSTMForecaster) -> jax.Array:
            return batch_loss(mm, xb, yb)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        updates, new_st = optim.update(
            grads, st, eqx.filter(m, eqx.is_array)
        )
        new_m = eqx.apply_updates(m, updates)
        return new_m, new_st, loss

    n_train = len(x_train)
    best_val = float("inf")
    best_model = model
    best_epoch = -1

    rng = np.random.default_rng(rng_seed)

    for ep in range(epochs):
        perm = rng.permutation(n_train)
        train_loss_accum = 0.0
        steps = max(1, int(np.ceil(n_train / batch_size)))
        for s in range(steps):
            idx = perm[s * batch_size : (s + 1) * batch_size]
            xb = jnp.asarray(x_train[idx])
            yb = jnp.asarray(y_train[idx])
            model, opt_state, loss_val = step(model, opt_state, xb, yb)
            train_loss_accum += float(loss_val)

        train_mean = train_loss_accum / steps
        vloss = float(batch_loss(model, jnp.asarray(x_val), jnp.asarray(y_val)))
        if vloss < best_val:
            best_val = vloss
            best_model = model
            best_epoch = ep

        if ep - best_epoch > 8 and ep >= 12:
            break

    metrics = {
        "final_train_loss": float(train_mean),
        "final_val_loss": float(best_val),
        "epochs_ran": ep + 1,
    }

    final_model = best_model if best_epoch >= 0 else model
    return final_model, scaler, metrics


def predict_future(
    model: LSTMForecaster,
    scaler: MinMaxScaler,
    closes: pd.Series,
    days: int = 7,
) -> np.ndarray:
    """Return next `days` predicted prices (inverse scaled)."""
    data = closes.astype(np.float64).values.reshape(-1, 1)
    scaled_full = scaler.transform(data).astype(np.float64)
    window = scaled_full[-SEQUENCE_LENGTH:, 0].astype(np.float32).copy()
    preds_scaled: list[float] = []

    for _ in range(days):
        x = jnp.asarray(window[-SEQUENCE_LENGTH:])
        nxt = float(forward(model, x))
        preds_scaled.append(nxt)
        window = np.append(window, np.float32(nxt))

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return preds


def one_step_on_history(
    model: LSTMForecaster, scaler: MinMaxScaler, closes: pd.Series
) -> float:
    """Scaled one-step predicted next close after the last SEQUENCE_LENGTH bars."""
    data = closes.astype(np.float64).values.reshape(-1, 1)
    scaled_full = scaler.transform(data).astype(np.float32)
    win = jnp.asarray(scaled_full[-SEQUENCE_LENGTH:, 0])
    pred_s = float(forward(model, win))
    return float(scaler.inverse_transform([[pred_s]])[0, 0])


def model_dir(base: str, ticker: str) -> str:
    safe = "".join(c for c in ticker.upper() if c.isalnum())
    path = os.path.join(base, safe)
    os.makedirs(path, exist_ok=True)
    return path


def save_bundle(
    base_dir: str,
    ticker: str,
    model: LSTMForecaster,
    scaler: MinMaxScaler,
    meta: dict,
) -> str:
    d = model_dir(base_dir, ticker)
    eqx.tree_serialise_leaves(os.path.join(d, "model.eqx"), model)
    joblib.dump(scaler, os.path.join(d, "scaler.joblib"))
    joblib.dump(meta, os.path.join(d, "meta.joblib"))
    return d


def load_bundle(
    base_dir: str, ticker: str
) -> tuple[LSTMForecaster | None, MinMaxScaler | None, dict | None]:
    safe = "".join(c for c in ticker.upper() if c.isalnum())
    d = os.path.join(base_dir, safe)
    m_path = os.path.join(d, "model.eqx")
    if not os.path.isfile(m_path):
        return None, None, None

    template = _new_model(jax.random.key(0))
    model = eqx.tree_deserialise_leaves(m_path, template)
    scaler = joblib.load(os.path.join(d, "scaler.joblib"))
    meta = joblib.load(os.path.join(d, "meta.joblib"))
    return model, scaler, meta
