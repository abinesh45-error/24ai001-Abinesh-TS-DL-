import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from google.colab import files
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

uploaded = files.upload()
filename = next(iter(uploaded))

if filename.lower().endswith(".zip"):
    with zipfile.ZipFile(filename, "r") as z:
        csv_files = [
            f for f in z.namelist()
            if f.lower().endswith(".csv")
        ]
        if not csv_files:
            raise ValueError("The ZIP file does not contain a CSV file.")

        csv_name = next(
            (f for f in csv_files
             if os.path.basename(f).startswith("train_")),
            csv_files[0]
        )
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
else:
    df = pd.read_csv(filename)

page_col = "Page" if "Page" in df.columns else df.columns[0]

date_info = []
for col in df.columns:
    if col == page_col:
        continue
    parsed = pd.to_datetime(col, errors="coerce")
    if not pd.isna(parsed):
        date_info.append((col, parsed))

date_info.sort(key=lambda x: x[1])
date_cols = [x[0] for x in date_info]
date_index = pd.DatetimeIndex([x[1] for x in date_info])

if not date_cols:
    raise ValueError("No date columns were detected in the uploaded dataset.")

df[date_cols] = df[date_cols].apply(pd.to_numeric, errors="coerce")

df["available_days"] = df[date_cols].notna().sum(axis=1)
selected = df.nlargest(5, "available_days").copy()

traffic = selected.set_index(page_col)[date_cols].T
traffic.index = date_index
traffic.index.name = "Date"
traffic = traffic.sort_index()
traffic = traffic.loc[~traffic.index.duplicated(keep="first")]
traffic = traffic.asfreq("D")
traffic = traffic.dropna(axis=1, how="all")

if len(traffic) < 356 or traffic.shape[1] != 5:
    raise ValueError(
        "The dataset must contain at least 356 daily records "
        "and five usable articles."
    )

traffic = traffic.interpolate(method="time", limit_area="inside")
traffic = traffic.ffill().bfill()

traffic.plot(subplots=True, layout=(3, 2), figsize=(14, 9), legend=False)
plt.tight_layout()
plt.show()

LOOKBACK = 30
N = len(traffic)

train_end = int(N * 0.70)
val_end = int(N * 0.85)

scaler = MinMaxScaler()
scaler.fit(traffic.values[:train_end])
scaled = scaler.transform(traffic.values).astype(np.float32)

def make_sequences(start_target, end_target):
    X, y, indices = [], [], []
    for i in range(max(LOOKBACK, start_target), end_target):
        X.append(scaled[i - LOOKBACK:i])
        y.append(scaled[i])
        indices.append(i)
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(indices)
    )

X_train, y_train, _ = make_sequences(LOOKBACK, train_end)
X_val, y_val, _ = make_sequences(train_end, val_end)
X_test, y_test, test_indices = make_sequences(val_end, N)

model = Sequential([
    tf.keras.Input(shape=(LOOKBACK, 5)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(5)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    shuffle=False,
    callbacks=[early_stop],
    verbose=1
)

plt.figure(figsize=(9, 4))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("LSTM Training History")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

pred_scaled = model.predict(X_test, verbose=0)
actual = scaler.inverse_transform(y_test)
predicted = scaler.inverse_transform(pred_scaled)
predicted = np.maximum(predicted, 0)

results = []
for j, article in enumerate(traffic.columns):
    results.append({
        "Article": article,
        "MAE": mean_absolute_error(actual[:, j], predicted[:, j]),
        "RMSE": np.sqrt(mean_squared_error(actual[:, j], predicted[:, j]))
    })

metrics_df = pd.DataFrame(results)
display(metrics_df.round(2))

print("Overall MAE:", round(
    mean_absolute_error(actual.flatten(), predicted.flatten()), 2
))
print("Overall RMSE:", round(
    np.sqrt(mean_squared_error(actual.flatten(), predicted.flatten())), 2
))

test_dates = traffic.index[test_indices]
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

for j, ax in enumerate(axes):
    ax.plot(test_dates, actual[:, j], label="Actual Traffic")
    ax.plot(test_dates, predicted[:, j], label="Predicted Traffic")
    ax.set_title(str(traffic.columns[j])[:65])
    ax.set_ylabel("Page Views")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.xlabel("Date")
plt.tight_layout()
plt.show()

FORECAST_DAYS = 7
current_sequence = scaled[-LOOKBACK:].copy()
future_scaled = []

for _ in range(FORECAST_DAYS):
    next_prediction = model.predict(
        current_sequence[np.newaxis, :, :], verbose=0
    )[0]
    future_scaled.append(next_prediction)
    current_sequence = np.concatenate([
        current_sequence[1:],
        next_prediction.reshape(1, -1)
    ], axis=0)

future_values = scaler.inverse_transform(np.asarray(future_scaled))
future_values = np.maximum(future_values, 0)

future_dates = pd.date_range(
    start=traffic.index[-1] + pd.Timedelta(days=1),
    periods=FORECAST_DAYS,
    freq="D"
)

future_df = pd.DataFrame(
    np.rint(future_values).astype(int),
    index=future_dates,
    columns=traffic.columns
)
future_df.index.name = "Date"
display(future_df)

for article in traffic.columns:
    plt.figure(figsize=(12, 4))
    plt.plot(
        traffic.index[-60:],
        traffic[article].iloc[-60:],
        label="Historical Traffic"
    )
    plt.plot(
        future_df.index,
        future_df[article],
        marker="o",
        linestyle="--",
        label="Forecast Traffic"
    )
    plt.axvline(
        traffic.index[-1],
        linestyle=":",
        label="Forecast Start"
    )
    plt.title(f"Seven-Day Forecast: {article}")
    plt.xlabel("Date")
    plt.ylabel("Page Views")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

metrics_df.to_csv("lstm_evaluation_metrics.csv", index=False)
future_df.to_csv("future_traffic_forecast.csv")
model.save("web_traffic_lstm.keras")

print("Saved: lstm_evaluation_metrics.csv")
print("Saved: future_traffic_forecast.csv")
print("Saved: web_traffic_lstm.keras")
