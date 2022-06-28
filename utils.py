import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def split(df, label, fraction=0.8, format="normal" or "tf" or "both"):
    split_size = int(len(df) * fraction)
    X = df.drop(columns=[label], axis=1)
    y = df[label]

    X_train, y_train = X[:split_size], y[:split_size]
    X_valid, y_valid = X[split_size:], y[split_size:]

    if format == "normal":
        return (X_train, y_train), (X_valid, y_valid)
    elif format == "tf":
        return tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)).batch(128).prefetch(tf.data.AUTOTUNE), tf.data.Dataset.from_tensor_slices(
            (X_valid, y_valid)).batch(128).prefetch(tf.data.AUTOTUNE)
    else:
        return ((X_train, y_train), (X_valid, y_valid)), (tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)).batch(128).prefetch(tf.data.AUTOTUNE), tf.data.Dataset.from_tensor_slices(
            (X_valid, y_valid)).batch(128).prefetch(tf.data.AUTOTUNE))


def make_windowed_dataset(dataframe: pd.DataFrame, window: int, features: list) -> pd.DataFrame:
    df = dataframe.copy()
    for f in features:
        for w in range(window):
            df[f"{f}+{w+1}"] = df[f"{f}"].shift(periods=w+1)
    df.dropna(inplace=True)
    return df


def load_timeseries_data(window: int, features: list, drop: list, label: str = "", how="all" or "split" or "dict", fraction=0.8, root="stock") -> pd.DataFrame or dict:
    en = LabelEncoder()
    data = pd.DataFrame()
    for dir in os.listdir(root):
        df = pd.read_csv(os.path.join(root, dir), parse_dates=['Date'])
        window_df = make_windowed_dataset(df, window, features)
        data = data.append(window_df, ignore_index=True)
    data['Stock'] = en.fit_transform(data['Stock'])

    if how == "all":
        data.drop(drop, axis=1, inplace=True)
        return data

    elif how == "split":
        data = {}
        for dir in os.listdir(root):
            df = pd.read_csv(os.path.join(root, dir))
            window_df = make_windowed_dataset(df, window, features)
            data[dir.split(".")[0]] = window_df.reset_index(drop=True)

        X_train_df = pd.DataFrame()
        y_train_df = pd.DataFrame()
        X_valid_df = pd.DataFrame()
        y_valid_df = pd.DataFrame()

        for s in data:
            split_size = int(len(data[s]) * fraction)
            data[s].drop(drop, axis=1, inplace=True)
            X = data[s].drop(columns=[label], axis=1)
            y = pd.DataFrame(data[s][label])
            X_train_df = X_train_df.append(X[:split_size])
            y_train_df = y_train_df.append(y[:split_size])
            X_valid_df = X_valid_df.append(X[split_size:])
            y_valid_df = y_valid_df.append(y[split_size:])

        X_train_df.reset_index(drop=True, inplace=True)
        X_train_df['Stock'] = en.transform(X_train_df['Stock'])
        y_train_df.reset_index(drop=True, inplace=True)
        X_valid_df.reset_index(drop=True, inplace=True)
        X_valid_df['Stock'] = en.transform(X_valid_df['Stock'])
        y_valid_df.reset_index(drop=True, inplace=True)

        return (X_train_df, y_train_df, X_valid_df, y_valid_df)

    else:
        data = {}
        for dir in os.listdir(root):
            df = pd.read_csv(os.path.join(root, dir))
            window_df = make_windowed_dataset(df, window, features)
            window_df['Date'] = pd.to_datetime(window_df['Date'])
            window_df.drop(columns=drop, axis=1, inplace=True)
            window_df['Stock'] = en.transform(window_df['Stock'])
            data[dir.split(".")[0]] = window_df.reset_index(drop=True)
        return data


def get_timesteps(window, timeseries_features, into_future):
    TIMESTEPS = {}
    stock = load_timeseries_data(
        window=window, features=timeseries_features, drop=[], how="dict")
    for s in stock:
        date = get_future_dates(
            stock[s]['Date'].iloc[-1], into_future)
        TIMESTEPS[s] = date
    return TIMESTEPS


def get_future_dates(start_date, into_future, offset=1):
    """
    Returns array of datetime values ranging from start_date to start_date+into_future
    """
    start_date = start_date + \
        np.timedelta64(offset, "D")  # specify start date, "D" stands for date
    end_date = start_date + np.timedelta64(into_future, "D")

    return np.arange(start_date, end_date, dtype="datetime64[D]")


def evaluate(y_true, y_pred):

    def mean_absolute_scaled_error(y_true, y_pred):
        """
        Implement MASE (assuming no seasonality of data).
        """
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))

        # Find MAE of naive forecast (no seasonality)
        # our seasonality is 1 day (hence the shift of 1)
        mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))
        return mae / mae_naive_no_season

    # Make sure float32 datatype (for metric calculation)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Calculate various evaluation metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)

    return {
        "mae": mae.numpy(),
        "mse": mse.numpy(),
        "rmse": rmse.numpy(),
        "mape": mape.numpy(),
        "mase": mase.numpy()
    }


def make_future_forecasts(model, into_future, window_size, drop, timeseries_features):
    """
    Make future forecasts into_future steps after values ends.

    Returns future forecasts as a list of floats.
    """
    # 2. Create an empty list for future forecasts/prepare data to forecast on
    future_forecast = {}
    stock = load_timeseries_data(
        window=window_size, features=timeseries_features, drop=drop + ["Close"], how="dict")

    for s in stock:
        last_window = stock[s].iloc[-1].to_list()
        stock_type = last_window.pop(0)
        future_forecast[s] = []

        for _ in range(into_future):
            pred = [stock_type] + last_window
            future_pred = model.predict(tf.expand_dims(pred, axis=0))
            future_pred = tf.squeeze(future_pred).numpy()
            # print(
            #     f"Predicting on:\n {last_window} -> Prediction: {future_pred}\n")

            future_forecast[s].append(future_pred)

            last_window.append(future_pred)
            last_window = last_window[-window_size:]

    return future_forecast


def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
    """
    Plots a timesteps (a series of points in time) against values (a series of values across timesteps).

    Parameters
    ---------
    timesteps : array of timesteps
    values : array of values across time
    format : style of plot, default "."
    start : where to start the plot (setting a value will index from start of timesteps & values)
    end : where to end the plot (setting a value will index from end of timesteps & values)
    label : label to show on plot of values
    """
    # Plot the series
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Price")
    if label:
        plt.legend(fontsize=14)  # make label bigger
    plt.grid(True)
