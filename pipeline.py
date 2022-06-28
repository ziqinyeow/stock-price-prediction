import os
import pandas as pd
import numpy as np
import tensorflow as tf
import utils
import warnings
warnings.filterwarnings("ignore")


class Pipeline:
    def __init__(self, window=7, into_future=90, timeseries_feature=["Close"], drop=["Date", "Open", "High", "Low", "Volume"]):
        self.window = window
        self.timeseries_feature = timeseries_feature
        self.into_future = 90
        self.etl()  # -> columns,
        self.other_feature = list(
            set(self.columns) - set(timeseries_feature + drop))
        self.feature_size = window * \
            len(timeseries_feature) + len(self.other_feature)
        self.timesteps = utils.get_timesteps(
            window, timeseries_feature, into_future)
        self.df = utils.load_timeseries_data(
            window=window, features=timeseries_feature, drop=drop)
        self.dict = utils.load_timeseries_data(
            window=window, features=timeseries_feature, drop=[], how="dict")
        self.X_train, self.y_train, self.X_valid, self.y_valid = utils.load_timeseries_data(
            window=window, features=timeseries_feature, drop=drop, how="split", label="Close")
        self.train_dataset, self.valid_dataset = (tf.data.Dataset.from_tensor_slices(
            (self.X_train, self.y_train)).batch(128).prefetch(tf.data.AUTOTUNE), tf.data.Dataset.from_tensor_slices(
            (self.X_valid, self.y_valid)).batch(128).prefetch(tf.data.AUTOTUNE))
        self.normalizer = tf.keras.layers.Normalization()
        self.normalizer.adapt(self.X_train)

    def etl(self):
        stock = pd.read_csv('Capital Dynamics - Dataset.csv',
                            nrows=0).columns.tolist()
        stock = [x for x in stock if "Unnamed" not in x]
        df = pd.read_csv("Capital Dynamics - Dataset.csv",
                         header=1, low_memory=False)
        nan_columns = df.columns[df.isna().all()].tolist()
        for i, col in enumerate(nan_columns):
            df[col] = stock[i]
            df.rename(columns={col: f"Stock.{i}"}, inplace=True)

        stock_df = {}
        stock_counter = 0
        stock_col = 7

        for i, s in enumerate(stock):
            stock_df[s] = df[df.columns[stock_counter:stock_counter + stock_col]]
            stock_counter += stock_col

            # rename the column name
            stock_df[s].columns = stock_df[s].columns.str.replace(f'.{i}', '')

            # remove all nan row
            stock_df[s].dropna(subset=['Date', 'Open', 'High',
                               'Low', 'Close', 'Volume'], how="all", inplace=True)
            stock_df[s].reset_index(drop=True, inplace=True)

            # change the date to datetime format
            stock_df[s]['Date'] = pd.to_datetime(stock_df[s]['Date'])

            # sort the date
            stock_df[s].sort_values(by=['Date'], inplace=True)

        stock_df['WELLCAL']['Stock'] = "WELLCAT"

        df: pd.DataFrame = pd.DataFrame()

        for i, s in enumerate(stock_df):
            df = df.append(stock_df[s], ignore_index=True)

        self.columns = df.columns

        df.to_csv("main.csv", index=False)

        if not os.path.exists("./stock"):
            os.mkdir("./stock")

        for s in stock_df:
            stock_df[s].to_csv(f"./stock/{s}.csv", index=False)

        close = df.drop(columns=['Open', 'High', 'Low'])
        close.to_csv("close.csv", index=False)

    def train(self, model, epochs=5):
        tf.random.set_seed(42)
        model.compile(
            loss="mae",
            optimizer=tf.keras.optimizers.Adam()
        )
        history = model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.valid_dataset
        )
        self.model = model
        return history

    def predict(self, stock_type, stock_price: list):
        stock_price.reverse()
        pred = self.model.predict([stock_type] + stock_price)
        pred = tf.squeeze(pred)
        return pred

    def save_model(self):
        self.model.save_model("model.h5")
        pass
