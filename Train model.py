import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import csv
import os.path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # gets rid of tensorflow warnings
from tensorflow import keras
from keras.models import load_model
from itertools import zip_longest
from sklearn.preprocessing import MinMaxScaler
from yaml import load
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential

print("Enter the pair of currency you want to predict (BTC-INR): ")
crypto_currency, against_currency = input().split("-")

start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()
data = web.DataReader(f"{crypto_currency}-{against_currency}", "yahoo", start, end)

# prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

prediction_days = 60
future_day = 30

# Using pre-trained model or train new model
if os.path.exists(f"./models/{crypto_currency}-{against_currency}.h5"):
    model = load_model(f"./models/{crypto_currency}-{against_currency}.h5")
else:
    x_train, y_train = [], []

    for x in range(prediction_days, len(scaled_data) - future_day):
        x_train.append(scaled_data[x - prediction_days : x, 0])
        y_train.append(scaled_data[x + future_day, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, batch_size=32, epochs=25)
    model.save(f"./models/{crypto_currency}-{against_currency}.h5")

# Testing the model
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()
time_col = pd.date_range(test_start, test_end)
test_data = web.DataReader(
    f"{crypto_currency}-{against_currency}", "yahoo", test_start, test_end
)
actual_price = test_data["Close"].values
total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)
model_inputs = total_dataset[
    len(total_dataset) - len(test_data) - prediction_days :
].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days : x, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)
prediction_prices = prediction_prices.flatten()

# Creating the csv file with Date, Actual Price, Predicted Price
with open(f"./values/{crypto_currency}-{against_currency}.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["Date", "Actual Price", "Predicted Price"])
    for x, y, z in zip_longest(time_col, actual_price, prediction_prices):
        w.writerow([x, y, z])

# Plot the graph and save it in the destined folder
plt.plot(actual_price, color="blue", label="Actual Prices")
plt.plot(prediction_prices, color="green", label="Predicted Prices")
plt.title(f"{crypto_currency}-{against_currency} price prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.savefig(f"./graphs/{crypto_currency}-{against_currency}.png")

# Predict next day
real_data = [
    model_inputs[len(model_inputs) + 1 - prediction_days : len(model_inputs) + 1, 0]
]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
prediction = prediction.flatten()
print(prediction[0])
