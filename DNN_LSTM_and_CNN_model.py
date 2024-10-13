# -*- coding: utf-8 -*-"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras



"""# Function for Windowing dataset"""

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)

    # Optimize the dataset for training
    dataset = dataset.cache().prefetch(1)

    return dataset


"""# MODEL: DNN, LSTM AND CNN"""


# Parameters
window_size = 60
batch_size = 250
shuffle_buffer_size = 1000

# Generate the dataset windows
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

"""# Learning Rate Optimization

# + Optimization:
- I have 3000 data point in train set. I choose numbers for the numbers of neuron and filter, batch_size, window_size which are exactly divisible by this number
"""

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(window_size,1)),
    #CNN
    tf.keras.layers.Conv1D(filters=60, kernel_size=3,
                      strides=1, padding="causal",
                      activation="relu"),
    #LSTM
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(60),

    #DNN
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),

    #Output Layer
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400)
])

# Print the model summary
model.summary()

# Add callbacks for optimization Learning Rate
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.SGD(momentum=0.9)

# Compile the model. Choos Huber as a loss function which is less sensitive outliers
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics = ["mae"])

# Train the model
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])

# choose defauld learning rate and see changes in the graph
lrs = 1e-8 * (10 ** (np.arange(100) / 20))

# graph
plt.figure(figsize=(10, 6))
plt.grid(True)
plt.semilogx(lrs, history.history["loss"])
plt.tick_params('both', length=10, width=1, which='both')
plt.axis([1e-8, 1e-3, 0, 100])

"""# Chosee optimum leraning rate according to graph : 2e-6

# Build Model (DNN-LSTM-CNN) with Optimized LR
"""

# Reset veriables
tf.keras.backend.clear_session()

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(window_size,1)),
    #CNN
    tf.keras.layers.Conv1D(filters=60, kernel_size=3,
                      strides=1, padding="causal",
                      activation="relu"),
    #LSTM
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(60),

    #DNN
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),

    #Output Layer
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400)
])

# Print the model summary
model.summary()

learning_rate = 2e-6

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

# Train the model
history = model.fit(train_set,epochs=100)

# Get mae and loss from history log
mae=history.history['mae']
loss=history.history['loss']

# Get number of epochs
epochs=range(len(loss))

# Plot mae and loss
plot_series(
    x=epochs,
    y=(mae, loss),
    title='MAE and Loss',
    xlabel='epoch',
    ylabel='Loss',
    legend=['MAE', 'Loss']
    )

# Only plot the last 80% of the epochs
zoom_split = int(epochs[-1] * 0.2)
epochs_zoom = epochs[zoom_split:]
mae_zoom = mae[zoom_split:]
loss_zoom = loss[zoom_split:]

# Plot zoomed mae and loss
plot_series(
    x=epochs_zoom,
    y=(mae_zoom, loss_zoom),
    title='MAE and Loss',
    xlabel='epoch',
    ylabel='Loss',
    legend=['MAE', 'Loss']
    )

forecast_series = series[split_time-window_size:-1]
forecast = model_forecast(model, forecast_series, window_size, batch_size)

results = forecast.squeeze()

# Plot the results
plot_series(time_valid, (x_valid, results))

# Compute the MAE
print(tf.keras.metrics.mae(x_valid, results).numpy())

