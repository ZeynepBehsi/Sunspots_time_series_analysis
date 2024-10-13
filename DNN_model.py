# -*- coding: utf-8 -*-

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

"""# tune window_size and other parameters"""

# Parameters
window_size = 30
batch_size = 32
shuffle_buffer_size = 1000

train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

"""#  BUILD MODEL WITH NEURAL NETWORKS (DNN)

#  Model for Laerning Rate Optimization
"""

# Build model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(window_size,)),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(15, activation="relu"),
    tf.keras.layers.Dense(1, activation = "relu")
])

# save initial weights
init_weights = model.get_weights()

# add callbacks for optimization LR
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

# Compile model
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=tf.keras.optimizers.SGD(momentum=0.9))

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

"""# 🧠 Optimum Learning Rate = 2e-5

# Build new model with optimizized learning rate
"""

# Reset weights and veriables:

# reset veriables
tf.keras.backend.clear_session()

# reset weights
model.set_weights(init_weights)

# Build optimized model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(window_size,)),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(15, activation="relu"),
    tf.keras.layers.Dense(1, activation = "relu")
    ])

# Compile model
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=tf.keras.optimizers.SGD(learning_rate=2e-5, momentum=0.9),
              metrics=["mae"])

# Train the new model
history = model.fit(train_set, epochs=100)

# performans metrics
mae = history.history['mae']
loss = history.history['loss']

# define epochs
epochs = range(len(loss))

# Draw the graph
plt.figure(figsize=(10, 6))
plt.plot(epochs, mae, label='Training MAE')
plt.plot(epochs, loss, label='Training Loss')
plt.title('Training and Validation MAE and Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

"""# Prediction"""

# the function to draw graphs easily
def plot_series(x, y, format="-", start=0, end=None,
                title=None, xlabel=None, ylabel=None, legend=None ):

    plt.figure(figsize=(10, 6))

    if type(y) is tuple:
      for y_curr in y:
        plt.plot(x[start:end], y_curr[start:end], format)
    else:
      plt.plot(x[start:end], y[start:end], format)


    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
      plt.legend(legend)
    plt.title(title)
    plt.grid(True)
    plt.show()

# The function for prediction
def model_forecast(model, series, window_size, batch_size):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(1)
    forecast = model.predict(dataset, verbose=0)

    return forecast

# Reduce the original series
forecast_series = series[split_time-window_size:-1]
forecast = model_forecast(model, forecast_series, window_size, batch_size)

results = forecast.squeeze()

# Plot the results
plot_series(time_valid, (x_valid, results))

# Compute the MAE
print(tf.keras.metrics.mae(x_valid, results).numpy())


