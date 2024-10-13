# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import kagglehub

# Download latest version
path = kagglehub.dataset_download("robervalt/sunspots")

print("Path to dataset files:", path)

# create dataframe
csv_file_path = f"{path}/Sunspots.csv"
df = pd.read_csv(csv_file_path)

df.shape

df.head()

# define time and series veriable according to dataframe
time = pd.to_datetime(df['Date'])
series = df['Monthly Mean Total Sunspot Number']

# check it
print(time.head())
print(series.head())

# turn into numpy array
time.to_numpy()
series.to_numpy()

# Draw the garph
plt.figure(figsize=(16, 6))
plt.plot(time, series)
plt.title("Sunspots Over Time ")
plt.xlabel("Time")
plt.ylabel("Sunspot Count")
plt.grid(True)
plt.show()


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


"""# MODEL WITH LSTM AND CNN"""

# Define the split time
split_time = 3000

# Get the train set
time_train = time[:split_time]
x_train = series[:split_time]

# Get the validation set
time_valid = time[split_time:]
x_valid = series[split_time:]

"""# tune window_size and other parameters"""
# Parameters
window_size = 60
batch_size = 32
shuffle_buffer_size = 1000

train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

