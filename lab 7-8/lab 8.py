import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.stats import f_oneway

# Load the data from the file
data = np.loadtxt('-0.35V_sp567.dat')

# Select the first column for analysis
data_column = data[:, 1]  # Change the column index as needed

# Step 1: Select a random section of 1024 data points
np.random.seed(0)  # For reproducibility
random_start = np.random.randint(0, len(data_column) - 1023)
selected_data = data_column[random_start:random_start + 1024]

# Step 2: Plot the original data
plt.figure(figsize=(10, 6))
plt.plot(selected_data)
plt.title('Original Data')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()

# Step 3: Construct a histogram
plt.figure(figsize=(10, 6))
plt.hist(selected_data, bins=50)
plt.title('Histogram of Selected Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Identify regions using histogram
counts, bins = np.histogram(selected_data, bins=50)
background_value = bins[np.argmax(counts)]

# Step 4: Apply a median filter
filtered_data = medfilt(selected_data, kernel_size=5)

# Step 5: Plot the smoothed data
plt.figure(figsize=(10, 6))
plt.plot(filtered_data)
plt.title('Smoothed Data (Median Filter)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()

# Step 6: Identify and mark different regions
# Assuming regions based on simple thresholding for illustration
signal_threshold = background_value + (bins[1] - bins[0]) * 5  # Arbitrary threshold
transition_threshold = background_value + (bins[1] - bins[0]) * 2  # Arbitrary threshold

background_region = filtered_data < transition_threshold
signal_region = filtered_data > signal_threshold
transition_region = ~background_region & ~signal_region

plt.figure(figsize=(10, 6))
plt.plot(filtered_data, label='Filtered Data')
plt.plot(np.where(background_region)[0], filtered_data[background_region], 'go', label='Background')
plt.plot(np.where(signal_region)[0], filtered_data[signal_region], 'ro', label='Signal')
plt.plot(np.where(transition_region)[0], filtered_data[transition_region], 'bo', label='Transition')
plt.title('Identified Regions in Smoothed Data')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()

# Step 7: Apply Fisher's test to check homogeneity
background_data = filtered_data[background_region]
signal_data = filtered_data[signal_region]
transition_data = filtered_data[transition_region]

f_val, p_val = f_oneway(background_data, signal_data, transition_data)

# Presenting results
results_table = {
    'Region': ['Background', 'Signal', 'Transition'],
    'Mean Value': [np.mean(background_data), np.mean(signal_data), np.mean(transition_data)],
    'Fisher Value': [f_val] * 3,
    'p-value': [p_val] * 3
}

import pandas as pd
results_df = pd.DataFrame(results_table)

# Printing the table
print(results_df)