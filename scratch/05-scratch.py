import numpy as np
from scipy.signal import convolve2d

# Suppose function_values is your 2D numpy array with np.nan for unknown values
function_values = np.random.randn(10, 10)
function_values[0:3, 0:3] = np.nan

# Create a mask of valid (non-NaN) values
valid_mask = ~np.isnan(function_values)

# Create a kernel for convolution
kernel = np.ones((3, 3), dtype=int)

# Count valid neighbors for each cell
valid_neighbor_count = convolve2d(
    valid_mask, kernel, mode="same", boundary="fill", fillvalue=0
)

# Replace NaNs with 0 for convolution calculation
temp_values = np.nan_to_num(function_values)

# Sum of neighbor values
sum_neighbors = convolve2d(
    temp_values, kernel, mode="same", boundary="fill", fillvalue=0
)

# Calculate the average, avoiding division by zero
average_neighbors = np.divide(
    sum_neighbors,
    valid_neighbor_count,
    out=np.zeros_like(sum_neighbors),
    where=valid_neighbor_count != 0,
)

# Restore NaN where the original value was NaN and there are no valid neighbors to average
# average_neighbors[(valid_neighbor_count == 0) | np.isnan(function_values)] = np.nan

...
