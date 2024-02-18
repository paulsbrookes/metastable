import numpy as np
from scipy.signal import convolve2d

# Example boolean array
bool_array = np.array([[True, False, True], [False, False, True], [True, True, False]])

# Step 1: Convert to integer array
int_array = bool_array.astype(int)

# Step 2: Define the convolution kernel
kernel = np.ones((3, 3), dtype=int)

# Step 3: Apply convolution
# Use 'same' mode to keep the output size the same as the input size
conv_result = convolve2d(int_array, kernel, mode="same", boundary="fill", fillvalue=0)

# Step 4: Identify False cells with True neighbors
# A cell is a False cell with True neighbors if it was 0 in the original array
# but has a positive value in the convolved result
# We also ensure we exclude the True cells themselves from the result
false_cells_with_true_neighbors = (conv_result > 0) & (int_array == 0)

# Step 5: Extract indices
indices = np.where(false_cells_with_true_neighbors)

# Print the indices
print("Indices of False cells next to True cells:", list(zip(indices[0], indices[1])))
