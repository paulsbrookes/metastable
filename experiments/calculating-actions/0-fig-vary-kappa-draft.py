import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pyarrow as pa

# Read data from Parquet files - specify engine="pyarrow" to avoid the fastparquet issue
actions_bright_to_saddle = pd.read_parquet("actions_bright_to_saddle.parquet", engine="pyarrow")
actions_bright_to_saddle.columns = ["Bright to Saddle"]
actions_dim_to_saddle = pd.read_parquet("actions_dim_to_saddle.parquet", engine="pyarrow")
actions_dim_to_saddle.columns = ["Dim to Saddle"]

dykman_actions_bright_to_saddle = pd.read_parquet(
    "dykman_actions_bright_to_saddle.parquet", engine="pyarrow"
)
dykman_actions_bright_to_saddle.columns = ["Dykman: Bright to Saddle"]

dykman_actions_dim_to_saddle = pd.read_parquet("dykman_actions_dim_to_saddle.parquet", engine="pyarrow")
dykman_actions_dim_to_saddle.columns = ["Dykman: Dim to Saddle"]

# Create the plot
fig, axes = plt.subplots(1, 1, figsize=(10, 10))

# Plot each DataFrame and specify the label for the legend
actions_bright_to_saddle.plot(ax=axes, label="Bright to Saddle")
actions_dim_to_saddle.plot(ax=axes, label="Dim to Saddle")

dykman_actions_bright_to_saddle.plot(ax=axes, label="Dykman: Bright to Saddle")
dykman_actions_dim_to_saddle.plot(ax=axes, label="Dykman: Dim to Saddle")

fs = 20
# Set the x and y labels with larger font size
axes.set_xlabel("kappa", fontsize=fs)
axes.set_ylabel("action", fontsize=fs)

# Add the legend with larger font size
axes.legend(fontsize=fs)

# Show the plot
plt.savefig("plot.png")  # Save the plot as a PNG file
# plt.show()  # Comment this out if saving to a file
