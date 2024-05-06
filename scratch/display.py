from metastable.state import FixedPointMap
import matplotlib.pyplot as plt
import numpy as np

# Set the default font size
plt.rcParams.update({"font.size": 14})

state = FixedPointMap.load("/home/paul/Projects/keldysh/metastable/scratch/map.npz")

# Create a new figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(6, 18), sharex=True)


names = ["Saddle Point", "Bright State", "Dim State"]

plt.subplots_adjust(right=0.8)

for idx, ax in enumerate(axes):
    norm_fixed_points = np.linalg.norm(state.fixed_points[:, :, idx], axis=2)

    # Plot the heatmap
    heatmap = ax.imshow(
        norm_fixed_points,
        cmap="viridis",
        aspect="auto",
        origin="lower",
        extent=[
            state.kappa_linspace[0],
            state.kappa_linspace[-1],
            state.epsilon_linspace[0],
            state.epsilon_linspace[-1],
        ],
    )

    # Add y-axis label
    ax.set_ylabel(r"$\epsilon/\delta$")

    # Set the title of the subplot
    ax.set_title(f"{names[idx]}")

# Add x-axis label to the bottom subplot
axes[-1].set_xlabel(r"$\kappa/\delta$")

# Create a new axis for the colorbar
cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])


# Add a single colorbar for all subplots
cbar = fig.colorbar(heatmap, cax=cbar_ax)
cbar.set_label("Norm of Fixed Points")

plt.savefig("fig.png")
# Display the plot
plt.show()
