from metastable.state import FixedPointMap
import matplotlib

matplotlib.use("TkAgg")  # Example for Tkinter based interactive backend
import matplotlib.pyplot as plt

import numpy as np

# Set the default font size
plt.rcParams.update({"font.size": 14})

state = FixedPointMap.load("./map.npz")

# Prepare 3D figure and axis
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

names = ["Saddle Point", "Bright State", "Dim State"]

# Setup a meshgrid for the axes
kappa, epsilon = np.meshgrid(state.kappa_linspace, state.epsilon_linspace)

# Define the z-levels for each plot to avoid overlap
z_levels = [0, 0, 0]  # Adjust these values to manage the vertical spacing

for idx, z in enumerate(z_levels):
    norm_fixed_points = np.linalg.norm(state.fixed_points[:, :, idx], axis=2)

    # Plot each colormap as a surface in the 3D space
    surf = ax.plot_surface(
        kappa,
        epsilon,
        norm_fixed_points,  # Offset each surface by z to separate them in the 3D space
        cmap="viridis",
        edgecolor="none",
    )
    ax.text2D(0.05, 0.95 - idx * 0.05, f"{names[idx]}", transform=ax.transAxes)

ax.set_xlabel(r"$\kappa/\delta$")
ax.set_ylabel(r"$\epsilon/\delta$")
ax.set_zlabel("Norm of Fixed Points")

# Colorbar setup
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
cbar.set_label("Norm of Fixed Points")

plt.savefig("3d_fig.png")
plt.show()
