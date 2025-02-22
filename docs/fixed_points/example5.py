import numpy as np
import matplotlib.pyplot as plt
from metastable.map.map import FixedPointMap

map = FixedPointMap.load(
    "/home/paul/Projects/keldysh/metastable/docs/fixed_points/map-601x401.npz"
)
l2_norms = np.linalg.norm(map.fixed_points, axis=3)

# Create two subplots: one for κ cuts, one for ε cuts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot cuts at constant κ values
kappa_indices = [10, 20, 30]  # Choose some interesting κ values
for k_idx in kappa_indices:
    k_value = map.kappa_linspace[k_idx]
    for fp_idx in range(3):
        ax1.plot(
            map.epsilon_linspace,
            l2_norms[:, k_idx, fp_idx],
            label=f"FP{fp_idx+1}, κ={k_value:.2f}",
        )
ax1.set_xlabel("ε")
ax1.set_ylabel("L2 norm")
ax1.set_title("L2 norms at constant κ")
ax1.legend()

# Plot cuts at constant ε values
epsilon_indices = [15, 30, 45]  # Choose some interesting ε values
for e_idx in epsilon_indices:
    e_value = map.epsilon_linspace[e_idx]
    for fp_idx in range(3):
        ax2.plot(
            map.kappa_linspace,
            l2_norms[e_idx, :, fp_idx],
            label=f"FP{fp_idx+1}, ε={e_value:.2f}",
        )
ax2.set_xlabel("κ")
ax2.set_ylabel("L2 norm")
ax2.set_title("L2 norms at constant ε")
ax2.legend()

plt.tight_layout()
plt.savefig("fixed_points_cuts.png")
