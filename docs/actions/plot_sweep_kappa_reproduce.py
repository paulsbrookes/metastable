import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from metastable.map.map import FixedPointMap
from metastable.rescaled import (
    calculate_kappa_rescaled,
    calculate_beta_limits,
    map_beta_to_epsilon,
)
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from metastable.rescaled.barriers import dykman_actions_calc
from metastable.paths import get_bistable_kappa_range


# Set global parameters
fs = 20  # Font size for all text elements
lw = 3  # Linewidth for all plots
x_max = 0.6  # Upper limit for x-axis
delta = 7.8  # Scaling factor

plt.rcParams.update({"font.size": fs})

# Create figure with two rows and one column, sharing x-axis
fig, (ax0, ax1) = plt.subplots(
    2, 1, figsize=(17, 10), height_ratios=[1, 1], sharex=True
)
fig.subplots_adjust(hspace=0.05)  # Reduce space between subplots

# Upper row: Epsilon limits plot
map_path = "/home/paul/Projects/misc/keldysh/metastable/docs/paths/examples/output/9/output_map_with_actions.npz"
fixed_point_map = FixedPointMap.load(map_path)

kappa_rescaled_linspace = calculate_kappa_rescaled(
    fixed_point_map.kappa_linspace, delta=fixed_point_map.delta
)
beta_limits_array = calculate_beta_limits(kappa_rescaled_linspace)
epsilon_limits_array = map_beta_to_epsilon(
    beta_limits_array, chi=fixed_point_map.chi, delta=fixed_point_map.delta
)

# Lower row: Custom graph data
epsilon_idx = 380
# this is a dataframe with an index containing kappa
actions_bright_to_saddle = pd.DataFrame(
    fixed_point_map.path_actions[epsilon_idx, :, 0, 0],
    index=fixed_point_map.kappa_linspace,
    columns=[r"$R_{b \to u}$"]
)

actions_dim_to_saddle = pd.DataFrame(
    fixed_point_map.path_actions[epsilon_idx, :, 1, 0],
    index=fixed_point_map.kappa_linspace,
    columns=[r"$R_{d \to u}$"]
)

# Get epsilon value from the fixed point map
epsilon = fixed_point_map.epsilon_linspace[epsilon_idx]


kappa_boundaries = get_bistable_kappa_range(fixed_point_map.bistable_region, epsilon_idx)
kappa_min = kappa_boundaries.dim_saddle.kappa_idx
kappa_max = kappa_boundaries.bright_saddle.kappa_idx


# Calculate Dykman actions - function returns both bright_to_saddle and dim_to_saddle values
dykman_bright_to_saddle_values, dykman_dim_to_saddle_values = dykman_actions_calc(
    delta=fixed_point_map.delta,
    chi=fixed_point_map.chi,
    eps=epsilon,
    kappa=fixed_point_map.kappa_linspace[kappa_min:kappa_max],
)

# Create DataFrames with calculated values
dykman_actions_bright_to_saddle = pd.DataFrame(
    dykman_bright_to_saddle_values,
    index=fixed_point_map.kappa_linspace[kappa_min:kappa_max],
    columns=[r"$R_{b \to u}$"]
)

dykman_actions_dim_to_saddle = pd.DataFrame(
    dykman_dim_to_saddle_values,
    index=fixed_point_map.kappa_linspace[kappa_min:kappa_max],
    columns=[r"$R_{d \to u}$"]
)

# Rescale the index (x-axis) for all DataFrames
for df in [
    actions_bright_to_saddle,
    actions_dim_to_saddle,
    dykman_actions_bright_to_saddle,
    dykman_actions_dim_to_saddle,
]:
    df.index = df.index / delta

# Get the upper limit from dykman_actions_bright_to_saddle and lower limit from dykman_actions_dim_to_saddle
lower_epsilon = dykman_actions_dim_to_saddle.index.min()
upper_epsilon = dykman_actions_bright_to_saddle.index.max()

# Add light grey shading to upper panel
ax0.axvspan(lower_epsilon, upper_epsilon, color="lightgrey", alpha=0.5)

# Plot upper panel data
ax0.plot(
    fixed_point_map.kappa_linspace / delta,
    epsilon_limits_array[0] / delta,
    color="red",
    label="Unstable-Bright",
    linewidth=lw,
)
ax0.plot(
    fixed_point_map.kappa_linspace / delta,
    epsilon_limits_array[1] / delta,
    color="blue",
    label="Unstable-Dim",
    linewidth=lw,
)

# Add dashed black line at epsilon = 19.0 / delta
ax0.axhline(y=19.0 / delta, color="black", linestyle="--", linewidth=lw)

ax0.set_ylabel(r"$\epsilon / \delta$", fontsize=fs)
legend0 = ax0.legend(fontsize=fs, title="Bifurcation Type")
legend0.get_title().set_fontsize(fs)
legend0.get_title().set_fontweight("bold")  # Make the title bold
ax0.set_ylim([0.0, 3.5])
ax0.tick_params(labelsize=fs)
ax0.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
ax0.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

# Plot lower panel data
ax1.plot(
    actions_bright_to_saddle.index,
    -actions_bright_to_saddle.values,
    linewidth=lw,
    color="red",
)
ax1.plot(
    actions_dim_to_saddle.index,
    -actions_dim_to_saddle.values,
    linewidth=lw,
    color="blue",
)
ax1.plot(
    dykman_actions_bright_to_saddle.index,
    -dykman_actions_bright_to_saddle.values,
    linewidth=lw,
    color="purple",
    ls="-.",
)
ax1.plot(
    dykman_actions_dim_to_saddle.index,
    -dykman_actions_dim_to_saddle.values,
    linewidth=lw,
    color="green",
    ls="-.",
)

# Add light grey shading to lower panel
ax1.axvspan(lower_epsilon, upper_epsilon, color="lightgrey", alpha=0.5)

ax1.set_xlabel(r"$\kappa / \delta$", fontsize=fs)
ax1.set_ylabel(r"$R_{j \to u} / \lambda$", fontsize=fs)
ax1.set_xlim(0, x_max)
ax1.set_ylim([0, 8.0])

# Create legend handles
legend_elements = [
    Line2D([0], [0], color="red", lw=lw, label=r"$R_{b \to u}$"),
    Line2D([0], [0], color="blue", lw=lw, label=r"$R_{d \to u}$"),
    Line2D([0], [0], color="purple", lw=lw, ls="-.", label=r"$R_{b \to u}$"),
    Line2D([0], [0], color="green", lw=lw, ls="-.", label=r"$R_{d \to u}$"),
]

# Create a single legend with two sections
legend = ax1.legend(
    handles=legend_elements,
    loc="lower right",
    fontsize=fs,
    title="    Keldysh      Kramers    ",
    ncol=2,
    columnspacing=1.5,
    handlelength=1.5,
    handletextpad=0.5,
)

# Adjust legend title
legend.get_title().set_fontsize(fs)
legend.get_title().set_fontweight("bold")

ax1.tick_params(labelsize=fs)
ax1.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

# Save the figure
plt.tight_layout()
plt.savefig("two_row_figure.png", bbox_inches="tight", dpi=200)

