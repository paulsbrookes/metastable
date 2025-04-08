# 2. Switching Paths

In this section we can now turn our attention to finding the switching paths connecting the stable fixed points to the saddle point. We will use our prevoius stability analysis to set the boundary conditions for the switching paths. We will then use a collocation method implemented in SciPy [41] to solve the resulting boundary value problem and find the switching paths.

## 2.1. Problem Overview

### 2.1.1. Path

Our key objective is to find a path $\mathbf{Z}(t)$ that connects a **stable fixed point** $\mathbf{Z}_0$ (node or focus) to a **saddle point** $\mathbf{Z}_s$.

The key idea is to use the eigenvectors of the Jacobian at both points to set the boundary conditions. Small deviations from the fixed points can be expressed in the eigenvector basis of the Jacobian and we control the boundary conditions by specifying the coefficients of the eigenvectors in those deviations.

At each fixed point the Jacobian has a pair of incoming and outgoing eigenvectors. We simply apply the condition that at the start of the path the deviation is purely along the eigenvectors leaving the stable point and at the end of the path the deviation is purely along the eigenvectors arriving at the saddle point.

### 2.1.2. Action

With these paths we can finally calculate the corresponding actions using:

$$
\begin{aligned}
S_{\mathrm{aux}} = -\int dt\Bigl[\dot{x}_c\,x_q + \dot{p}_c\,p_q - H(x_c,p_c,x_q,p_q)\Bigr].
\end{aligned}
$$

Since the Hamiltonian is conservative (i.e. time independent) and the initial and final points are in the classical plane where $H=0$, the action can be simplified to:

$$
S_{\mathrm{aux}} = -\int dt\Bigl[\dot{x}_c\,x_q + \dot{p}_c\,p_q\Bigr].
$$


## 2.2. Mathematical Framework

### 2.2.1. At the Stable Fixed Point

Let $\mathbf{Z}_0$ denote the coordinates of the stable fixed point. Small deviations from $\mathbf{Z}_0$ can be expressed in the eigenvector basis of the Jacobian $J(\mathbf{Z}_0)$:
  
$$
\Delta \mathbf{Z}(t) \equiv \mathbf{Z}(t) - \mathbf{Z}_0 = \sum_{i=1}^{4} c_i\, \mathbf{v}_i\, e^{\lambda_i t},
$$
  
where:
  - $\mathbf{v}_i$ are the eigenvectors,
  - $\lambda_i$ are the corresponding eigenvalues, ordered by their real parts with $\text{Re}(\lambda_1), \text{Re}(\lambda_2) < 0$ (stable) and $\text{Re}(\lambda_3), \text{Re}(\lambda_4) > 0$ (unstable),
  - $c_i$ are coefficients determined by the initial displacement.

### 2.2.2. At the Saddle Point

- Let $\mathbf{Z}_s$ denote the coordinates of the saddle point.
- Near $\mathbf{Z}_s$, the deviation is similarly expressed in terms of the eigenvectors of the Jacobian $J(\mathbf{Z}_s)$:
  
$$
\Delta \mathbf{Z}(t) \equiv \mathbf{Z}(t) - \mathbf{Z}_s = \sum_{j=1}^{4} d_j\, \mathbf{u}_j\, e^{\mu_j t},
$$
  
where:
  - $\mathbf{u}_j$ are the eigenvectors at the saddle,
  - $\mu_j$ are the corresponding eigenvalues, ordered by their real parts with $\text{Re}(\mu_1), \text{Re}(\mu_2) < 0$ (stable) and $\text{Re}(\mu_3), \text{Re}(\mu_4) > 0$ (unstable),
  - $d_j$ are coefficients that characterize the deviation.

---

## 2.3. Formulating the Boundary Value Problem

### 2.3.1. Physical Interpretation of Boundary Conditions

The switching trajectory represents the optimal (least-action) path connecting a metastable fixed point to the saddle point. It is a solution to the equations of motion derived from the auxiliary Hamiltonian. Our stability analysis then provides natural boundary conditions:

**Stable Fixed Point $\mathbf{Z}_0$**

At $t \to -\infty$ (approaching the stable fixed point) we have:
  
$$
\mathbf{Z}(t) \to \mathbf{Z}_0 \quad \Longrightarrow \quad \Delta \mathbf{Z}(t) \approx \sum_{i} c_i\, \mathbf{v}_i\, e^{\lambda_i t},
$$

Since the trajectory must *depart* within the unstable subspace, the initial deviation $\Delta \mathbf{Z}(t)$ aligns with eigenvectors having $\text{Re}(\lambda_i) > 0$ and we can set $c_0 = 0$ and $c_1 = 0$.

Depending on exactly which initial direction we choose, the system will follow a different path. Our goal is to the path which connects to the saddle point.

**Saddle Point $\mathbf{Z}_s$**

At $t \to +\infty$ (approaching the saddle point):
  
$$
\mathbf{Z}(t) \to \mathbf{Z}_s \quad \Longrightarrow \quad \Delta \mathbf{Z}(t) \approx \sum_{j} d_j\, \mathbf{u}_j\, e^{\mu_j t},
$$

Since the trajectory must *arrive* within the stable subspace, the final deviation $\Delta \mathbf{Z}(t)$ lies in the subspace of eigenvectors with $\text{Re}(\mu_j) < 0$. Therefore we can set $d_2 = 0$ and $d_3 = 0$. If we have chosen the correct initial trajectory, the system will follow a path that naturally arrives at the saddle point within this subspace.

### 2.3.2 Projection

To enforce our asymptotic boundary conditions at finite times $t_i$ and $t_f$, we must project the deviations onto the eigenbasis of the Jacobian at the fixed points. This is achieved by using the left eigenvectors to extract the coefficients in the expansion of the deviation in terms of the right eigenvectors.

Let $\Delta Z(t)$ be the deviation from a fixed point (either $Z_0$ or $Z_s$). Suppose the right eigenvectors are arranged in a matrix $R$ and the corresponding left eigenvectors (normalized such that $L_i \cdot R_j = \delta_{ij}$) are arranged in a matrix $L$. Then the expansion

$$\Delta Z(t) = \sum_i c_i v_i \iff \Delta Z(t) = R c$$

allows us to extract the coefficients by projecting onto the left eigenvectors:

$$c = L \Delta Z(t).$$

At the stable fixed point, for instance, we only want deviations along the unstable directions. This requirement translates into setting the coefficients corresponding to the stable modes to zero:

$$c_i = L_i \cdot [Z(t) - Z_0] = 0 \text{ for stable modes.}$$

Similarly, at the saddle point, to ensure the trajectory approaches along the stable manifold, the coefficients for the unstable directions must vanish:

$$d_j = L_j \cdot [Z(t) - Z_s] = 0 \text{ for unstable modes.}$$

By enforcing these constraints at $t_i$ and $t_f$, the finite-time boundary value problem inherits the correct asymptotic behavior, ensuring that the switching trajectory departs and arrives in the proper subspaces.

### 2.3.3. Numerical Solution

The task of finding the switching paths is now cast as a boundary value problem (BVP). Our next task is apply a numerical method to find a solution. For this we will use SciPy's `solve_bvp` [41] implementation of a collocation method, meaning a numerical technique for solving differential equations by approximating the solution using a set of basis functions, such as cubic splines, and enforcing the differential equations are satisfied to a given tolerance at specific points called collocation points.

#### Convergence

When applying this solver to our problem there are two key considerations to check convergence of the solution:

1. **Finite Time Domain**: Since we cannot numerically integrate from $t \to -\infty$ to $t \to +\infty$, we must choose finite initial and final times $t_i$ and $t_f$. These should be chosen such that the system is sufficiently close to the fixed points at the boundaries. 

2. **Numerical Parameters**: The key parameter for solution quality is the error tolerance, which caps the allowed residuals at the collocation points [41]. Lower error tolerances can be reached by increasing the number of collocation points, at the cost of increased computational time and memory usage.

Whether or not they are we have converged to a desired solution can be judged by whether or not the action along the path has reached to a stable value. This can be checked by recalculating the paths and actions using different values of $t_f - t_i$, error tolerances and numbers of collocation points and checking if the resulting actions are consistent.

#### Initial Guess

The BVP solver requires an initial guess for the solution. This can be constructed by linear interpolation between the fixed points. This initial guess is most effective near the saddle-node bifurcations where the stable and saddle points are closest to each other. For more distant points with more complex paths, this linear guess may not converge to a solution, in which case we can reuse solutions from neighbouring points in the parameter space (numerical continuation).

### 2.3.4. Implementation Example

Here's an example of how to implement the switching paths calculation in Python using our codebase [40]:

```python
from pathlib import Path
from metastable.map.map import FixedPointMap, FixedPointType
from metastable.map.visualisations.bifurcation_lines import plot_bifurcation_diagram
from metastable.paths import (
    get_bistable_kappa_range, 
    generate_sweep_index_pairs,
    map_switching_paths
)
from metastable.action.map import map_actions
from metastable.paths.visualization import plot_parameter_sweeps


# Load the fixed point map
map_path = Path("fixed_points/examples/map-with-stability.npz")
fixed_point_map = FixedPointMap.load(map_path)

# Create the bifurcation diagram
fig = plot_bifurcation_diagram(fixed_point_map)

# Choose an epsilon index for the kappa cut
epsilon_idx = 380

# Get the bistable kappa range for this epsilon
kappa_boundaries = get_bistable_kappa_range(fixed_point_map.bistable_region, epsilon_idx)

# Generate kappa cuts
kappa_cuts = generate_sweep_index_pairs(kappa_boundaries, bright_sweep_fraction=0.4, dim_sweep_fraction=0.95)

# Get the actual epsilon value from index
epsilon_value = fixed_point_map.epsilon_linspace[epsilon_idx]

# Plot the bistable range
if kappa_boundaries.dim_saddle is not None:
    kappa_start = fixed_point_map.kappa_linspace[kappa_boundaries.dim_saddle.kappa_idx]
    kappa_end = fixed_point_map.kappa_linspace[kappa_boundaries.bright_saddle.kappa_idx]
    fig.add_scatter(
        x=[kappa_start, kappa_end], 
        y=[epsilon_value, epsilon_value], 
        mode='markers', 
        marker=dict(size=10, color='red'), 
        name='Bistable Range ($\kappa$)'
    )

# Plot parameter sweeps
fig = plot_parameter_sweeps(fixed_point_map, kappa_cuts, fig)

output_path = Path("sweep")

# Map switching paths for bright fixed point
path_results_bright = map_switching_paths(
    fixed_point_map, 
    kappa_cuts.bright_saddle, 
    output_path,
    t_end=10.0,
    endpoint_type=FixedPointType.BRIGHT,
    max_nodes=1000000,
    tol=1e-3
)

# Map switching paths for dim fixed point
path_results_dim = map_switching_paths(
    fixed_point_map, 
    kappa_cuts.dim_saddle, 
    output_path, 
    t_end=10.0,
    endpoint_type=FixedPointType.DIM,
)

# Calculate actions for all switching paths
fixed_point_map = FixedPointMap.load(output_path / "map.npz")
fixed_point_map_with_actions = map_actions(fixed_point_map)
fixed_point_map_with_actions.save(output_path / "map.npz")
```

Key aspects of the implementation:

1. **Parameter Selection**: We first choose a fixed value of $\epsilon$ (using index 380) and find the bistable range of $\kappa$ values.

2. **Visualization Setup**: The code creates a bifurcation diagram and visualizes the parameter sweep regions.

3. **Path Calculation**: Two separate calls to `map_switching_paths` calculate:
   - Paths from bright fixed points to saddle points
   - Paths from dim fixed points to saddle points

4. **Numerical Parameters**: 
   - `t_end=10.0`: Integration time domain (from -10.0 to 10.0)
   - `tol=1e-3`: Error tolerance for the bright-to-saddle paths
   - `max_nodes=1000000`: Maximum number of collocation points for the bright paths

5. **Action Calculation**: After finding the paths, the corresponding actions are calculated using `map_actions`.

This implementation allows us to systematically explore the switching paths across different regions of parameter space and visualize the results together with the bifurcation structure.

## 2.4. Results

We continue to examine the system studied in the stability analysis. We begin by finding the switching paths and actions as a function of $\kappa$ at fixed $\epsilon/\delta = 2.44$. The results are shown in the interactive visualization below, which consists of two panels:

1. **Upper Panel (Bifurcation Diagram)**: Shows the bifurcation structure in the $(\kappa/\delta, \epsilon/\delta)$ plane. The red and blue lines represent the unstable-bright and unstable-dim bifurcation boundaries respectively. The horizontal dashed black line indicates our chosen $\epsilon/\delta = 2.44$ cut, and the grey shaded region indicates the bistable region where switching paths exist.

2. **Lower Panel (Action Values)**: Displays the calculated actions for the switching paths:
   - Red line: Keldysh action $R_{b\to u}$ for the bright-to-unstable transition
   - Blue line: Keldysh action $R_{d\to u}$ for the dim-to-unstable transition
   - Purple dash-dot line: Kramers (analytical) prediction for $R_{b\to u}$
   - Green dash-dot line: Kramers (analytical) prediction for $R_{d\to u}$

The actions are measured relative to the scaled Planck cosntant $\lambda = \chi/\delta$. The close agreement between the numerically computed Keldysh actions and the analytical Kramers predictions [13, 14] validates our numerical approach close to the bifurcation points. Deeper into the bistable regime we see that the two methods diverge indicating the breakdown of the 1-D approximation on which the Kramers approach is based.

![Kappa sweep with actions](paths/supplemental_png/kappa_sweep_with_actions.png)

The switching paths were computed using boundary conditions that ensure proper alignment with the stable and unstable manifolds at each fixed point, with thresholds set to $10^{-3}$ for both stable and saddle points. The numerical integration was performed over a finite time domain $(t_f - t_i )\delta = 78.0$, which proved sufficient for convergence of the action values.

Then as a function of $\epsilon$ at fixed $\kappa/\delta = 0.240$. The results are shown in the interactive visualization below, which consists of two panels:

1. **Upper Panel (Bifurcation Diagram)**: Shows the bifurcation structure in the $(\kappa/\delta, \epsilon/\delta)$ plane. The red and blue lines represent the unstable-bright and unstable-dim bifurcation boundaries respectively. The vertical dashed black line indicates our chosen $\kappa/\delta = 0.240$ cut, and the grey shaded region indicates the bistable region where switching paths exist.

2. **Lower Panel (Action Values)**: Displays the calculated actions for the switching paths:
   - Red line: Keldysh action $R_{b\to u}$ for the bright-to-unstable transition
   - Blue line: Keldysh action $R_{d\to u}$ for the dim-to-unstable transition
   - Purple dash-dot line: Kramers (analytical) prediction for $R_{b\to u}$
   - Green dash-dot line: Kramers (analytical) prediction for $R_{d\to u}$

As above, the agreement between the numerically computed Keldysh actions and the analytical Kramers predictions helps to validate our numerical approach.

![Epsilon sweep with actions](paths/supplemental_png/epsilon_sweep_with_actions.png)

The switching paths were computed using boundary conditions that ensure proper alignment with the stable and unstable manifolds at each fixed point, with thresholds set to $10^{-2}$ for both stable and saddle points. The numerical integration was performed over a finite time domain $( t_f - t_i ) \delta = 85.8$, which proved sufficient for convergence of the action values close to the bifurcation points. However in the middle of the bistable regime convergence was not achieved so actions are not plotted here.

