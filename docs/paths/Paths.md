# Switching Paths

In this section we can now turn our attention to finding the switching paths connecting the stable fixed points to the saddle point. We will use our [Stability Analysis](../fixed_points/StabilityAnalysis.md) from the previous section to set the boundary conditions for the switching paths. We will then use a collocation method implemented in SciPy [1] to solve the resulting boundary value problem and find the switching paths.

## 1. Problem Overview

### 1.1. Path

Our key objective is to find a path $\mathbf{Z}(t)$ that connects a **stable fixed point** $\mathbf{Z}_0$ (node or focus) to a **saddle point** $\mathbf{Z}_s$.

The key idea is to use the eigenvectors of the Jacobian at both points to set the boundary conditions. Small deviations from the fixed points can be expressed in the eigenvector basis of the Jacobian and we control the boundary conditions by specifying the coefficients of the eigenvectors in those deviations.

At each fixed point the Jacobian has a pair of incoming and outgoing eigenvectors. We simply apply the condition that at the start of the path the deviation is purely along the eigenvectors leaving the stable point and at the end of the path the deviation is purely along the eigenvectors arriving at the saddle point.

### 1.2. Action

With these paths we can finally calculate the corresponding actions using the formula shown in [Keldysh Auxiliary Hamiltonian](../derivation/KeldyshAuxiliaryHamiltonian.md):

$$
\begin{aligned}
S_{\mathrm{aux}} = -\int dt\Bigl[\dot{x}_c\,x_q + \dot{p}_c\,p_q - H(x_c,p_c,x_q,p_q)\Bigr].
\end{aligned}
$$

Since the Hamiltonian is conservative (i.e. time independent) and the initial and final points are in the classical plane where $H=0$, the action can be simplified to:

$$
S_{\mathrm{aux}} = -\int dt\Bigl[\dot{x}_c\,x_q + \dot{p}_c\,p_q\Bigr].
$$


## 2. Mathematical Framework

### 2.1. At the Stable Fixed Point

Let $\mathbf{Z}_0$ denote the coordinates of the stable fixed point. Small deviations from $\mathbf{Z}_0$ can be expressed in the eigenvector basis of the Jacobian $J(\mathbf{Z}_0)$:
  
$$
\Delta \mathbf{Z}(t) \equiv \mathbf{Z}(t) - \mathbf{Z}_0 = \sum_{i=1}^{4} c_i\, \mathbf{v}_i\, e^{\lambda_i t},
$$
  
where:
  - $\mathbf{v}_i$ are the eigenvectors,
  - $\lambda_i$ are the corresponding eigenvalues, ordered by their real parts with $\text{Re}(\lambda_1), \text{Re}(\lambda_2) < 0$ (stable) and $\text{Re}(\lambda_3), \text{Re}(\lambda_4) > 0$ (unstable),
  - $c_i$ are coefficients determined by the initial displacement.

### 2.2. At the Saddle Point

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

## 3. Formulating the Boundary Value Problem

### 3.1. Physical Interpretation of Boundary Conditions

The switching trajectory represents the optimal (least-action) path connecting a metastable fixed point to the saddle point. It is a solution to the equations of motion derived from the auxiliary Hamiltonian as seen in [FixedPoints](../fixed_points/FixedPoints.md). The eigenvector analysis from the [Stability Analysis](../fixed_points/StabilityAnalysis.md) then provides natural boundary conditions:

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

### 3.2 Projection

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

### 3.3. Numerical Solution

The task of finding the switching paths is now cast as a boundary value problem (BVP). Our next task is apply a numerical method to find a solution. For this we will use SciPy's `solve_bvp` [1] implementation of a collocation method, meaning a numerical technique for solving differential equations by approximating the solution using a set of basis functions, such as cubic splines, and enforcing the differential equations are satisfied to a given tolerance at specific points called collocation points.

#### Convergence

When applying this solver to our problem there are two key considerations to check convergence of the solution:

1. **Finite Time Domain**: Since we cannot numerically integrate from $t \to -\infty$ to $t \to +\infty$, we must choose finite initial and final times $t_i$ and $t_f$. These should be chosen such that the system is sufficiently close to the fixed points at the boundaries. 

2. **Numerical Parameters**: The key parameter for solution quality is the error tolerance, which caps the allowed residuals at the collocation points [1]. Lower error tolerances can be reached by increasing the number of collocation points, at the cost of increased computational time and memory usage.

Whether or not they are we have converged to a desired solution can be judged by whether or not the action along the path has reached to a stable value. This can be checked by recalculating the paths and actions using different values of $t_f - t_i$, error tolerances and numbers of collocation points and checking if the resulting actions are consistent.

#### Initial Guess

The BVP solver requires an initial guess for the solution. This can be constructed by linear interpolation between the fixed points. This initial guess is most effective near the saddle-node bifurcations where the stable and saddle points are closest to each other. For more distant points with more complex paths, this linear guess may not converge to a solution, in which case we can reuse solutions from neighbouring points in the parameter space (numerical continuation).

## 4. Results

We continue to examine the system studied in [Stability Analysis](../fixed_points/StabilityAnalysis.md). We begin by finding the switching paths and actions as a function of $\kappa$ at ...


<div class="plotly-container" style="position: relative; width: 100%; height: 850px; margin: 0 auto;">
    <iframe src="kappa_sweep_with_actions.html" 
            style="position: absolute; width: 100%; height: 100%; border: none;"
            allowfullscreen>
    </iframe>
</div>

[Open visualization in new window](kappa_sweep_with_actions.html)


Then as a function of $\epsilon$ at ...

<div class="plotly-container" style="position: relative; width: 100%; height: 850px; margin: 0 auto;">
    <iframe src="epsilon_sweep_with_actions.html" 
            style="position: absolute; width: 100%; height: 100%; border: none;"
            allowfullscreen>
    </iframe>
</div>

[Open visualization in new window](epsilon_sweep_with_actions.html)



## References

[1] SciPy documentation, "scipy.integrate.solve_bvp", [https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html).

[2] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27, Number 3, pp. 299-316, 2001.

