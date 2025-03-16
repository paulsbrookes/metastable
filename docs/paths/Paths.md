# Switching Paths

In this section we can now turn our attention to finding the switching paths connecting the stable fixed points to the saddle point. We will use our [Stability Analysis](../fixed_points/StabilityAnalysis.md) from the previous section to set the boundary conditions for the switching paths. We will then use a collocation method implemented in SciPy [1] to solve the resulting boundary value problem and find the switching paths.

## 1. Problem Overview

Our key objective is to find a path $\mathbf{Z}(t)$ that connects a **stable fixed point** $\mathbf{Z}_0$ (node or focus) to a **saddle point** $\mathbf{Z}_s$.

The key idea is to use the eigenvectors of the Jacobian at both points to set the boundary conditions. Small deviations from the fixed points can be expressed in the eigenvector basis of the Jacobian and we control the boundary conditions by specifying the coefficients of the eigenvectors in those deviations.

At each fixed point the Jacobian has a pair of incoming and outgoing eigenvectors. We simply apply the condition that at the start of the path the deviation is purely along the eigenvectors leaving the stable point and at the end of the path the deviation is purely along the eigenvectors arriving at the saddle point.

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

The switching trajectory represents the optimal (least-action) path connecting a metastable fixed point to the saddle point. It is a solution to the equations of motion derived from the Hamiltonian. The eigenvector analysis provides natural boundary conditions:

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

### 3.3. Matching the Trajectory

- The task is to adjust the coefficients ${c_i}$ and ${d_j}$ such that the trajectory $\mathbf{Z}(t)$ satisfies the equations of motion and simultaneously meets the boundary conditions at both $\mathbf{Z}_0$ and $\mathbf{Z}_s$.

- Numerical methods (e.g., shooting or collocation techniques) are then applied to iteratively refine the trajectory and ensure that it connects the two fixed points in phase space.


### 3.2. Differential Equations

The full dynamics are governed by the Hamiltonian equations of motion:

$$
\dot{\mathbf{Z}} = \mathbf{F}(\mathbf{Z}),
$$
  
where $\mathbf{F}(\mathbf{Z})$ represents the set of equations derived from the full Hamiltonian.

---

## 4. Numerical Implementation Outline

- **Initial Guess**: Provide an initial trajectory guess that interpolates between $\mathbf{Z}_0$ and $\mathbf{Z}_s$.

- **Iterative Solver**: Use a boundary value solver to adjust the trajectory such that:
  - The residual of $\dot{\mathbf{Z}} - \mathbf{F}(\mathbf{Z})$ is minimized.
  - The deviations at $t \to \pm\infty$ match the prescribed eigenvector expansions.

- **Validation**: Check that:
  - Near $\mathbf{Z}_0$, the trajectory is well approximated by $\mathbf{Z}_0 + \sum_i c_i\, \mathbf{v}_i\, e^{\lambda_i t}$.
  - Near $\mathbf{Z}_s$, the trajectory follows $\mathbf{Z}_s + \sum_j d_j\, \mathbf{u}_j\, e^{\mu_j t}$.

---

## 5. Summary and Next Steps

- **Summary**: We have recast the problem of finding switching trajectories as a boundary value problem, expressing the displacements from both the fixed point and the saddle in terms of the eigenvectors of the Jacobian.

- **Next Steps**: 
  - Detail the specific numerical methods (e.g., shooting, collocation) and their implementation.
  - Present sample results and discuss the challenges in convergence and accuracy.

## References

[1] "SciPy Boundary Value Problem Solver Documentation", see [scipy.integrate.solve_bvp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html).

