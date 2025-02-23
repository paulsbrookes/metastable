# Boundary Value Methods for Switching Trajectories

In this section we focus on finding the switching trajectories using boundary value methods. The goal is to determine the optimal escape path that connects a stable fixed point to a saddle point, taking advantage of the eigenstructure of the Jacobian matrices at these points.

## 1. Problem Overview

- **Objective**: Find a trajectory $\mathbf{Z}(t)$ that connects:
  - A **stable fixed point** $\mathbf{Z}_0$ (node or focus)
  - A **saddle point** $\mathbf{Z}_s$

- **Key Idea**: Use the eigenvectors of the Jacobian at both points to:
  1. Express deviations from fixed points naturally
  2. Set appropriate boundary conditions
  3. Guide numerical solution methods

## 2. Mathematical Framework

### 2.1. System Dynamics

## 2. Displacement from Fixed Points in the Eigenvector Basis

### 2.1. At the Stable Fixed Point

- Let $\mathbf{Z}_0$ denote the coordinates of the stable fixed point.
- Small deviations from $\mathbf{Z}_0$ can be expressed in the eigenvector basis of the Jacobian $J(\mathbf{Z}_0)$:
  
$$
\Delta \mathbf{Z}(t) \equiv \mathbf{Z}(t) - \mathbf{Z}_0 = \sum_{i=1}^{4} c_i\, \mathbf{v}_i\, e^{\lambda_i t},
$$
  
where:
  - $\mathbf{v}_i$ are the eigenvectors,
  - $\lambda_i$ are the corresponding eigenvalues,
  - $c_i$ are coefficients determined by the initial displacement.

### 2.2. At the Saddle Point

- Let $\mathbf{Z}_s$ denote the coordinates of the saddle point.
- Near $\mathbf{Z}_s$, the deviation is similarly expressed in terms of the eigenvectors of the Jacobian $J(\mathbf{Z}_s)$:
  
$$
\Delta \mathbf{Z}(t) \equiv \mathbf{Z}(t) - \mathbf{Z}_s = \sum_{j=1}^{4} d_j\, \mathbf{u}_j\, e^{\mu_j t},
$$
  
where:
  - $\mathbf{u}_j$ are the eigenvectors at the saddle,
  - $\mu_j$ are the corresponding eigenvalues,
  - $d_j$ are coefficients that characterize the deviation.

---

## 3. Formulating the Boundary Value Problem

### 3.1. Physical Interpretation of Boundary Conditions

The switching trajectory represents the optimal (least-action) path connecting a metastable fixed point to the saddle point. The eigenvector analysis provides natural boundary conditions:

- **At the Stable Fixed Point $\mathbf{Z}_0$:**
  - The trajectory must *depart* along the unstable manifold
  - Initial deviation $\Delta \mathbf{Z}(t)$ aligns with eigenvectors having $\text{Re}(\lambda_i) > 0$
  - These directions indicate where fluctuations can push the system out of the basin of attraction

- **At the Saddle Point $\mathbf{Z}_s$:**
  - The trajectory must *arrive* along the stable manifold
  - Final deviation $\Delta \mathbf{Z}(t)$ lies in the subspace of eigenvectors with $\text{Re}(\mu_j) < 0$
  - This ensures smooth connection to classical dynamics leading to the other metastable state

### 3.2. Differential Equations

- The full dynamics are governed by the Hamiltonian equations of motion:

$$
\dot{\mathbf{Z}} = \mathbf{F}(\mathbf{Z}),
$$
  
where $\mathbf{F}(\mathbf{Z})$ represents the set of equations derived from the full Hamiltonian.

### 3.3. Boundary Conditions

- **At $t \to -\infty$** (approaching the stable fixed point):
  
$$
\mathbf{Z}(t) \to \mathbf{Z}_0 \quad \Longrightarrow \quad \Delta \mathbf{Z}(t) \approx \sum_{i} c_i\, \mathbf{v}_i\, e^{\lambda_i t},
$$
  
where only the modes with $\text{Re}(\lambda_i) > 0$ (unstable directions) are allowed to grow.

- **At $t \to +\infty$** (approaching the saddle point):
  
$$
\mathbf{Z}(t) \to \mathbf{Z}_s \quad \Longrightarrow \quad \Delta \mathbf{Z}(t) \approx \sum_{j} d_j\, \mathbf{u}_j\, e^{\mu_j t},
$$
  
where only the modes with $\text{Re}(\mu_j) < 0$ (stable directions) remain bounded.

### 3.4. Matching the Trajectory

- The task is to adjust the coefficients ${c_i}$ and ${d_j}$ such that the trajectory $\mathbf{Z}(t)$ satisfies the equations of motion and simultaneously meets the boundary conditions at both $\mathbf{Z}_0$ and $\mathbf{Z}_s$.

- Numerical methods (e.g., shooting or collocation techniques) are then applied to iteratively refine the trajectory and ensure that it connects the two fixed points in phase space.

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

