# Next Section

Because the phase space is four-dimensional, the Jacobian has four eigenvalues and eigenvectors:

- For a **metastable fixed point** (bright or dim state), typically two eigenvalues (with eigenvectors that lie predominantly in the classical plane) will have **negative real parts**, indicating that small perturbations in these directions decay. In contrast, the other two eigenvalues (which generally have significant components in the quantum directions) will have **positive real parts**—these indicate the directions along which the system can escape.
  
- For the **saddle (unstable) fixed point**, the eigenvalues will have mixed signs. Typically one (or more) direction will be stable (negative real part) while another is unstable (positive real part).

A graphical representation (not shown here) can help visualize the stable manifold (directions corresponding to decaying perturbations) and the unstable manifold (directions along which perturbations grow) at a fixed point.

## Expectations on the Eigenvalues

# Eigenvalue and Eigenvector Expectations in Stability Analysis

Because the full system is Hamiltonian—with canonical coordinates for the classical and quantum fields—the Jacobian obtained by linearizing the equations of motion is a Hamiltonian (or symplectic) matrix. This fact immediately leads to some general expectations regarding its eigenvalues and eigenvectors.

## Paired Spectrum

In any Hamiltonian system, if $\lambda$ is an eigenvalue, then $-\lambda$ must also be an eigenvalue. For complex eigenvalues, the spectrum appears in quadruplets: $\lambda$, $-\lambda$, $\bar{\lambda}$, and $-\bar{\lambda}$. This symmetry ensures that the spectrum is symmetric with respect to the real axis, meaning any unstable growth (positive real part) is balanced by an opposing decay (negative real part).

## Node Fixed Points

In the classical subsystem (2D phase space), the fixed points identified as “nodes” are stable; small perturbations decay over time. However, when quantum fields are included, the phase space expands to 4D, introducing new stability behavior:

- **Classical directions**: Two eigenvalues (or one conjugate pair) primarily affecting the classical variables remain stable (having negative real parts or being purely imaginary in a conservative system).
- **Quantum directions**: The additional quantum degrees of freedom introduce new eigenvalues. Typically, one of these eigenvalues has a positive real part, leading to an **unstable direction**—an "escape mode"—even though the node was classically stable.

Thus, the previously stable classical node now allows for quantum-assisted escape trajectories.

## Saddle Fixed Points

For a saddle fixed point, the classical stability analysis already reveals one stable and one unstable direction. When the quantum fields are included:

- The eigenvalues remain paired as $\pm\lambda$.
- One of the additional eigenvectors (primarily in the quantum subspace) often becomes stable (having a negative real part), allowing the system to **approach the saddle** from the quantum phase space.

This stable quantum eigenvector is essential for constructing switching trajectories, as it provides a **channel** through which the system can reach the saddle point.

## Switching Trajectories and Eigenmodes

Since small perturbations evolve as

$$
\Delta \mathbf{Z}(t) = \sum_{i=1}^{4} c_i \,\mathbf{v}_i\, e^{\lambda_i t},
$$

the eigenvectors define the directions in phase space along which perturbations either grow or decay:

- **For a node**, the unstable eigenvector (emerging from the quantum sector) provides the escape path.
- **For a saddle**, the stable quantum eigenvector determines the direction from which the system can approach.

These stability properties are crucial for defining **boundary conditions** for the switching trajectories, which describe how the system escapes from stable nodes and reaches the saddle point.


### Notes

In the classical case, when the system is close to the stable fixed points it spirals around them while decaying towards them. The fixed poitns are stable focus points across most of parameter space, but at the bifurcation points they are nodes and there is no spiralling motion governed by a pair of complex conjugate eigenvalues with a negative real component governing the decay towards the fixed point and an imaginary component governing frequency of the oscillations.

When we extend the system to include the quantum fields, the system is governed Hamiltonian dynamics. The Jacobian is symplectic and has a paired spectrum. For each eigenvalue $\lambda$ there is a corresponding eigenvalue $-\lambda$. Furthermore, if $\lambda$ is complex, then $\lambda^{\*}$ is also an eigenvalue. For this reason we expect the spectrum around the focus points to include both ingoing and outgoing eigenvectors with eigenvalues $\pm \lambda$ and $\pm \lambda^{\*}$.

The beahviour at the saddle points is different. Whereas the in the classical case we had eigenvalues $- \lvert \kappa_1 \rvert$ and $-\lvert \kappa_2 \rvert$ corresponding to the incoming and outgoing directions, in the quantum case the Hamiltonian dynamics will give us a new pair of eigenvectors with opposite eigenvalues $\lvert \kappa_1 \rvert$ and $- \lvert \kappa_2 \rvert$.

We plot these below.

---

## 4. Setting the Boundary Conditions for Switching Trajectories

The switching trajectory is the optimal (least-action) path that connects a metastable fixed point to the saddle point. The eigenvector analysis of the Jacobian provides the natural boundary conditions for this trajectory:

- **At the metastable fixed point:**  
  The trajectory must *depart* along the unstable manifold. This means that, for a metastable state (bright or dim), the initial deviation $\delta \mathbf{Z}(0)$ should be aligned with the eigenvector(s) associated with eigenvalues having positive real parts. These eigenvectors indicate the direction in which a fluctuation can push the system out of the basin of attraction.

- **At the saddle (unstable) fixed point:**  
  The trajectory must *arrive* along the stable manifold. That is, as the trajectory approaches the saddle point, the deviation $\Delta\mathbf{Z}(t_f)$ should lie within the subspace spanned by the eigenvector(s) corresponding to eigenvalues with negative real parts. This ensures that the escape path connects smoothly to the classical dynamics leading from the saddle to the other metastable state.

These conditions allow us to set up a boundary value problem (BVP) for the full equations of motion:
- **Initial Condition:** $\mathbf{Z}(0)$ is chosen as the metastable fixed point plus a small displacement along the unstable eigenvector.
- **Final Condition:** $\mathbf{Z}(t_f)$ is required to approach the saddle fixed point along the stable eigenvector direction.

Once these boundary conditions are defined, numerical techniques (such as shooting methods or relaxation methods) can be used to solve the full nonlinear equations and extract the switching trajectory. The action computed along this path then directly enters the expression for the switching rate.

---

*This concludes Sections 1 through 5. In the next section we will detail the numerical solution of the boundary value problem and discuss how the computed action compares with the switching rates obtained via other methods.*
