# Stability Analysis of the Full Equations of Motion

In the previous section on the [Fixed Points](./FixedPoints.md) we identified steady states of the classical equations of motion and classified them as either saddles or nodes. But our intention is to find the switching trajectories which allow the system to escape from the nodes and reach the saddle point, and for this we need to extend our analysis to include the quantum fields.

In this section we will use the Jacobian matrix to linearise the full equations of motion around the nodes and reveal that, although they are classically stable, introducing the quantum fields allows for escape trajectories along new unstable eigenvectors. Meanwhile we will also linearise around the saddle point and reveal a stable eigenvector which allows the system to reach the saddle point from the quantum part of the phase space.

This stability analysis will be crucial for setting the boundary conditions for the switching trajectories, as we will see in the next section.

---

## 1. Equations of Motion

The starting point for our analysis is the full Hamiltonian of the system, which we recall is given by

$$
\begin{aligned}
H(x_c,p_c,x_q,p_q)=\; & \Biggl(\delta + \frac{\chi}{2}\bigl(x_c^2+p_c^2-x_q^2-p_q^2\bigr)\Biggr)(p_c\,x_q - x_c\,p_q)\\[1mm]
&\quad - \kappa\,(x_c\,x_q+p_c\,p_q)
+\kappa\,(x_q^2+p_q^2)
+2\epsilon\,x_q.
\end{aligned}
$$

The full dynamics are described in terms of the classical and quantum fields by the Hamilton's equations:

$$
\dot{\mathbf{z}}_c = \frac{\partial H}{\partial \mathbf{z}_q}, \qquad \dot{\mathbf{z}}_q = -\frac{\partial H}{\partial \mathbf{z}_c},
$$

where we have defined the field vectors as

$$
\mathbf{z}_c = (x_c, p_c), \quad \mathbf{z}_q = (x_q, p_q).
$$

We then define the full state vector as

$$
\mathbf{Z} = (x_c, p_c, x_q, p_q)
$$

## 2. Linearisation

In order to linearise the equation of motion, we write the state of the system as

$$
\mathbf{Z}(t) = \mathbf{Z}_0 + \Delta \mathbf{Z}(t),
$$

where

$$
\mathbf{Z}_0 = (x_{c0}, p_{c0}, x_{q0}, p_{q0})
$$

denotes the coordinates of a fixed point, and $\Delta \mathbf{Z}(t)$ represents a small deviation from that fixed point. Expanding the equations of motion to first order in $\Delta \mathbf{Z}(t)$ yields

$$
\frac{d}{dt}\,\Delta\mathbf{Z}(t) = J(\mathbf{Z}_0)\,\Delta\mathbf{Z}(t),
$$

where $J(\mathbf{Z}_0)$ is the Jacobian matrix evaluated at the fixed point $\mathbf{Z}_0$. The Jacobian matrix $J$, evaluated at a fixed point $\mathbf{Z}_0$, is given by:
$$
J(\mathbf{Z}_0)_{ij} = \left.\frac{\partial \dot{\mathbf{Z}}_i}{\partial \mathbf{Z}_j}\right|_{\mathbf{Z}=\mathbf{Z}_0}
$$

Since the equations of motion can then be written in vector form as

$$
\dot{\mathbf{Z}} = (\dot{x}_c, \dot{p}_c, \dot{x}_q, \dot{p}_q) = \left(\frac{\partial H}{\partial x_q}, \frac{\partial H}{\partial p_q}, -\frac{\partial H}{\partial x_c}, -\frac{\partial H}{\partial p_c}\right)
$$

In explicit form, the $4\times4$ Jacobian matrix is given by:

$$
J(\mathbf{Z}_0) = \begin{pmatrix}
\frac{\partial^2 H}{\partial x_q\partial x_c} & \frac{\partial^2 H}{\partial x_q\partial p_c} & \frac{\partial^2 H}{\partial x_q^2} & \frac{\partial^2 H}{\partial x_q\partial p_q} \\[2mm]
\frac{\partial^2 H}{\partial p_q\partial x_c} & \frac{\partial^2 H}{\partial p_q\partial p_c} & \frac{\partial^2 H}{\partial p_q\partial x_q} & \frac{\partial^2 H}{\partial p_q^2} \\[2mm]
-\frac{\partial^2 H}{\partial x_c^2} & -\frac{\partial^2 H}{\partial x_c\partial p_c} & -\frac{\partial^2 H}{\partial x_c\partial x_q} & -\frac{\partial^2 H}{\partial x_c\partial p_q} \\[2mm]
-\frac{\partial^2 H}{\partial p_c\partial x_c} & -\frac{\partial^2 H}{\partial p_c^2} & -\frac{\partial^2 H}{\partial p_c\partial x_q} & -\frac{\partial^2 H}{\partial p_c\partial p_q}
\end{pmatrix}
$$

We can now study the stability of any fixed point by finding the eigenvalues and eigenvectors of this matrix.

---

## 3. Eigenvector Analysis

The eigenvalues and eigenvectors of the Jacobian $J(\mathbf{Z}_0)$ provide the local stability properties of the fixed point. Specifically, consider the eigenvalue problem

$$
J(\mathbf{Z}_0)\,\mathbf{v}_i = \lambda_i\, \mathbf{v}_i.
$$

Any small deviation from the fixed point can be expressed as a linear combination of these eigenvectors. In particular, we write

$$
\Delta \mathbf{Z}(t) = \sum_{i=1}^{4} c_i \,\mathbf{v}_i\, e^{\lambda_i t},
$$

where the coefficients $c_i$ are determined by the initial condition

$$
\Delta \mathbf{Z}(0) = \sum_{i=1}^{4} c_i \,\mathbf{v}_i.
$$

This expansion illustrates how $\Delta\mathbf{Z}(t)$ evolves over time: each eigenmode evolves as $e^{\lambda t}$, where the real part of $\lambda$ governs the exponential growth or decay. Specifically, eigenvalues with positive real parts lead to exponential growth of the corresponding perturbations, while those with negative real parts result in exponential decay. Moreover, if an eigenvalue is complex, its imaginary part will introduce oscillatory behavior on top of this exponential trend.

Because the phase space is four-dimensional, the Jacobian has four eigenvalues (and corresponding eigenvectors):

- For a **metastable fixed point** (bright or dim state), typically two eigenvalues (with eigenvectors that lie predominantly in the classical plane) will have **negative real parts**, indicating that small perturbations in these directions decay. In contrast, the other two eigenvalues (which generally have significant components in the quantum directions) will have **positive real parts**—these indicate the directions along which the system can escape.
  
- For the **saddle (unstable) fixed point**, the eigenvalues will have mixed signs. Typically one (or more) direction will be stable (negative real part) while another is unstable (positive real part).

A graphical representation (not shown here) can help visualize the stable manifold (directions corresponding to decaying perturbations) and the unstable manifold (directions along which perturbations grow) at a fixed point.

## Expectations on the Eigenvalues

# Eigenvalue and Eigenvector Expectations in Stability Analysis

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
