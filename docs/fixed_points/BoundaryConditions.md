# Stability Analysis of the Full Equations of Motion

In the previous section on the [Fixed Points](./FixedPoints.md) we identified steady states of the classical equations of motion and classified them as either saddles or nodes. Our overall goal is to find the switching trajectories which allow the system to escape from the nodes and reach the saddle point, and then calculate the actions of these trajectories with a view to obtaining the switching rates. For this we need to extend our analysis to include the quantum fields.

In this section we will use the Jacobian matrix to linearise the full equations of motion. This will reveal that:
1. The quantum fields allow for escape trajectories along new unstable eigenvectors leaving the classically stable fixed points. 
2. Around the saddle point there is a new stable eigenvector which allows the system to reach the saddle point from the quantum part of the phase space.

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

The full dynamics are described in terms of the classical and quantum fields by Hamilton's equations:

$$
\dot{\mathbf{z}}_c = \frac{\partial H}{\partial \mathbf{z}_q}, \qquad \dot{\mathbf{z}}_q = -\frac{\partial H}{\partial \mathbf{z}_c},
$$

where we have defined the field vectors as

$$
\mathbf{z}_c = (x_c, p_c), \quad \mathbf{z}_q = (x_q, p_q).
$$

For convenience in the next section, we also denote the full state vector by

$$
\mathbf{Z} = (x_c, p_c, x_q, p_q).
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

denotes the coordinates of a fixed point, and $\Delta \mathbf{Z}(t)$ represents a small deviation from that point. Expanding the equations of motion to first order in $\Delta \mathbf{Z}(t)$ yields

$$
\frac{d}{dt}\,\Delta\mathbf{Z}(t) = J(\mathbf{Z}_0)\,\Delta\mathbf{Z}(t),
$$

where $J(\mathbf{Z}_0)$ is the Jacobian matrix evaluated at $\mathbf{Z}_0$. The Jacobian matrix $J$, evaluated at a fixed point $\mathbf{Z}_0$, is given by

$$
J(\mathbf{Z}_0)_{ij} = \left.\frac{\partial \dot{\mathbf{Z}}_i}{\partial \mathbf{Z}_j}\right|_{\mathbf{Z}=\mathbf{Z}_0}
$$

The equations of motion can be written in vector form as

$$
\dot{\mathbf{Z}} = (\dot{x}_c, \dot{p}_c, \dot{x}_q, \dot{p}_q) = \left(\frac{\partial H}{\partial x_q}, \frac{\partial H}{\partial p_q}, -\frac{\partial H}{\partial x_c}, -\frac{\partial H}{\partial p_c}\right)
$$

and in explicit form, the $4\times4$ Jacobian matrix is given by:

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

## 3. Stability Analysis

## Background

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

## Paired Spectrum

Because the full system is Hamiltonian—with canonical coordinates for the classical and quantum fields—the Jacobian obtained by linearizing the equations of motion is a symplectic matrix, meaning it satisfies:

$$
J^T \Omega J = \Omega, \quad \text{where} \quad \Omega = \begin{pmatrix} 0 & I_2 \\ -I_2 & 0 \end{pmatrix}
$$

where $I_2$ is the $2\times2$ identity matrix. This fact immediately leads to some general expectations regarding its eigenvalues.

For any symplectic matrix, if $\lambda$ is an eigenvalue, then $-\lambda$ must also be an eigenvalue. Furthmore, any complex eigenvalues also come in conjugate pairs, leading overall to a spectrum that appears in quadruplets: $\lambda$, $-\lambda$, $\lambda^{\*}$, and $-\lambda^{\*}$.

Finally we note that since the Jacobian is not Hermitian, the eigenvectors are not necessarily orthogonal. However, for a symplectic matrix, if $\mathbf{v}$ is a right eigenvector with eigenvalue $\lambda$, then $\Omega\mathbf{v}$ is a left eigenvector with eigenvalue $-\lambda$. This leads to a biorthogonality relation between eigenvectors: if $\mathbf{v}_1$ and $\mathbf{v}_2$ are eigenvectors with eigenvalues $\lambda_1$ and $\lambda_2$, then $\mathbf{v}_1^T\Omega\mathbf{v}_2 = 0$ unless $\lambda_1 = -\lambda_2$. This structure plays an important role in determining the stable and unstable manifolds around the fixed points.

### Stable Points

In the previous chapter we performed stability analysis on the classical equations of motion, which are not Hamiltonian due to the presence of drive and dissipation. Around the stable fixed points the eigenvalues are given either by:

$$
\lambda \in \{ -\kappa_1, -\kappa_2 \}
$$

near a saddle-node bifurcation, or by

$$
\lambda \in \{ - \kappa - i \omega, - \kappa + i \omega \}
$$

deeper into the bistable regime. The first case describes a node with different decay rates in the two directions, while the second describes a focus point with a spiralling decaying motion.


When we extend our analysis to include the quantum fields the system becomes Hamiltonian and we find two additional eigenvalues with positive real parts that complete the expected quadruplets. So we find either:

$$
\lambda \in \{ -\kappa_1, -\kappa_2, \kappa_1, \kappa_2 \}
$$

near a saddle-node bifurcation, or

$$
\lambda \in \{ - \kappa - i \omega, - \kappa + i \omega, \kappa - i \omega, \kappa + i \omega \}
$$

away from it. Due to the biorthogonality condition above, each eigenvector from the classical sector is paired with an eigenvector that partially lies in the quantum sector. However we note that the quadruplet of eigenvectors is not orthogonal and the new eigenvectors will in general have components in the classical sector as well as the quantum sector.

### Saddle Point

Meanwhile, the saddle point we find two eigenvalues in the classical case:

$$
\lambda \in \{ -\kappa_1, +\kappa_2 \}.
$$

These describe the incoming and outgoing motion around the saddle point in the classical co-ordinate plane. Again, when we extend our analysis to include the quantum fields these two eigenvalues are paired with two new eigenvalues that describe incoming and outgoing motion along directions in the full four-dimensional phase space:

$$
\lambda \in \{ -\kappa_1, +\kappa_2, +\kappa_1, -\kappa_2 \}.
$$

As above, the new eigenvectors obey a biorthogonality condition with the classical eigenvectors and in general will have components in both the classical and quantum sectors.

## Numerical Results

Now that we have an understanding of the eigenvalues and eigenvectors of the Jacobian, we can use this to study the dynamics of the system across parameter space. We first plot the real and imaginary parts of 0th and 1st eigenvalues around the low amplitude fixed point.



















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
