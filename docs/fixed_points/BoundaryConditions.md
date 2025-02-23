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

Meanwhile, at the saddle point in the classical case the two eigenvalues are given by:

$$
\lambda \in \{ -\kappa_1, +\kappa_2 \}.
$$

These describe the incoming and outgoing motion around the saddle point in the classical co-ordinate plane. Again, when we extend our analysis to include the quantum fields these two eigenvalues are paired with two new eigenvalues that describe incoming and outgoing motion along directions in the full four-dimensional phase space:

$$
\lambda \in \{ -\kappa_1, +\kappa_2, +\kappa_1, -\kappa_2 \}.
$$

As above, the new eigenvectors obey a biorthogonality condition with the classical eigenvectors and in general will have components in both the classical and quantum sectors.

## Numerical Results

Now that we have an understanding of the eigenvalues and eigenvectors of the Jacobian, we can use this to study the dynamics of the system across parameter space. For this purpose we use the [generate_jacboian_spectrum.py](examples/generate_jacobian_spectrum.py) script:

```python
from metastable.map.map import FixedPointMap
from metastable.generate_stability_map import generate_stability_map

if __name__ == '__main__':
    # Load the fixed point map
    map = FixedPointMap.load("map.npz")
    
    # Generate stability map
    map_with_stability = generate_stability_map(map, n_workers=20)
    
    # Save the updated map
    map_with_stability.save("map-with-stability.npz")
```

The `generate_stability_map` function iterates over the fixed points stored in the `FixedPointMap` object and computes the eigenvalues and eigenvectors of the Jacobian at each fixed point. The results can then be accessed at the `FixedPointMap.eigenvalues` and `FixedPointMap.eigenvectors` attributes.

Using the `FixedPointMap` produced in the previous chapter, we now plot the real and imaginary parts of the $\lambda_0$ and $\lambda_1$ eigenvalues around the low amplitude fixed point. We also plot the upper and lower bifurcation lines to clearly mark the limits of the bistable regime.

<div class="plotly-container" style="position: relative; width: 100%; height: 850px; margin: 0 auto;">
    <iframe src="examples/jacobian_spectrum_dim_fixed_point.html" 
            style="position: absolute; width: 100%; height: 100%; border: none;"
            allowfullscreen>
    </iframe>
</div>

[Open visualization in new window](examples/jacobian_spectrum_dim_fixed_point.html)

Let's examine the key features of the eigenvalue spectrum:

1. **Complex Eigenvalues and Oscillatory Motion**: Across most of the parameter space, the eigenvalues form complex conjugate pairs. This indicates oscillatory motion near the fixed points, classifying them as focus points.

2. **Relationship to Decay Rate**: For focus points, the real component's magnitude equals the system decay rate $\kappa$. As $\kappa \to 0$, while the real component vanishes, the imaginary component persists. This suggests that oscillatory motion occurs on a much faster timescale than decay or escape processes.

3. **Behavior Near Bifurcation**: Near the dim-saddle bifurcation point, the eigenvalues become purely real, transforming the fixed point into a node. At this transition:
   - The previously equal real parts take on different values.
   - One eigenvalue approaches zero while the other remains finite.
   - The vanishing eigenvalue reveals a soft mode connecting the fixed point to the saddle point
   This regime corresponds to the one-dimensional Kramers problem discussed in [REFERENCE].

