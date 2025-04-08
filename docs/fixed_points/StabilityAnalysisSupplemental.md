# Supplemental Material to A Real-time Instanton Approach to Quantum Activation

# 1. Stability Analysis

Using the classical equations of motion we are able to identify steady states of the system and classify them as either saddles or stable points. Our overall goal is to find the switching trajectories which allow the system to escape from the stable points and reach the saddle point, and then calculate the actions of these trajectories with a view to obtain the switching rates. These paths don't exist in the classical case, but they do in the full system including the quantum fields.

In this section we will use the Jacobian matrix to linearise the full equations of motion. This will reveal that:
1. The quantum fields allow for escape trajectories along new unstable eigenvectors leaving the classically stable fixed points. 
2. Around the saddle point there is a new stable eigenvector which allows the system to reach the saddle point from the quantum part of the phase space.

This stability analysis will be crucial for setting the boundary conditions for the switching trajectories, as we will see in the next section.

---

## 1.1. Equations of Motion

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

## 1.2. Linearisation

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

## 1.3. Stability Analysis

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

Finally we note that since the Jacobian is not Hermitian, the eigenvectors are not necessarily orthogonal. However, for a symplectic matrix, if $\mathbf{v}$ is a right eigenvector with eigenvalue $\lambda$, then $\Omega\mathbf{v}$ is a left eigenvector with eigenvalue $-\lambda$. This leads to a biorthogonality relation between eigenvectors: if $\mathbf{v}_1$ and $\mathbf{v}_2$ are eigenvectors with eigenvalues $\lambda_1$ and $\lambda_2$, then $\mathbf{v}_1^T\Omega\mathbf{v}_2 = 0$ unless $\lambda_1 = -\lambda_2$. This structure plays an important role in determining the stable and unstable subspaces around the fixed points.

### Stable Points

The classical equations of motion are not Hamiltonian due to the presence of drive and dissipation. When we perform stability analysis around the stable fixed points the eigenvalues of the Jacobian are given either by:

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

## 1.4. Numerical Results

Now that we have an understanding of the eigenvalues and eigenvectors of the Jacobian, we can use this to study the dynamics of the system across parameter space. For this purpose we use the `generate_map.py` and `generate_jacobian_spectrum.py` scripts from [40].

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

The `generate_stability_map` function iterates over the fixed points stored in the `FixedPointMap` object created using `generate_map.py` and computes the eigenvalues and eigenvectors of the Jacobian at each fixed point. The results can then be accessed at the `FixedPointMap.eigenvalues` and `FixedPointMap.eigenvectors` attributes.

### Dim Fixed Point

Using a `FixedPointMap` produced at $\chi=-0.1$ and $\delta=7.8$, we now plot the real and imaginary parts of the $\lambda_0$ and $\lambda_1$ eigenvalues around the dim fixed point. We also plot the upper and lower bifurcation lines to clearly mark the limits of the bistable regime.

![Jacobian spectrum of dim fixed point](fixed_points/supplemental_png/jacobian_spectrum_dim_fixed_point.png)

Let's examine the key features of the eigenvalue spectrum:

1. **Complex Eigenvalues and Oscillatory Motion**: Across most of the parameter space, the eigenvalues form complex conjugate pairs. This indicates oscillatory motion near the fixed points, classifying them as focus points.

2. **Relationship to Decay Rate**: For focus points, the real component's magnitude equals the system decay rate $\kappa$. As $\kappa \to 0$, while the real component vanishes, the imaginary component persists. This suggests that oscillatory motion occurs on a much faster timescale than decay or escape processes.

3. **Behavior Near Bifurcation**: Near the dim-saddle bifurcation point, the eigenvalues become purely real, transforming the fixed point into a node. At this transition:
   - The previously equal real parts take on different values.
   - One eigenvalue approaches zero while the other remains finite.
   - The vanishing eigenvalue reveals a soft mode connecting the fixed point to the saddle point
   This regime corresponds to the one-dimensional Kramers problem discussed in [13].

These points can be illustrated in more detail by examining specific trajectories near the fixed points:

![Trajectories and bifurcation](fixed_points/supplemental_png/trajectories_and_bifurcation.png)

In the figure above we plot two classical trajectories of the system as it decays from the saddle point to the dim fixed point. In the top-left panel we choose to examine the system at small decay rate ($\kappa=0.1$, $\epsilon=10.0$) deep within the bistable regime. As mentioned above, this leads to a focus point with a decaying spiralling motion. The small value of the decay rate causes the motion to be almost circular. 

This will pose a significant challenge for us when we attempt to find escape trajectories. The eigenvectors of the Jacobian which allow escape are part of the same quadruplet as ingoing eigenvectors which control decaying motion. Just as the decaying path has significant spiralling motion with slow relaxation to the fixed point, the escaping trajectories will have significant spiralling motion with slow escape rate. The widely differing timescales and the accumulation of numerical errors will make it challenging to find the exact path going from the fixed point to the saddle point.

In the top-right panel we plot the same trajectory close to the cusp point where the bistable regime closes ($\kappa=4.3$, $\epsilon=25.6$). The motion is quite different. We now see that the dim fixed point is a node and the system moves almost linearly from one point to the other. This makes it much easier to find the exact trajectory.

We also include a table of the eigenvalues and parameters, as well as the bottom panel to show where the top panels lie within the bistable regime.

### Bright Fixed Point

Next we move on to the bright fixed point. As above we observe complex eigenvalues corresponding to focus points in the majority of the parameter space. At small decay rates the real components of the eigenvalues are close to zero and the motion is almost circular. Just as above we also see that close to the bifurcation point the imaginary components of the eigenvalues vanish and the fixed points become nodes.

![Jacobian spectrum of bright fixed point](fixed_points/supplemental_png/jacobian_spectrum_bright_fixed_point.png)


### Saddle Point

Next, we examine the eigenvalue spectrum around the saddle point. Unlike the stable fixed points, the saddle point has real eigenvalues of opposite signs throughout the bistable regime, reflecting its unstable nature. This is consistent with our earlier theoretical analysis where we predicted eigenvalues of the form $\lambda \in \{ -\kappa_1, +\kappa_2, +\kappa_1, -\kappa_2 \}$.

![Jacobian spectrum of saddle fixed point](fixed_points/supplemental_png/jacobian_spectrum_saddle_fixed_point.png)

The key features of the saddle point spectrum are:

1. **Pure Real Eigenvalues**: Unlike the stable fixed points which typically show complex conjugate pairs, the saddle point maintains real eigenvalues throughout the parameter space, indicating pure exponential growth or decay without oscillations.

2. **Symmetry in Magnitude**: The eigenvalues appear in pairs of equal magnitude but opposite sign, reflecting the Hamiltonian nature of the full system and ensuring conservation of phase space volume.

3. **Bifurcation Behavior**: Near the bifurcation points, one pair of eigenvalues approaches zero while the other pair remains finite. This corresponds to the merging of the saddle point with one of the stable fixed points at the bifurcation.


### Regimes

We note that the Jacobian spectra of both the dim and bright fixed points indicate that we will struggle to find escape trajectories at small decay rates. In the figure below we plot the ratio of the imaginary to real parts of the eigenvalues for the dim and bright fixed points. As expected, the ratio diverges as the decay rate approaches zero, but this can be counteracted to some extent by staying close to a bifurcation point.

![Instability ratios](fixed_points/supplemental_png/instability_ratios.png)

Finally we plot the ratio of the eigenvalues of the Jacobian at the saddle point to illustrate the onset of the soft mode.

![Saddle eigenvalue ratio](fixed_points/supplemental_png/saddle_eigenvalue_ratio.png)

The vanishing of the $\lambda_0$ eigenvalue at the bifurcation point is indicative of the vanishing of the force pushing the system from the saddle point to the node as they merge.