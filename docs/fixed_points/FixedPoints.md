# Fixed Points of the Auxiliary Hamiltonian

In this section we explain how to map and characterise the fixed points of the auxiliary Hamiltonian. Recall that our overall goal is to study the switching dynamics of the driven–dissipative Kerr oscillator, and we wish to do this by finding the switching trajectories in the phase space of the auxiliary Hamiltonian. In order to do this we must:
- Find the fixed points of the classical equations of motion derived from the auxiliary Hamiltonian.
- Classify the stability of these fixed points by linearising the classical equations of motion around them using the Jacobian.

After these steps have been completed will need to move back to the full equations of motion including the quantum fields. This will be dealt with in the next section.

---

## 1. Equations of Motion

The auxiliary Hamiltonian was derived earlier in [Keldysh Auxiliary Hamiltonian](../derivation/KeldyshAuxiliaryHamiltonian.md) after deforming the integration contours of the quantum fields. The full dynamics of the system are governed by Hamilton's equations expressed in terms of the classical and quantum fields. Defining

$$
\mathbf{z}_c = (x_c, p_c),\qquad \mathbf{z}_q = (x_q, p_q),
$$

The equations of motion can be written as

$$
\dot{\mathbf{z}}_c = \frac{\partial H}{\partial \mathbf{z}_q},\qquad
\dot{\mathbf{z}}_q = -\frac{\partial H}{\partial \mathbf{z}_c},
$$

with the auxiliary Hamiltonian given by

$$
\begin{aligned}
H(x_c,p_c,x_q,p_q)=\; & \Biggl(\delta + \frac{\chi}{2}\bigl(x_c^2+p_c^2-x_q^2-p_q^2\bigr)\Biggr)(p_c\,x_q - x_c\,p_q)\\[1mm]
&\quad - \kappa\,(x_c\,x_q+p_c\,p_q)
+\kappa\,(x_q^2+p_q^2)
+2\varepsilon\,x_q.
\end{aligned}
$$

The equations of motion are then given by

$$
\begin{aligned}
\dot{x}_c =& \delta p_c + 2 \epsilon + 2 \kappa x_q - \kappa x_c + \frac{1}{2} \chi (p_c^3 - 3 p_c x_q^2 + p_c x_c^2 - p_c p_q^2 +2 x_q x_c p_q) \\
\dot{p}_c =& - \delta x_c - \kappa (p_c - 2 p_q)- \frac{1}{2} \chi (p_c^2 x_c + 2 p_c x_q p_q - x_q^2 x_c + x_c^3 - 3 x_c p_q^2) \\
\dot{x}_q =& \frac{1}{2} \chi (p_q p_c^2 - p_q^3 + 3 p_q x_c^2 - p_q x_q^2 - x_c p_c x_q) + \delta p_q + \kappa x_q \\
\dot{p}_q =& \frac{1}{2} \chi (2 p_q p_c x_c + x_q p_q^2 + x_q^3 - x_q x_c^2 - 3 p_c^2 x_q) - \delta x_q + \kappa p_q
\end{aligned}
$$

---

## 2. Fixed Point Conditions

### 2.1. General Case

The metastable states of the system correspond to the fixed points of the classical dynamics. To obtain them we now set the quantum fields to zero:

$$
x_q = 0,\qquad p_q = 0.
$$

This gives us the classical equations of motion:

$$
\begin{aligned}
\dot{x}_c &= \frac{\chi}{2}p_c(p_c^2 + x_c^2) + \delta p_c + 2\varepsilon - \kappa x_c, \\
\dot{p}_c &= -\frac{\chi}{2}x_c(p_c^2 + x_c^2) - \delta x_c - \kappa p_c.
\end{aligned}
$$

If we now set $\dot{x}_c = \dot{p}_c = 0$ we get coupled cubic equations for $x_c$ and $p_c$. Solving these is a non-trivial task. In general we may not know how many real solutions exist and even if we do, it is not clear how to find them. Root-finding algorithms can be used to find the solutions, but they may fail to converge to all the desired solutions, especially if the solutions are close to each other such as at a saddle-node bifurcation, or if we don't start from a good initial guess.


### 2.2. Zero Damping
At first we can simplify the problem significantly by setting $\kappa = 0$. When $\kappa = 0$, the fixed point conditions become:

$$
\begin{aligned}
p_c\left(\delta + \frac{\chi}{2}(x_c^2+p_c^2)\right) + 2\varepsilon &= 0,\\[1mm]
-x_c\left(\delta + \frac{\chi}{2}(x_c^2+p_c^2)\right) &= 0.
\end{aligned}
$$

From the second equation we can see that either $x_c = 0$ or $\delta + \frac{\chi}{2}(x_c^2+p_c^2) = 0$, but to solve both equations simultaneously we require $x_c = 0$. Substituting this into the first equation gives us a cubic equation for $p_c$:

$$
\frac{\chi}{2}p_c^3 + \delta p_c + 2\varepsilon = 0.
$$

This cubic equation can be reliably solved either analytically or numerically. In our implementation we use NumPy's `roots` function [2], which finds the roots by computing eigenvalues of the companion matrix of the polynomial. This method is both efficient and numerically stable, able to find all roots simultaneously without requiring initial guesses.

The coefficients of the cubic equation in standard form $ax^3 + bx^2 + cx + d = 0$ are:

$$
\begin{aligned}
a &= \chi / 2, \\
b &= 0, \\
c &= \delta, \\
d &= 2\varepsilon.
\end{aligned}
$$

For this cubic equation, the discriminant $\Delta$ determines the number of real roots:

$$
\begin{aligned}
\Delta &= 18abcd - 4b^3d + b^2c^2 - 4ac^3 - 27a^2d^2 \\
      &= -2\chi\delta^3 - 27\chi^2\varepsilon^2.
\end{aligned}
$$

When $\Delta > 0$ there are three distinct real roots, and when $\Delta < 0$ there is one real root and two complex conjugate roots. Meanwhile when $\Delta = 0$ all three roots are real and at least two of them are equal. The physical solutions correspond only to real roots and when there are three real roots we are in the bistable regime, in which we can expect to find two stable fixed points and one unstable one.

### 2.3. Stability Analysis



---

## 3. Mapping the Fixed Points

### 3.1 Numerical Continuation to Non-Zero Damping

We now have a method for finding the fixed points at zero damping. But to proceed we need to extend this to the general case. We can do this by using the method of numerical continuation. Given a solution at zero damping, we can use it as an initial guess to find the solution for a small but non-zero damping rate using an appropriate root-finding algorithm.

In our case we use the SciPy implementation of Powell's hybrid method `scipy.optimize.root` with `method='hybr'` [3], which is based on a modified Powell hybrid algorithm. This suits our problem well because it works even in the multi-dimensional case, and has good convergence properties even when the initial guess is far from the solution.

### 3.2. The `generate_fixed_point_map` Function

Bringing together the zero damping solution with method of numerical continuation, we have written the `metastable.generate_fixed_point_map` function, which creates a map of the fixed points across a two-dimensional parameter space of $\epsilon$ and $\kappa$, starting from zero damping.

**Initialisation**:

To initialise the problem we first fix values of $\delta$ and $\chi$. We then create an empty map of fixed points over a grid of $(\epsilon, \kappa)$ values. This grid ranges from $(0,0)$ to $(\epsilon_{\text{max}}, \kappa_{\text{max}})$ and uses $N_\epsilon \times N_\kappa$ points for resolution.

**Algorithm Steps**:
1. First we [find all three fixed points at zero damping](https://github.com/paulsbrookes/metastable/blob/d37e1c1a32d7eb25d01ae4ad7505b8497070639f/src/metastable/generate_fixed_point_map.py#L54-L58) ($\kappa=0$) and a small but finite drive strength ($\epsilon>0$) using the analytical solution of the cubic equation for $p_c$ above .
2. We then add these initial solutions to the map.
3. We employ numerical continuation to extend the known solutions to neighbouring points in parameter space.
4. We repeat step 3 until the entire parameter space has been covered.

Here's an example usage that explores a region where the system exhibits bistability, which you can download from [generate_map.py](examples/generate_map.py):

```python
from metastable.generate_fixed_point_map import generate_fixed_point_map, FixedPointMap

map: FixedPointMap = generate_fixed_point_map(
    epsilon_max=30.0,     # Maximum drive strength
    kappa_max=5.0,        # Maximum damping rate
    epsilon_points=601,   # Number of points along ε axis
    kappa_points=401,     # Number of points along κ axis
    delta=7.8,            # Detuning parameter
    chi=-0.1,             # Nonlinearity parameter
    max_workers=20        # Number of parallel processes
)

map.save(Path("map.npz")) # Save the map to a file for future use
```

The `FixedPointMap` returned by this function will contain the fixed points across the entire parameter space being studied, which we can now visualise. For example, if we wish to plot the occupation of the oscillator as a function of the drive strength and damping rate, we take

$$
n = \lvert a_c \rvert^2.
$$

and from [1] we find

$$
n = \frac{x_c^2 + p_c^2}{2}
$$

Now if we process the map saved above with [visualise_occupations.py](examples/visualise_occupations.py) we get the following plot:

<div class="plotly-container" style="position: relative; width: 100%; height: 600px; margin: 0 auto;">
    <iframe src="examples/occupations.html" 
            style="position: absolute; width: 100%; height: 100%; border: none;"
            allowfullscreen>
    </iframe>
</div>

[Open visualization in new window](examples/occupations.html)

We see three distinct regions in the parameter space, corresponding to the three possible fixed points of the system:

1. **Dim State**: Shows low occupation numbers (dark blue), representing the lower stable fixed point of the system.
2. **Bright State**: Exhibits high occupation numbers (yellow/orange), representing the upper stable fixed point.
3. **Saddle Point**: Displays intermediate occupation numbers (purple/pink), representing the unstable fixed point that separates the two stable states.

The bistable region is where all three fixed points coexist. This region is bounded by bifurcation lines where stable and unstable fixed points merge and disappear (saddle-node bifurcations).

## References

[1] "Derivation of the Auxiliary Hamiltonian from the Keldysh Lagrangian", see [KeldyshAuxiliaryHamiltonian.md](../derivation/KeldyshAuxiliaryHamiltonian.md).

[2] "NumPy roots function documentation", see [numpy.org/doc/2.2/reference/generated/numpy.roots.html](https://numpy.org/doc/2.2/reference/generated/numpy.roots.html).

[3] "SciPy Powell's Hybrid Method Documentation", see [scipy.optimize.root-hybr](https://docs.scipy.org/doc/scipy/reference/optimize.root-hybr.html). This method combines the advantages of quasi-Newton methods and modified Powell updates for solving nonlinear systems of equations.

## Interactive Visualization

