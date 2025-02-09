# Fixed Points of the Auxiliary Hamiltonian

In this section we explain how to map and characterise the fixed points of the auxiliary Hamiltonian. Recall that our overall goal is to study the switching dynamics of the driven–dissipative Kerr oscillator, and we wish to do this by finding the switching trajectories in the phase space of the auxiliary Hamiltonian. In order to do this we must:
1. Find the fixed points of the classical equations of motion derived from the auxiliary Hamiltonian.
2. Classify the stability of the fixed points by linearising the equations of motion around them using the Jacobian.
3. Use the fixed points and the eigenvectors of the Jacobian to set the boundary conditions for the switching trajectories.

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
\begin{align}
\dot{x}_c =& \delta p_c + 2 \epsilon + 2 \kappa x_q - \kappa x_c + \frac{1}{2} \chi (p_c^3 - 3 p_c x_q^2 + p_c x_c^2 - p_c p_q^2 +2 x_q x_c p_q) \\
\dot{p}_c =& - \delta x_c - \kappa (p_c - 2 p_q)- \frac{1}{2} \chi (p_c^2 x_c + 2 p_c x_q p_q - x_q^2 x_c + x_c^3 - 3 x_c p_q^2) \\
\dot{x}_q =& \frac{1}{2} \chi (p_q p_c^2 - p_q^3 + 3 p_q x_c^2 - p_q x_q^2 - x_c p_c x_q) + \delta p_q + \kappa x_q \\
\dot{p}_q =& \frac{1}{2} \chi (2 p_q p_c x_c + x_q p_q^2 + x_q^3 - x_q x_c^2 - 3 p_c^2 x_q) - \delta x_q + \kappa p_q
\end{align}
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

When $\Delta > 0$ there are three distinct real roots, and when $\Delta \leq 0$ there is one real root and two complex conjugate roots. The physical solutions correspond only to the real roots, as $p_c$ must be real-valued. When there are three real roots we are in the bistable regime and we can expect to find two stable fixed points and one unstable one.

---

## References

[1] "Derivation of the Auxiliary Hamiltonian from the Keldysh Lagrangian", see [KeldyshAuxiliaryHamiltonian.md](../derivation/KeldyshAuxiliaryHamiltonian.md).

[2] "NumPy roots function documentation", see [numpy.org/doc/2.2/reference/generated/numpy.roots.html](https://numpy.org/doc/2.2/reference/generated/numpy.roots.html).