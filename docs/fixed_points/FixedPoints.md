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
\partial_t x_c =& \delta p_c + 2 \epsilon + 2 \kappa x_q - \kappa x_c + \frac{1}{2} \chi (p_c^3 - 3 p_c x_q^2 + p_c x_c^2 - p_c p_q^2 +2 x_q x_c p_q) \\
\partial_t p_c =& - \delta x_c - \kappa (p_c - 2 p_q)- \frac{1}{2} \chi (p_c^2 x_c + 2 p_c x_q p_q - x_q^2 x_c + x_c^3 - 3 x_c p_q^2) \\
\partial_t x_q =& \frac{1}{2} \chi (p_q p_c^2 - p_q^3 + 3 p_q x_c^2 - p_q x_q^2 - x_c p_c x_q) + \delta p_q + \kappa x_q \\
\partial_t p_q =& \frac{1}{2} \chi (2 p_q p_c x_c + x_q p_q^2 + x_q^3 - x_q x_c^2 - 3 p_c^2 x_q) - \delta x_q + \kappa p_q
\end{align}
$$

---

## 2. Fixed Point Conditions

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


---

## 3. Summary

By starting from the auxiliary Hamiltonian derived via a deformation of the quantum fields in the Keldysh formalism, we have obtained the fixed point conditions for the classical dynamics:

$$
\begin{aligned}
p_c\left(\delta + \frac{\chi}{2}(x_c^2+p_c^2)\right) - \kappa\,x_c + 2\varepsilon &= 0,\\[1mm]
-x_c\left(\delta + \frac{\chi}{2}(x_c^2+p_c^2)\right) - \kappa\,p_c &= 0.
\end{aligned}
$$

These equations serve as the starting point for further investigations into metastability and switching phenomena in driven–dissipative Kerr oscillators.

---

## References

[1] "Derivation of the Auxiliary Hamiltonian from the Keldysh Lagrangian", see [KeldyshAuxiliaryHamiltonian.md](../derivation/KeldyshAuxiliaryHamiltonian.md).