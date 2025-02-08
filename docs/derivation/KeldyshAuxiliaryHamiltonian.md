# Derivation of the Auxiliary Hamiltonian from the Keldysh Lagrangian

This document presents a derivation of the auxiliary Hamiltonian for a driven–dissipative Kerr oscillator, starting from the Keldysh Lagrangian.

---

## 1. The Keldysh Lagrangian

We start from the Lagrangian derived in [Keldysh Lagrangian](KeldyshLagrangian.md). The system is described in terms of the forward ($a_+$) and backward ($a_-$) fields as follows:

$$
\begin{aligned}
L &= a_{+}^{*} \,i\partial_t a_{+} - a_{-}^{*}\, i\partial_t a_{-}
+ i\varepsilon \Bigl( a_{+}^{*} - a_{+} - a_{-}^{*} + a_{-} \Bigr)\\[1mm]
&\quad - \delta\Bigl(a_{+}^{*}a_{+} - a_{-}^{*}a_{-}\Bigr)
- \chi\Bigl(a_{+}^{*2}a_{+}^2 - a_{-}^{*2}a_{-}^2\Bigr)\\[1mm]
&\quad - i\kappa\Bigl( 2a_{+}a_{-}^{*} - a_{+}^{*}a_{+} - a_{-}^{*}a_{-}\Bigr)
\end{aligned}
$$

and the partition function is given by

$$
\mathcal{Z} = \int \mathcal{D}a_{-}\,\mathcal{D}a_{-}^{*}\,\mathcal{D}a_{+}\,\mathcal{D}a_{+}^{*}\;e^{iS[a_+,a_+^*,a_-,a_-^*]}.
$$

The action is defined as the time-integral of the Lagrangian:

$$
S[a_+,a_+^*,a_-,a_-^*] = \int_{t_i}^{t_f} dt\, L(a_+,a_+^*,a_-,a_-^*),
$$

where $t_i$ and $t_f$ are the initial and final times of the evolution.

Our overall goal is to calculate the rate at which the system escapes from metastable states, and we are attempting to do this via finding the instanton (saddle-point) solutions to the equation of motion which dominate the partition function.

Below we rewrite the Lagrangian in terms of classical and quantum fields and show the equations of motion for these quantities. We then argue that the instanton solutions should be obtained by deforming the integration contour for the quantum fields along the imaginary axis. This results in new equations of motion which can be obtained from an auxiliary Hamiltonian.

---

## 2. Keldysh Rotation and Decomposition into Real Fields

We introduce the **classical** and **quantum** fields by performing the Keldysh rotation:

$$
a_c = \frac{a_+ + a_-}{\sqrt{2}},\qquad
a_q = \frac{a_+ - a_-}{\sqrt{2}}.
$$

In general these are complex fields and we can express them in terms of real components as follows:

$$
a_c = \frac{x_c + i\,p_c}{\sqrt{2}},\qquad
a_q = \frac{\tilde{x}_q + i\,\tilde{p}_q}{\sqrt{2}}.
$$

In these variables, after some algebra (and up to total time derivatives), the Lagrangian takes the form:

$$
\begin{aligned}
L &= \dot{x}_c\,\tilde{p}_q - \dot{p}_c\,\tilde{x}_q -\Biggl[\delta + \frac{\chi}{2}\Bigl(x_c^2+p_c^2+\tilde{x}_q^2+\tilde{p}_q^2\Bigr)\Biggr]\,(x_c\,\tilde{x}_q+p_c\,\tilde{p}_q) \\
&\quad + \kappa\,(x_c\,\tilde{p}_q-p_c\,\tilde{x}_q) + i\kappa\,(\tilde{x}_q^2+\tilde{p}_q^2) + 2\varepsilon\,\tilde{p}_q.
\end{aligned}
$$

In these new variables, the partition function becomes

$$
\mathcal{Z} = \int \mathcal{D}x_c\,\mathcal{D}p_c\,\mathcal{D}\tilde{x}_q\,\mathcal{D}\tilde{p}_q \; e^{i\,S[x_c,p_c,\tilde{x}_q,\tilde{p}_q]},
$$

with the action given by

$$
S[x_c,p_c,\tilde{x}_q,\tilde{p}_q] = \int_{t_i}^{t_f} dt\, L(x_c,p_c,\tilde{x}_q,\tilde{p}_q).
$$


Finally, as derived in [Original Equations of Motion](OriginalEom.md), the equations of motion are given by

$$
\begin{aligned}
\dot{x}_c &= \chi\,\tilde{p}_q\,(x_c\,\tilde{x}_q+p_c\,\tilde{p}_q)
+\left(\delta+\frac{\chi}{2}(x_c^2+p_c^2+\tilde{x}_q^2+\tilde{p}_q^2)\right)p_c
-\kappa\,x_c-2i\kappa\,\tilde{p}_q-2\varepsilon\,,\\[1mm]
\dot{\tilde{p}}_q &= -\chi\,x_c\,(x_c\,\tilde{x}_q+p_c\,\tilde{p}_q)
-\left(\delta+\frac{\chi}{2}(x_c^2+p_c^2+\tilde{x}_q^2+\tilde{p}_q^2)\right)\tilde{x}_q
+\kappa\,\tilde{p}_q\,,\\[1mm]
\dot{p}_c &= -\chi\,\tilde{x}_q\,(x_c\,\tilde{x}_q+p_c\,\tilde{p}_q)
-\left(\delta+\frac{\chi}{2}(x_c^2+p_c^2+\tilde{x}_q^2+\tilde{p}_q^2)\right)x_c
-\kappa\,p_c+2i\kappa\,\tilde{x}_q\,,\\[1mm]
\dot{\tilde{x}}_q &= \chi\,p_c\,(x_c\,\tilde{x}_q+p_c\,\tilde{p}_q)
+\left(\delta+\frac{\chi}{2}(x_c^2+p_c^2+\tilde{x}_q^2+\tilde{p}_q^2)\right)\tilde{p}_q
+\kappa\,\tilde{x}_q\,.
\end{aligned}
$$

---

## 3. Transformation of the Quantum Fields

In order to capture instanton trajectories connecting metastable states we wish to find solutions to the equations of motion above. Under examination we should see that if $x_c$ and $p_c$ are both real, then $\tilde{x}_q$ and $\tilde{p}_q$ must be purely imaginary. Since $\tilde{x}_q$ and $\tilde{p}_q$ supposed to also be real, this would restrict us to the classical solutions, i.e. $\tilde{x}_q$ and $\tilde{p}_q$ are both zero. These classical solutions will not permit any switching and will only show relaxation towards the fixed points.

However, the equations of motion do indeed have solutions with imaginary $\tilde{x}_q$ and $\tilde{p}_q$ and we can make use of them if we deform the integration contours of the quantum fields along the imaginary axis.

$$
\tilde{x}_q \to -i\,p_q,\qquad \tilde{p}_q \to i\,x_q,
$$

This is permitted by Cauchy's theorem provided that the integrand decays sufficiently rapidly at infinity and that no singularities are crossed by the contour shift.


@TODO: Reference Kamenev's book for the transformation.

Applying this transformation to the Lagrangian and collecting terms we get

$$
\begin{aligned}
L \to \; & i\Bigl\{ \dot{x}_c\,x_q + \dot{p}_c\,p_q - \Bigl[\delta + \frac{\chi}{2}\bigl(x_c^2+p_c^2-x_q^2-p_q^2\bigr)\Bigr](p_c\,x_q-x_c\,p_q)\\[1mm]
& \quad + \kappa\,(x_c\,x_q+p_c\,p_q)
- \kappa\,(x_q^2+p_q^2)
- 2\varepsilon\,x_q \Bigr\}.
\end{aligned}
$$


## 4. The Auxiliary Hamiltonian

Using this transformed Lagrangian we can write a transformed action as

$$
\begin{aligned}
S_{\mathrm{aux}} &= iS \\
&= i\int dt\,L \\
&= -\int dt\Bigl[\dot{x}_c\,p_c + \dot{p}_c\,p_q - H(x_c,p_c,x_q,p_q)\Bigr].
\end{aligned}
$$

Where we have defined an auxiliary Hamiltonian $H(x_c,p_c,x_q,p_q)$ as

$$
\boxed{
\begin{aligned}
H(x_c,p_c,x_q,p_q)=\; & \Biggl(\delta + \frac{\chi}{2}\bigl(x_c^2+p_c^2-x_q^2-p_q^2\bigr)\Biggr)(p_c\,x_q - x_c\,p_q)\\[1mm]
&\quad - \kappa\,(x_c\,x_q+p_c\,p_q)
+\kappa\,(x_q^2+p_q^2)
+2\varepsilon\,x_q.
\end{aligned}
}
$$

Meanwhile the partition function is given by

$$
\mathcal{Z} = \int \mathcal{D}x_c\,\mathcal{D}p_c\,\mathcal{D}x_q\,\mathcal{D}p_q \; e^{S_{\mathrm{aux}}[x_c,p_c,x_q,p_q]},
$$


Finally, to write down the equations of motion we use the notation

$$
\mathbf{z}_c=(x_c,p_c),\quad \mathbf{z}_q=(x_q,p_q),
$$

in terms of which the Hamiltonian formulation gives us

$$
\dot{\mathbf{z}}_c = \frac{\partial H}{\partial \mathbf{z}_q},\qquad
\dot{\mathbf{z}}_q = -\frac{\partial H}{\partial \mathbf{z}_c}.
$$
