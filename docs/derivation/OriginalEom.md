# Derivation of the Equations of Motion

In this document we explain how to derive the equations of motion from the Lagrangian

$$
\begin{aligned}
L &= \dot{x}_c\,\tilde{p}_q - \dot{p}_c\,\tilde{x}_q - \Biggl\{\delta  + \frac{\chi}{2} \Bigl(x_c^2 + p_c^2 + \tilde{x}_q^2 + \tilde{p}_q^2 \Bigr) \Biggr\}(x_c\,\tilde{x}_q + p_c\,\tilde{p}_q) \\
&\quad + \kappa (x_c\,\tilde{p}_q - p_c\,\tilde{x}_q)  + i \kappa (\tilde{x}_q^2 + \tilde{p}_q^2) + 2 \varepsilon\,\tilde{p}_q\,.
\end{aligned}
$$

## 1. Recognizing the Canonical Structure

Notice that the Lagrangian is *first order* in time derivatives, with the kinetic (or "symplectic") part given by

$$
L_{\rm kin} = \dot{x}_c\,\tilde{p}_q - \dot{p}_c\,\tilde{x}_q\,.
$$

This form identifies the canonical pairs as:
- $x_c$ with conjugate momentum $\tilde{p}_q$, and
- $p_c$ with conjugate momentum $-\tilde{x}_q$.

Thus, the Lagrangian can be rewritten as

$$
L = \tilde{p}_q\,\dot{x}_c - \tilde{x}_q\,\dot{p}_c - H(x_c, p_c, \tilde{x}_q, \tilde{p}_q)\,,
$$

where the "Hamiltonian" is given by

$$
\begin{aligned}
H(x_c, p_c, \tilde{x}_q, \tilde{p}_q) = &\; \Biggl[\delta + \frac{\chi}{2} \Bigl(x_c^2+p_c^2+\tilde{x}_q^2+\tilde{p}_q^2\Bigr)\Biggr](x_c\,\tilde{x}_q+p_c\,\tilde{p}_q) \\
&\; -\kappa(x_c\,\tilde{p}_q-p_c\,\tilde{x}_q) - i\kappa(\tilde{x}_q^2+\tilde{p}_q^2) - 2\varepsilon\,\tilde{p}_q\,.
\end{aligned}
$$

## 2. Writing Hamilton's Equations

Since the kinetic term is in canonical form, the Euler–Lagrange equations for our variables are equivalent to Hamilton's equations. In particular, we have

$$
\begin{aligned}
\dot{x}_c &= \frac{\partial H}{\partial \tilde{p}_q}\,, &\quad \dot{\tilde{p}}_q &= -\frac{\partial H}{\partial x_c}\,,\\[1mm]
\dot{p}_c &= -\frac{\partial H}{\partial \tilde{x}_q}\,, &\quad \dot{\tilde{x}}_q &= \frac{\partial H}{\partial p_c}\,.
\end{aligned}
$$

## 3. Computing the Partial Derivatives

To compute these derivatives efficiently, one may introduce the shorthand

$$
A \equiv \delta+\frac{\chi}{2}\Bigl(x_c^2+p_c^2+\tilde{x}_q^2+\tilde{p}_q^2\Bigr)
\quad \text{and} \quad
B \equiv x_c\,\tilde{x}_q+p_c\,\tilde{p}_q\,.
$$

### Derivative with Respect to $\tilde{p}_q$:

$$
\frac{\partial H}{\partial \tilde{p}_q}
= \chi\,\tilde{p}_q\,B + A\,p_c - \kappa\,x_c - 2i\kappa\,\tilde{p}_q - 2\varepsilon\,.
$$

Thus, the equation of motion for $x_c$ is

$$
\dot{x}_c = \chi\,\tilde{p}_q\,B + A\,p_c - \kappa\,x_c - 2i\kappa\,\tilde{p}_q - 2\varepsilon\,.
$$

### Derivative with Respect to $x_c$:

$$
\frac{\partial H}{\partial x_c}
= \chi\,x_c\,B + A\,\tilde{x}_q - \kappa\,\tilde{p}_q\,.
$$

Thus, the equation of motion for $\tilde{p}_q$ is

$$
\dot{\tilde{p}}_q = -\chi\,x_c\,B - A\,\tilde{x}_q + \kappa\,\tilde{p}_q\,.
$$

### Derivative with Respect to $\tilde{x}_q$:

$$
\frac{\partial H}{\partial \tilde{x}_q}
= \chi\,\tilde{x}_q\,B + A\,x_c + \kappa\,p_c - 2i\kappa\,\tilde{x}_q\,.
$$

Thus, the equation of motion for $p_c$ is

$$
\dot{p}_c = -\chi\,\tilde{x}_q\,B - A\,x_c - \kappa\,p_c + 2i\kappa\,\tilde{x}_q\,.
$$

### Derivative with Respect to $p_c$:

$$
\frac{\partial H}{\partial p_c}
= \chi\,p_c\,B + A\,\tilde{p}_q + \kappa\,\tilde{x}_q\,.
$$

Thus, the equation of motion for $\tilde{x}_q$ is

$$
\dot{\tilde{x}}_q = \chi\,p_c\,B + A\,\tilde{p}_q + \kappa\,\tilde{x}_q\,.
$$

## 4. Final Set of Equations

Summarizing, the equations of motion are

$$
\boxed{
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
}
$$
