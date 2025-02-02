# Why Are $x_c$ and $p_c$ Real Whereas the Quantum Fields are Purely Imaginary?

In many formulations—particularly in path-integral or response-function approaches—the degrees of freedom are split into two sets:
- **Classical (physical) fields:** $x_c$ and $p_c$
- **Quantum (response) fields:** $\tilde{x}_q$ and $\tilde{p}_q$

Below we explain why this separation leads to $x_c$ and $p_c$ being real, while the quantum fields are taken to be purely imaginary.

## 1. Physical Versus Response Variables

- **$x_c$ and $p_c$ as Physical Variables:**  
  These fields represent the actual observable degrees of freedom in the system (for example, the position and momentum in a classical limit). Since physical observables are real quantities, $x_c$ and $p_c$ are chosen to be real.

- **$\tilde{x}_q$ and $\tilde{p}_q$ as Response Fields:**  
  These fields are introduced as auxiliary variables in formulations like the Martin–Siggia–Rose or Keldysh techniques. Their role is to enforce the equations of motion, incorporate fluctuations, and ensure proper causality in the theory. To achieve these aims, the response fields are often rotated into the imaginary axis (or equivalently, are defined as purely imaginary).

## 2. The Kinetic Term and Canonical Pairing

Examine the kinetic (first-order) part of the Lagrangian:

$$
L_{\rm kin} = \dot{x}_c\,\tilde{p}_q - \dot{p}_c\,\tilde{x}_q\,.
$$

This term is reminiscent of the canonical form $p\,\dot{q}$. Here:
- $x_c$ is paired with $\tilde{p}_q$,
- $p_c$ is paired with $-\tilde{x}_q$.

For the overall structure to be consistent (especially when writing the action $S = \int dt\, L$ in the exponential $e^{iS}$), it is convenient to choose the response fields such that when one "pulls out" an extra factor of $i$, the contributions from these terms yield a real and convergent path-integral weight.

## 3. Convergence and Causality in the Path-Integral

- **Path-Integral Convergence:**  
  In the path-integral formulation, the integration over the classical (physical) fields is performed along the real axis. If the response fields were also real, the resulting action might lead to oscillatory integrals that are hard to control. By rotating the integration contour for $\tilde{x}_q$ and $\tilde{p}_q$ into the imaginary direction (or by defining them as purely imaginary), one can improve the convergence properties of the path integral.

- **Causality and the Response Function:**  
  The response fields play a key role in enforcing causality (e.g., ensuring that the response to a perturbation vanishes for times before the perturbation). The imaginary nature of these fields is often linked to the requirement that the response functions have the correct analytic properties, which is essential in many nonequilibrium and dissipative systems.

## 4. Summary

- **Physical Fields ($x_c$ and $p_c$) Are Real:**  
  They represent observable quantities and must be real to reflect measurable physical properties.

- **Quantum/Response Fields ($\tilde{x}_q$ and $\tilde{p}_q$) Are Imaginary:**  
  Their imaginary nature is a consequence of:
  - Ensuring that the overall action has the correct convergence properties in the path-integral formulation.
  - Maintaining the proper causal structure of the response functions.
  - Allowing for a canonical formulation where the kinetic term naturally pairs a real physical field with an imaginary conjugate field (or, equivalently, by redefining the quantum fields with an extra factor of $i$).

This separation of reality properties is not arbitrary—it is a built-in feature of the theoretical framework designed to make the mathematical formulation both well-behaved and physically meaningful.
