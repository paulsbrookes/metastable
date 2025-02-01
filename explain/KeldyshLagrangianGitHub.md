# Derivation of the Keldysh Lagrangian from a Lindblad Master Equation

This document presents a derivation of the Keldysh Lagrangian for a driven–dissipative Kerr oscillator following.

## 1. Starting Point: The Lindblad Master Equation

The Lindblad master equation for the density matrix evolution is given by

$\partial_t \rho = -i[H,\rho] + \kappa\left(2a\rho a^\dagger - \{a^\dagger a,\rho\}\right)$

where the Hamiltonian is

$H = \delta a^\dagger a + \chi a^\dagger a^\dagger aa + i\varepsilon(a^\dagger - a)$

The jump operator can be expressed as

$L = \sqrt{2\kappa}a$

so that the dissipative part is in the standard Lindblad form.

## 2. Matrix Element of the Liouvillian Superoperator

In the coherent-state representation, we work with forward (+) and backward (-) contours. Denoting the fields by $a_+$ and $a_-$ (with their complex conjugates $a_+^*$ and $a_-^*$), we define the matrix element of the Liouvillian superoperator as

$L_{\rm super}(a_+^*,a_+,a_-^*,a_-) = -i\left[H(a_+^*,a_+) - H(a_-^*,a_-)\right] + \kappa\left[2a_+a_-^* - a_+^*a_+ - a_-^*a_-\right]$

Here, the Hamiltonian evaluated on each contour is

$H(a_\pm^*,a_\pm) = \delta a_\pm^*a_\pm + \chi a_\pm^{*2}a_\pm^2 + i\varepsilon(a_\pm^* - a_\pm)$

## 3. Constructing the Keldysh Lagrangian

Following the recipe, the full Keldysh Lagrangian is written as

$L_K = a_+^*i\partial_t a_+ - a_-^*i\partial_t a_- - iL_{\rm super}(a_+^*,a_+,a_-^*,a_-)$

Substituting the expression for $L_{\rm super}$, we have

$\begin{aligned} 
L_K &= a_+^*i\partial_t a_+ - a_-^*i\partial_t a_- \\
&\quad {} - i\left\{-i\left[H(a_+^*,a_+) - H(a_-^*,a_-)\right] + \kappa\left[2a_+a_-^* - a_+^*a_+ - a_-^*a_-\right]\right\} \\
&= a_+^*i\partial_t a_+ - a_-^*i\partial_t a_- + \left[H(a_+^*,a_+) - H(a_-^*,a_-)\right] \\
&\quad {} - i\kappa\left[2a_+a_-^* - a_+^*a_+ - a_-^*a_-\right]
\end{aligned}$

Inserting the explicit form of the Hamiltonian difference

$\begin{aligned}
H(a_+^*,a_+) - H(a_-^*,a_-) &= \delta\left(a_+^*a_+ - a_-^*a_-\right) + \chi\left(a_+^{*2}a_+^2 - a_-^{*2}a_-^2\right) \\
&\quad {} + i\varepsilon\left[(a_+^* - a_+) - (a_-^* - a_-)\right]
\end{aligned}$

the final form of the Keldysh Lagrangian becomes

$\begin{aligned}
L_K = & a_+^*i\partial_t a_+ - a_-^*i\partial_t a_- \\
&\quad {} + \delta\left(a_+^*a_+ - a_-^*a_-\right) + \chi\left(a_+^{*2}a_+^2 - a_-^{*2}a_-^2\right) \\
&\quad {} + i\varepsilon\left[(a_+^* - a_+) - (a_-^* - a_-)\right] \\
&\quad {} - i\kappa\left[2a_+a_-^* - a_+^*a_+ - a_-^*a_-\right]
\end{aligned}$

This expression clearly separates:

- The **time-evolution** terms: $a_+^*i\partial_t a_+ - a_-^*i\partial_t a_-$
- The **coherent Hamiltonian** contributions
- The **dissipative** contributions

## 4. Summary

1. **Starting Point:** The Lindblad master equation is written in terms of the Hamiltonian and a dissipative term with jump operator $L = \sqrt{2\kappa}a$.

2. **Liouvillian Superoperator:** In the coherent-state basis with forward (+) and backward (-) contours, the superoperator is defined by 
   $L_{\rm super}(a_+^*,a_+,a_-^*,a_-) = -i\left[H(a_+^*,a_+) - H(a_-^*,a_-)\right] + \kappa\left[2a_+a_-^* - a_+^*a_+ - a_-^*a_-\right]$

3. **Keldysh Lagrangian:** The full Lagrangian is constructed as 
   $L_K = a_+^*i\partial_t a_+ - a_-^*i\partial_t a_- - iL_{\rm super}(a_+^*,a_+,a_-^*,a_-)$
   leading to the final expression above.

This derivation provides a compact path-integral description of the driven–dissipative Kerr oscillator, capturing both its unitary (Hamiltonian) dynamics and its dissipative (Lindblad) dynamics.