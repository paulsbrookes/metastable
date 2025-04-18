# Derivation of the Keldysh Lagrangian from a Lindblad Master Equation

This document presents a derivation of the Keldysh Lagrangian for a driven–dissipative Kerr oscillator following.

---

## 1. Starting Point: The Lindblad Master Equation

The Lindblad master equation for the density matrix evolution is given by

$$
\partial_t \rho = -i\,[H,\rho] + \kappa\Bigl(2a\,\rho\,a^\dagger - \{a^\dagger a,\rho\}\Bigr),
$$

where the Hamiltonian is

$$
H = \delta\,a^\dagger a + \chi\,a^\dagger a^\dagger aa + i\varepsilon\,(a^\dagger - a).
$$

The Lindblad operator can be expressed as

$$
L = \sqrt{2\kappa}\,a,
$$

so that the dissipative part is in the standard Lindblad form.

---

## 2. Matrix Element of the Liouvillian Superoperator

In the coherent-state representation, we work with forward ($a_+$) and backward ($a_-$) contours. Denoting the fields by $a_+$ and $a_-$ we define the matrix element of the Liouvillian superoperator as

$$
L_{\rm super}(a_+^*,a_+,a_-^*,a_-) = -i\Bigl[H(a_+^*,a_+) - H(a_-^*,a_-)\Bigr] + \kappa\Bigl[2\,a_+a_-^* - a_+^*a_+ - a_-^*a_-\Bigr].
$$

Here, the Hamiltonian evaluated on each contour is

$$
H(a_\pm^*,a_\pm) = \delta\,a_\pm^*a_\pm + \chi\,a_\pm^{*2}a_\pm^2 + i\varepsilon\,(a_\pm^* - a_\pm).
$$

---

## 3. Constructing the Keldysh Lagrangian

Following the recipe, the full Keldysh Lagrangian is written as

$$
L_K = a_+^*\,i\partial_t a_+ - a_-^*\,i\partial_t a_- - i\,L_{\rm super}(a_+^*,a_+,a_-^*,a_-).
$$

Substituting the expression for $L_{\rm super}$, we have

$$
\begin{aligned}
L_K &= a_+^*\,i\partial_t a_+ - a_-^*\,i\partial_t a_- \\
&\quad {} - i\Bigl\{-i\Bigl[H(a_+^*,a_+) - H(a_-^*,a_-)\Bigr] + \kappa\Bigl[2\,a_+a_-^* - a_+^*a_+ - a_-^*a_-\Bigr]\Bigr\} \\
&= a_+^*\,i\partial_t a_+ - a_-^*\,i\partial_t a_- + \Bigl[H(a_+^*,a_+) - H(a_-^*,a_-)\Bigr] \\
&\quad {} - i\,\kappa\Bigl[2\,a_+a_-^* - a_+^*a_+ - a_-^*a_-\Bigr].
\end{aligned}
$$

Inserting the explicit form of the Hamiltonian difference

$$
\begin{aligned}
H(a_+^*,a_+) - H(a_-^*,a_-) &= \delta\,\Bigl(a_+^*a_+ - a_-^*a_-\Bigr) + \chi\,\Bigl(a_+^{*2}a_+^2 - a_-^{*2}a_-^2\Bigr) \\
&\quad {} + i\varepsilon\Bigl[(a_+^* - a_+) - (a_-^* - a_-)\Bigr],
\end{aligned}
$$

the final form of the Keldysh Lagrangian becomes

$$
\boxed{
\begin{aligned}
L_K =\; & a_+^*\,i\partial_t a_+ - a_-^*\,i\partial_t a_- \\
& - \delta\,\Bigl(a_+^*a_+ - a_-^*a_-\Bigr) - \chi\,\Bigl(a_+^{*2}a_+^2 - a_-^{*2}a_-^2\Bigr) \\
& - i\varepsilon\,\Bigl[(a_+^* - a_+) - (a_-^* - a_-)\Bigr] \\
& - i\,\kappa\Bigl[2\,a_+a_-^* - a_+^*a_+ - a_-^*a_-\Bigr].
\end{aligned}
}
$$
