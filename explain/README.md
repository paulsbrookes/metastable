# Switching Dynamics in a Driven Nonlinear Oscillator

Here we study the switching dynamics of a driven nonlinear oscillator at zero temperature in the bistable regime. Switching refers to the process by which the system transitions from one metastable state to another. We will review three methods below, before focusing on quantum activation, which will study in more depth using an instanton approach based on Keldysh field theory [1].

## Fixed Points in the Bistable Regime

In the bistable regime, the classical equations of motion exhibit three fixed points: an unstable saddle point and two stable nodes. The nodes are surrounded by basins of attraction, and the saddle point lies on the line in phase space which divides them, known as the separatrix.

## Transition Mechanisms Between Nodes

Let's review three methods by which the system may transition from one node to another:

### 1. Thermal Activation

In systems at finite temperature, thermal fluctuations from an external bath can provide the energy needed to overcome the potential barrier separating the two stable states. The transition rate follows Kramers' law, with an exponential dependence on the ratio of the barrier height to the thermal energy ($k_BT$) according to the Boltzmann distribution [2]. This mechanism dominates at high temperatures, but here we will only consider the zero temperature case where this process vanishes.

### 2. Quantum Tunneling

At very low temperatures, quantum tunneling through the potential barrier may occur. While this process exists even at zero temperature, it becomes exponentially suppressed as the barrier height or width increases [3]. The tunneling rate takes the form $\exp(-2S_{\text{tun}}/\lambda)$, where $S_{\text{tun}}$ is the tunneling action.

### 3. Quantum Activation

In driven dissipative systems far from equilibrium, a distinct mechanism called quantum activation becomes dominant [4, 5]. This process arises from the quantum noise that accompanies relaxation due to coupling with a thermal bath. Unlike tunneling, quantum activation involves transitions over the effective barrier through these quantum fluctuations, even at zero temperature [6, 7]. The switching rate takes the form $W_{\text{sw}} \propto \exp(-R_A/\lambda)$, where $R_A$ is the effective activation energy and $\lambda$ is the effective Planck constant in the rotating frame. This process has no analog in equilibrium systems and yields exponentially larger switching rates compared to tunneling, making it the dominant switching mechanism unless the relaxation rate is exponentially small.

## References

[1] A. Kamenev, "Field Theory of Non-Equilibrium Systems" (Cambridge University Press, 2011).

[2] P. Hänggi, P. Talkner, and M. Borkovec, "Reaction-rate theory: fifty years after kramers," Reviews of modern physics 62, 251 (1990).

[3] W. H. Miller, "Semiclassical limit of quantum mechanical transition state theory for nonseparable systems," The Journal of chemical physics 62, 1899 (1975).

[4] M. Dykman and V. Smelyanskij, "Quantum theory of transitions between stable states of nonlinear oscillator, interacting with medium in a resonance field," Sov. Phys. JETP 67, 1769 (1988).

[5] M. Marthaler and M. Dykman, "Switching via quantum activation: A parametrically modulated oscillator," Physical Review A 73, 042108 (2006).

[6] M. Dykman, "Critical exponents in metastable decay via quantum activation," Physical Review E—Statistical, Nonlinear, and Soft Matter Physics 75, 011101 (2007).

[7] M. Dykman, "Fluctuating Nonlinear Oscillators: FromNanomechanics to Quantum Superconducting Circuits" (Oxford University Press, 2012).