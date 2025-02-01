# Switching Dynamics in a Driven Nonlinear Oscillator

In this section, we study the switching dynamics of a driven nonlinear oscillator at zero temperature in the bistable regime.

## Fixed Points in the Bistable Regime

In the bistable regime, the classical equations of motion exhibit three fixed points: an unstable saddle point and two stable nodes. The nodes are surrounded by basins of attraction, and the saddle point lies on the separatrix which divides them.

## Transition Mechanisms Between Nodes

Let's consider three methods by which the system may transition from one node to another:

### 1. Thermal Activation

In systems at finite temperature, thermal fluctuations from an external bath can provide the energy needed to overcome the potential barrier separating the two stable states. The transition rate follows Kramers' law, with an exponential dependence on the ratio of the barrier height to the thermal energy ($k_BT$) according to the Boltzmann distribution. This mechanism dominates at high temperatures, but here we will only consider the zero temperature case.

### 2. Quantum Tunneling

At very low temperatures, quantum tunneling through the potential barrier may occur. While this process exists even at zero temperature, it becomes exponentially suppressed as the barrier height or width increases. The tunneling rate takes the form $\exp(-2S_{\text{tun}}/\lambda)$, where $S_{\text{tun}}$ is the tunneling action.

### 3. Quantum Activation

In driven dissipative systems far from equilibrium, a distinct mechanism called quantum activation becomes dominant. This process arises from the quantum noise that accompanies relaxation due to coupling with a thermal bath. Unlike tunneling, quantum activation involves transitions over the effective barrier through these quantum fluctuations, even at zero temperature. The switching rate takes the form $W_{\text{sw}} \propto \exp(-R_A/\lambda)$, where $R_A$ is the effective activation energy and $\lambda$ is the effective Planck constant in the rotating frame. This process has no analog in equilibrium systems and yields exponentially larger switching rates compared to tunneling, making it the dominant switching mechanism unless the relaxation rate is exponentially small.