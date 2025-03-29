# Switching Dynamics in a Driven Nonlinear Oscillator

Here we study the switching dynamics of a driven nonlinear oscillator at zero temperature in the bistable regime. Switching refers to the process by which the system transitions from one metastable state to another, and below we will review three methods by which this may occur, before placing our focu on quantum activation. We will study this approach in more depth using an instanton approach based on Keldysh field theory [1].

## System Description

We study a driven-dissipative Kerr oscillator, described by the Hamiltonian:

$$
H = \delta\,a^\dagger a + \chi\,a^\dagger a^\dagger aa + i\varepsilon\,(a^\dagger - a)
$$

where:
- $\delta$ is the detuning between the drive and oscillator frequency
- $\chi$ is the Kerr nonlinearity strength
- $\varepsilon$ is the drive strength
- $a$ and $a^\dagger$ are the annihilation and creation operators

The system is coupled to a zero-temperature bath with decay rate $\kappa$, leading to dissipation described by the Lindblad operator $L = \sqrt{2\kappa}\,a$.

## Fixed Points in the Bistable Regime

In the bistable regime, the classical equations of motion exhibit three fixed points: an unstable saddle point and two stable nodes. The nodes are surrounded by basins of attraction, and the saddle point lies on the line in phase space which divides them, known as the separatrix.

### Parameter Space

The system exhibits bistability in a specific region of the $(\kappa, \epsilon, \delta)$ parameter space, where:
- $\kappa$ is the decay rate
- $\epsilon$ is the drive strength
- $\delta$ is the detuning between the drive and oscillator frequency

The bistable region is bounded by saddle-node bifurcations, where stable and unstable fixed points merge. Within this region, we find:
- Two stable fixed points: a dim state (low amplitude) and bright state (high amplitude)
- One unstable saddle point separating them

## Transition Mechanisms Between Nodes

Let's review three methods by which the system may transition from one node to another:

### 1. Thermal Activation

In systems at finite temperature, thermal fluctuations from an external bath can provide the energy needed to overcome the potential barrier separating the two stable states. The transition rate follows Kramers' law, with an exponential dependence on the ratio of the barrier height to the thermal energy ($k_BT$) according to the Boltzmann distribution [2]. This mechanism dominates at high temperatures, but here we will only consider the zero temperature case where this process vanishes.

### 2. Quantum Tunneling

At very low temperatures, quantum tunneling through the potential barrier may occur. While this process exists even at zero temperature, it becomes exponentially suppressed as the barrier height or width increases [3]. The tunneling rate takes the form $\exp(-2S_{\text{tun}}/\lambda)$, where $S_{\text{tun}}$ is the tunneling action and $\lambda=\chi/\delta$.

### 3. Quantum Activation

In driven dissipative systems far from equilibrium, a distinct mechanism called quantum activation becomes dominant [4, 5]. This process arises from the quantum noise that accompanies relaxation due to coupling with a thermal bath. Unlike tunneling, quantum activation involves transitions over the effective barrier through quantum fluctuations, even at zero temperature [6, 7]. 

The switching rate takes the form $W_{\text{sw}} \propto \exp(-R_A/\lambda)$, where:
- $R_A$ is the effective activation energy
- $\lambda$ is the effective Planck constant in the rotating frame

Key features that distinguish quantum activation from thermal activation and tunneling:
- Arises from intrinsic quantum noise in the dissipation
- Occurs even at zero temperature
- Involves transitions over (rather than through) the barrier
- Yields exponentially larger switching rates compared to tunneling
- Has no analog in equilibrium systems
- Dominates the switching dynamics unless the relaxation rate is exponentially small

## Instanton Approach to Quantum Activation

### Overview

Our task to find the rate at which the system transitions from one metastable state to another. Broadly this involves picking an initial state, evolving it over some time period, and finding the probability with which it has reached the target final state. Whereas a closed system could be described by a pure quantum state, and would evolve under the action of a unitary time evolution operator, our system is coupled to its environment and experiences significant drive and dissipation. Therefore it should be studied under the action of more general Liouvillian superoperator.

Finding transition rates can then be done in multiple ways. For example, one could write a Lindblad master equation for the system and evolve it numerically, or switching events could be simulated using a stochastic Schroedinger equation, but here we use the Keldysh path integral formalism. This approach is particularly powerful for our non-equilibrium system as it naturally incorporates both quantum effects and dissipation, while allowing us to identify the most probable switching paths through instanton solutions.

In this context, an instanton is the path of least action solution to the equations of motion in the Keldysh field theory formalism that connects the two stable nodes (metastable states) through the saddle point. This trajectory represents the most probable path by which the system surmounts the effective barrier via quantum fluctuations—even at zero temperature—due to the intrinsic non-equilibrium noise accompanying dissipation. The action of this instanton forms the exponential factor in the switching rate and therefore dominates its dependence on system parameters.

### Method

To calculate the instanton solutions and their associated switching rates, we follow these steps:

1. **Set up the Keldysh Framework**
   - Express the system's action using classical and quantum fields
   - Add dissipative terms from environmental coupling
   - Transform driving terms into the rotating frame

2. **Analyze Classical Fixed Points**
   - Find classical equations of motion (setting quantum fluctuations to zero)
   - Locate fixed points in parameter space:
     * Three points in bistable regime
     * Single point in monostable regime
   - Analyze stability via linearization and Jacobian eigenvalues

3. **Find Optimal Escape Paths**
   - Define instanton boundary conditions using:
     * Initial node position and outgoing Jacobian eigenvectors
     * Saddle point position and incoming Jacobian eigenvectors
   - Solve boundary value problem for the instanton trajectory

4. **Compute Switching Rates**
   - Calculate activation energy by integrating along optimal escape paths
   - Determine prefactor through comparison with numerical simulations

## Key Results

1. The authors develop a practical implementation for calculating switching paths using the instanton approach within Keldysh field theory and are able to calculate switching rates between bistable states in driven-dissipative nonlinear systems, specifically focusing on quantum fluctuation-induced switching.

2. The method allows the calculation of the exponential dependency of switching rates far from bifurcation points where previous methods have focused [6].

## References

[1] A. Kamenev, "Field Theory of Non-Equilibrium Systems" (Cambridge University Press, 2011).

[2] P. Hänggi, P. Talkner, and M. Borkovec, "Reaction-rate theory: fifty years after kramers," Reviews of modern physics 62, 251 (1990).

[3] W. H. Miller, "Semiclassical limit of quantum mechanical transition state theory for nonseparable systems," The Journal of chemical physics 62, 1899 (1975).

[4] M. Dykman and V. Smelyanskij, "Quantum theory of transitions between stable states of nonlinear oscillator, interacting with medium in a resonance field," Sov. Phys. JETP 67, 1769 (1988).

[5] M. Marthaler and M. Dykman, "Switching via quantum activation: A parametrically modulated oscillator," Physical Review A 73, 042108 (2006).

[6] M. Dykman, "Critical exponents in metastable decay via quantum activation," Physical Review E—Statistical, Nonlinear, and Soft Matter Physics 75, 011101 (2007).

[7] M. Dykman, "Fluctuating Nonlinear Oscillators: FromNanomechanics to Quantum Superconducting Circuits" (Oxford University Press, 2012).