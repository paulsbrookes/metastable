# Metastable States in Driven Nonlinear Systems

This repository contains research and documentation on the study of switching dynamics in driven nonlinear oscillators, with a particular focus on quantum activation in the bistable regime at zero temperature.

## Documentation

The documentation for this project can be viewed at: https://paulsbrookes.github.io/metastable/

## Core Concepts

1. **Overview** - Introduction to switching dynamics and transition mechanisms
2. **Keldysh Lagrangian** - Development of the Keldysh field theory approach
3. **Auxiliary Hamiltonian** - Analysis of the auxiliary Hamiltonian system
4. **Original Equations of Motion** - Derivation and analysis of the system's equations of motion
5. **Fixed Points** - Analysis of the fixed points of the auxiliary Hamiltonian
6. **Stability Analysis** - Analysis of the boundary conditions for the switching trajectories
7. **Paths** - Analysis of the paths for the switching trajectories

## About

This project accompanies the research paper ["A Real-time Instanton Approach to Quantum Activation"](https://arxiv.org/abs/2409.00681) (arXiv:2409.00681) by Chang-Woo Lee, Paul Brookes, Kee-Su Park, Marzena H. Szymańska, and Eran Ginossar.

The work covers research into quantum activation in driven dissipative systems far from equilibrium. The focus is on understanding how quantum noise associated with relaxation can lead to transitions between metastable states, even at zero temperature.

## Key Areas of Investigation

- Analysis of fixed points in the bistable regime
- Development of Keldysh field theory approach
- Study of quantum fluctuations and their role in state transitions

## Requirements

- Python ≥ 3.10
- Dependencies listed in pyproject.toml including:
  - qutip
  - numpy
  - scipy
  - sympy
  - matplotlib
  - pandas

## Installation

```bash
# Clone the repository
git clone https://github.com/paulsbrookes/metastable.git
cd metastable

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

## Usage

Refer to the documentation at https://paulsbrookes.github.io/metastable/ for theoretical background and usage examples. 