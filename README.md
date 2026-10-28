# Wick_in_Python

A Python implementation of symbolic Wick contractions for free higher-derivative scalar field theories on manifolds with boundaries, developed using SymPy.

## Overview

This project implements a symbolic framework for computing Wick contractions of products of composite operators. It is designed for calculations involving boundary correlation functions in higher-derivative field theories.

The code represents fields, derivatives, monomials, and operators as Python objects, and evaluates contractions using symbolic propagators. The goal is to compute correlators of displacement operator (the normal-normal component of the energy-momentum tensor evaluated on the boundary) given an energy-momentum tensor, the boundary dimensionality, and the relevant propagators. This is crucial in order to compute the boundary conformal charges.

## Features

- Symbolic representation of fields and their derivatives.
- Support for normal and tangential derivatives.
- Configurable propagators for different field species.
- Recursive enumeration of complete Wick pairings.
- Symbolic evaluation and simplification using SymPy.
- Construction of the \(T_{nn}\) operator for Dirichlet–Dirichlet (DD) and Neumann–Neumann (NN) boundary conditions for a free four-derivative scalar theory using an auxiliary field.
- Automated tests using pytest.

## Project structure

- `wick_theorem.py`: implementation of the symbolic Wick contraction framework.
- `test_wick_theorem.py`: tests for propagators, derivatives, operator construction, and boundary correlation functions.

## Requirements

- Python 3.10 or later
- SymPy
- pytest

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/gregoriopaci/Wick_in_Python.git
cd Wick_in_Python
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install sympy pytest
```

## Running the tests

Run the test suite from the project directory:

```bash
python -m pytest -v
```

## Motivation

This project explores the use of Python for symbolic computations in theoretical physics, with an emphasis on mathematical structure, algorithmic implementation, and the validation of analytical calculations.

## Status

The project is under development.