# Linear Programming Solver

A Python library for solving linear programming problems using various methods including graphical solutions, simplex method, and transportation problems.

## Features

- **Linear Programming Solutions:**
  - Graphical Method (2D and 3D visualization)
  - Simplex Method
  - Big M Method for problems with ≥ and = constraints
  - Integer Programming using Branch and Bound

- **Transportation Problem Solutions:**
  - North-West Corner Rule (NWCR)
  - Vogel's Approximation Method (VAM)
  - Least Cost Method (LCM)
  - Modified Distribution Method (MODI)

## Installation

```bash
pip install OTTools
```

## Usage

### Linear Programming

#### Graphical Method (2-Variable)

```python
from OTTools import graphical_method

# Maximize Z = 3x₁ + 2x₂
# Subject to:
# 2x₁ + x₂ ≤ 8
# x₁ + 2x₂ ≤ 6
# x₁, x₂ ≥ 0

c = [3, 2]           # Coefficients of objective function
A = [[2, 1], [1, 2]] # Coefficients of constraints
b = [8, 6]           # Right-hand side values

graphical_method(c, A, b)  # Visualizes the solution
```

#### Simplex Method

```python
from OTTools import simplex_method

# Maximize Z = 3x₁ + 2x₂
# Subject to:
# 2x₁ + x₂ ≤ 8
# x₁ + 2x₂ ≤ 6
# x₁, x₂ ≥ 0

c = [3, 2]
A = [[2, 1], [1, 2]]
b = [8, 6]

optimal_value, solution = simplex_method(c, A, b)
print(f"Optimal value: {optimal_value}")
print(f"Solution: {solution}")
```

#### Big M Method (for mixed constraints)

```python
from OTTools import big_m_method

# Maximize Z = 2x₁ + 3x₂
# Subject to:
# x₁ + x₂ ≤ 6
# x₁ + 2x₂ ≥ 8
# x₁ + x₂ = 5
# x₁, x₂ ≥ 0

c = [2, 3]
A = [[1, 1], [1, 2], [1, 1]]
b = [6, 8, 5]
constraint_types = ['<=', '>=', '=']

optimal_value, solution = big_m_method(c, A, b, constraint_types, Min=False)
print(f"Optimal value: {optimal_value}")
print(f"Solution: {solution}")
```

#### Big M Method (Minimization Example)

```python
from OTTools import big_m_method

# Minimize Z = 4x₁ + 2x₂
# Subject to:
# 3x₁ + x₂ ≥ 15
# x₁ + 2x₂ ≤ 20
# x₁ + x₂ = 10
# x₁, x₂ ≥ 0

c = [4, 2]
A = [[3, 1], [1, 2], [1, 1]]
b = [15, 20, 10]
constraint_types = ['>=', '<=', '=']

# Set Min=True for minimization
optimal_value, solution = big_m_method(c, A, b, constraint_types, Min=True)
print(f"Minimum value: {optimal_value}")
print(f"Solution: x₁ = {solution[0]}, x₂ = {solution[1]}")
```

### Transportation Problems

#### Solving with Different Methods

```python
from OTTools import TransportationProblem
import numpy as np

# Cost matrix
cost_matrix = [
    [4, 8, 8, 0],
    [16, 24, 16, 0],
    [8, 16, 24, 0]
]

# Supply and demand
supply = np.array([76, 82, 77])
demand = np.array([72, 102, 41, 20])

# Create a transportation problem instance
tp = TransportationProblem(cost_matrix, supply, demand)

# Solve with Vogel's Approximation Method
vam_solution = tp.solve('VAM')
print("Vogel's Solution:")
print(vam_solution)

# Solve with Modified Distribution Method (optimal solution)
modi_solution = tp.solve('MODI')
print("\nModi Method Solution (Optimal):")
print(modi_solution)

# OTToolsher available methods: 'NWCR' (North-West Corner Rule), 'LCM' (Least Cost Method)
```

## Visualization Examples

The graphical method provides interactive PlOTToolsly visualizations for bOTToolsh 2D and 3D linear programming problems:

### 2D Example
For 2-variable problems, the library visualizes:
- Constraint lines
- Feasible region
- Corner points
- Optimal solution
- Objective function line

### 3D Example
For 3-variable problems, the library visualizes:
- Constraint surfaces
- Feasible region
- Corner points
- Optimal solution

## Dependencies

- NumPy
- PlOTToolsly
- SciPy
- Pandas

## License

This project is licensed under the MIT License - see the LICENSE file for details.
