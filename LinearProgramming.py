import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

def graphical_method(c, A, b):
    """
    Solves a 2-variable or 3-variable LPP using the graphical method and visualizes it with Plotly.
    
    Parameters:
    c : list - Coefficients of the objective function (e.g., [3, 2] or [3, 2, 1])
    A : 2D array or list - Coefficients of the constraints (e.g., [[2, 1], [1, 2]])
    b : list - Right-hand side values (e.g., [8, 6])
    """
    # Convert A to a 2D NumPy array if it isn't already
    A = np.array(A)

    num_vars = len(c)
    
    if num_vars not in [2, 3]:
        raise ValueError("Graphical method supports only 2 or 3 variables.")
    if A.shape[1] != num_vars:
        raise ValueError(f"A must have {num_vars} columns for {num_vars} variables.")
    if A.shape[0] != len(b):
        raise ValueError(f"A must have {len(b)} rows to match the number of constraints.")

    if num_vars == 2:
        solve_2d(c, A, b)
    else:
        solve_3d(c, A, b)

def solve_2d(c, A, b):
    """Solve and visualize 2D LPP."""
    x1 = np.linspace(0, 10, 400)
    
    # Compute x2 for each constraint
    x2_c1 = (b[0] - A[0, 0] * x1) / A[0, 1]  # e.g., 2x1 + x2 <= 8
    x2_c2 = (b[1] - A[1, 0] * x1) / A[1, 1]  # e.g., x1 + 2x2 <= 6
    
    x2_c1 = np.maximum(x2_c1, 0)
    x2_c2 = np.maximum(x2_c2, 0)
    x1 = np.maximum(x1, 0)
    
    x2_feasible = np.minimum(x2_c1, x2_c2)
    
    # Corner points of the feasible region
    corners = [
        (0, 0),
        (b[0] / A[0, 0], 0),  # x2 = 0
        (0, b[1] / A[1, 1]),  # x1 = 0
        ((b[0] * A[1, 1] - b[1] * A[0, 1]) / (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]),
         (b[1] * A[0, 0] - b[0] * A[1, 0]) / (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]))
    ]
    
    z_values = [c[0] * x1_val + c[1] * x2_val for x1_val, x2_val in corners]
    optimal_idx = np.argmax(z_values)
    optimal_point = corners[optimal_idx]
    optimal_value = z_values[optimal_idx]
    
    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x1, y=x2_c1, mode='lines', name=f'{A[0,0]}x₁ + {A[0,1]}x₂ ≤ {b[0]}', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x1, y=x2_c2, mode='lines', name=f'{A[1,0]}x₁ + {A[1,1]}x₂ ≤ {b[1]}', line=dict(color='green')))
    fig.add_trace(go.Scatter(
        x=np.concatenate([x1, x1[::-1]]),
        y=np.concatenate([x2_feasible, np.zeros_like(x2_feasible)]),
        fill='toself',
        fillcolor='rgba(255, 182, 193, 0.5)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='Feasible Region'
    ))
    
    corner_x, corner_y = zip(*corners)
    fig.add_trace(go.Scatter(x=corner_x, y=corner_y, mode='markers', name='Corner Points', marker=dict(size=10, color='red')))
    fig.add_trace(go.Scatter(
        x=[optimal_point[0]], y=[optimal_point[1]], mode='markers', name=f'Optimal ({optimal_point[0]:.1f}, {optimal_point[1]:.1f})',
        marker=dict(size=12, color='gold', symbol='star')
    ))
    
    x2_obj = (optimal_value - c[0] * x1) / c[1]
    fig.add_trace(go.Scatter(x=x1, y=x2_obj, mode='lines', name=f'Z = {optimal_value:.1f}', line=dict(color='purple', dash='dash')))
    
    fig.update_layout(title='2D Graphical Method for LPP', xaxis_title='x₁', yaxis_title='x₂', showlegend=True)
    fig.show()
    
    print(f"Corner Points and Z values: {list(zip(corners, z_values))}")
    print(f"Optimal Value: {optimal_value:.1f}")
    print(f"Optimal Solution: x1 = {optimal_point[0]:.1f}, x2 = {optimal_point[1]:.1f}")


def solve_3d(c, A, b):
        """Solve and visualize 3D LPP."""
        # Define grid for x1, 
        # A = np.array(A)
        x1 = np.linspace(0, 10, 50)
        x2 = np.linspace(0, 10, 50)
        x1, x2 = np.meshgrid(x1, x2)
        
        # Compute x3 for each constraint and take the minimum (feasible region)
        x3 = np.full_like(x1, np.inf)
        for i in range(len(b)):
            denom = A[i, 2]
            if denom != 0:
                x3_i = (b[i] - A[i, 0] * x1 - A[i, 1] * x2) / denom
                x3 = np.minimum(x3, np.maximum(x3_i, 0))  # x3 >= 0
        
        # Define corner points (simplified for example; in practice, solve system of equations)
        corners = [
            (0, 0, 0),
            (b[0] / A[0, 0], 0, 0),
            (0, b[1] / A[1, 1], 0),
            (0, 0, b[2] / A[2, 2]),
            # Add more points by solving intersections (simplified here)
            (2, 1, 0),  # Example intersection point
        ]
        
        z_values = [c[0] * x1_val + c[1] * x2_val + c[2] * x3_val for x1_val, x2_val, x3_val in corners]
        optimal_idx = np.argmax(z_values)
        optimal_point = corners[optimal_idx]
        optimal_value = z_values[optimal_idx]
        
        fig = go.Figure()
        
        # Feasible region surface
        fig.add_trace(go.Surface(x=x1, y=x2, z=x3, opacity=0.5, colorscale='Reds', name='Feasible Region'))
        
        # Corner points
        corner_x, corner_y, corner_z = zip(*corners)
        fig.add_trace(go.Scatter3d(
            x=corner_x, y=corner_y, z=corner_z, mode='markers', name='Corner Points',
            marker=dict(size=5, color='blue')
        ))
        
        # Optimal point
        fig.add_trace(go.Scatter3d(
            x=[optimal_point[0]], y=[optimal_point[1]], z=[optimal_point[2]],
            mode='markers', name=f'Optimal ({optimal_point[0]:.1f}, {optimal_point[1]:.1f}, {optimal_point[2]:.1f})',
            marker=dict(size=8, color='gold', symbol='diamond')
        ))
        
        fig.update_layout(
            title='3D Graphical Method for LPP',
            scene=dict(
                xaxis_title='x₁',
                yaxis_title='x₂',
                zaxis_title='x₃'
            ),
            showlegend=True
        )
        fig.show()
        
        print(f"Corner Points and Z values: {list(zip(corners, z_values))}")
        print(f"Optimal Value: {optimal_value:.1f}")
        print(f"Optimal Solution: x1 = {optimal_point[0]:.1f}, x2 = {optimal_point[1]:.1f}, x3 = {optimal_point[2]:.1f}")

import numpy as np

def integer_simplex(c, A, b, Min=False):
    """
    Solves an Integer LPP using Branch-and-Bound with Simplex Method.
    
    Parameters:
    c : list - Objective function coefficients
    A : 2D array - Constraint coefficients
    b : list - Right-hand side values
    Min : bool - If True, minimize Z; if False (default), maximize Z
    
    Returns:
    best_value : float - Optimal integer objective value
    best_solution : array - Optimal integer solution
    """
    best_value = float('inf') if Min else float('-inf')
    best_solution = None
    subproblems = [(c, A, b)]  # List of (c, A, b) tuples to explore
    
    while subproblems:
        c_sub, A_sub, b_sub = subproblems.pop(0)
        opt_value, solution = simplex_method(c_sub, A_sub, b_sub, Min=Min)
        
        if opt_value is None:  # Unbounded or infeasible
            continue
        
        # For minimization, skip if value is greater than best; for maximization, skip if less
        if (Min and opt_value >= best_value) or (not Min and opt_value <= best_value):
            continue
        
        # Check if solution is integer
        if all(np.isclose(solution, np.round(solution), atol=1e-6)):
            if (Min and opt_value < best_value) or (not Min and opt_value > best_value):
                best_value = opt_value
                best_solution = solution.copy()
            continue
        
        # Branch on the first non-integer variable
        for i in range(len(solution)):
            if not np.isclose(solution[i], round(solution[i]), atol=1e-6):
                x_val = solution[i]
                floor_val = np.floor(x_val)
                ceil_val = np.ceil(x_val)
                
                # Subproblem 1: x_i <= floor(x_i)
                A1 = np.vstack([A_sub, np.zeros(A_sub.shape[1])])
                A1[-1, i] = 1
                b1 = np.append(b_sub, floor_val)
                
                # Subproblem 2: x_i >= ceil(x_i)
                A2 = np.vstack([A_sub, np.zeros(A_sub.shape[1])])
                A2[-1, i] = -1
                b2 = np.append(b_sub, -ceil_val)
                
                subproblems.append((c_sub, A1, b1))
                subproblems.append((c_sub, A2, b2))
                break
    
    return best_value, best_solution


def simplex_method(c, A, b, integer=False, Min=False):
    """
    Solves a Linear Programming Problem using the Simplex Method.
    
    Parameters:
    c : list or array - Coefficients of the objective function
    A : 2D list or array - Coefficients of the constraints (left-hand side)
    b : list or array - Right-hand side values of the constraints
    integer : bool - If True, use integer simplex; if False (default), regular simplex
    Min : bool - If True, minimize Z; if False (default), maximize Z
    
    Returns:
    optimal_value : float - Optimal value of the objective function
    solution : array - Values of the decision variables
    """
    if integer:
        return integer_simplex(c, A, b, Min=Min)
    elif integer is False:
        # Convert inputs to numpy arrays
        c = np.array(c, dtype=float)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        
        m, n = A.shape
        
        # Create the initial tableau
        tableau = np.zeros((m + 1, n + m + 1))
        
        # Fill the tableau
        # Objective function row: negate c for maximization, double negate for minimization
        tableau[-1, :n] = c if Min else -c  # For Min, we maximize -c; for Max, maximize c
        tableau[-1, n:n+m] = 0  # Slack variables
        tableau[-1, -1] = 0  # Initial objective value
        tableau[:m, :n] = A  # Constraint coefficients
        tableau[:m, n:n+m] = np.eye(m)  # Slack variables
        tableau[:m, -1] = b  # RHS
        
        # Simplex iterations
        while True:
            if all(tableau[-1, :-1] >= 0):  # Optimal when no negative coefficients
                break
            
            pivot_col = np.argmin(tableau[-1, :-1])
            ratios = []
            for i in range(m):
                if tableau[i, pivot_col] > 0:
                    ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                else:
                    ratios.append(float('inf'))
            
            if all(r == float('inf') for r in ratios):
                return "Problem is unbounded", None
            
            pivot_row = np.argmin(ratios)
            pivot_element = tableau[pivot_row, pivot_col]
            tableau[pivot_row] /= pivot_element
            
            for i in range(m + 1):
                if i != pivot_row:
                    factor = tableau[i, pivot_col]
                    tableau[i] -= factor * tableau[pivot_row]
        
        # Extract solution
        solution = np.zeros(n)
        for j in range(n):
            col = tableau[:-1, j]
            if np.sum(col == 1) == 1 and np.sum(col) == 1:
                row = np.where(col == 1)[0][0]
                solution[j] = tableau[row, -1]
        
        optimal_value = tableau[-1, -1]
        # Adjust for minimization: if Min=True, negate the result
        optimal_value = -optimal_value if Min else optimal_value
        
        return optimal_value, solution
    else:
        print('Invalid Arguments')
        return None, None

def big_m_method(c, A, b, constraint_types, Min=False):
    """
    Solves an LPP using the Big M Method.
    
    Parameters:
    c : list - Objective function coefficients (e.g., [3, 2])
    A : 2D array - Constraint coefficients (e.g., [[2, 1], [1, 2]])
    b : list - Right-hand side values (e.g., [8, 6])
    constraint_types : list - Types of constraints ('<=', '>=', '=')
    Min : bool - If True, minimize Z; if False (default), maximize Z
    
    Returns:
    optimal_value : float - Optimal value of Z
    solution : array - Values of decision variables
    """
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    m, n = A.shape
    M = 1000  # Big M value
    
    num_slack = constraint_types.count('<=')
    num_surplus = constraint_types.count('>=')
    num_artificial = constraint_types.count('>=') + constraint_types.count('=')
    
    tableau = np.zeros((m + 1, n + num_slack + num_surplus + num_artificial + 1))
    
    # Fill constraint rows
    slack_idx, surplus_idx, artificial_idx = n, n + num_slack, n + num_slack + num_surplus
    for i in range(m):
        tableau[i, :n] = A[i]
        if constraint_types[i] == '<=':
            tableau[i, slack_idx] = 1
            slack_idx += 1
        elif constraint_types[i] == '>=':
            tableau[i, surplus_idx] = -1
            surplus_idx += 1
            tableau[i, artificial_idx] = 1
            artificial_idx += 1
        elif constraint_types[i] == '=':
            tableau[i, artificial_idx] = 1
            artificial_idx += 1
        tableau[i, -1] = b[i]
    
    # Fill objective row
    tableau[-1, :n] = c if Min else -c  # For Min, maximize -c; for Max, maximize c
    artificial_idx = n + num_slack + num_surplus
    for i in range(m):
        if constraint_types[i] in ['>=', '=']:
            tableau[-1, :] -= M * tableau[i, :]  # Penalize artificial variables
    
    # Simplex iterations
    while True:
        if all(tableau[-1, :-1] >= 0):
            break
        
        pivot_col = np.argmin(tableau[-1, :-1])
        ratios = []
        for i in range(m):
            if tableau[i, pivot_col] > 0:
                ratios.append(tableau[i, -1] / tableau[i, pivot_col])
            else:
                ratios.append(float('inf'))
        
        if all(r == float('inf') for r in ratios):
            return "Problem is unbounded", None
        
        pivot_row = np.argmin(ratios)
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row] /= pivot_element
        
        for i in range(m + 1):
            if i != pivot_row:
                factor = tableau[i, pivot_col]
                tableau[i] -= factor * tableau[pivot_row]
    
    # Extract solution
    solution = np.zeros(n)
    for j in range(n):
        col = tableau[:-1, j]
        if np.sum(col == 1) == 1 and np.sum(col) == 1:
            row = np.where(col == 1)[0][0]
            solution[j] = tableau[row, -1]
    
    optimal_value = tableau[-1, -1]
    optimal_value = -optimal_value if Min else optimal_value  # Adjust for minimization
    
    # Check feasibility
    artificial_vals = tableau[:-1, n + num_slack + num_surplus:n + num_slack + num_surplus + num_artificial]
    if np.any(artificial_vals > 0):
        return "Problem is infeasible", None
    
    return optimal_value, solution
