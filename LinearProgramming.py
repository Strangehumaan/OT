import numpy as np
import plotly.graph_objects as go
from itertools import combinations
from scipy.spatial import ConvexHull

_TOL = 1e-9


# ---------------------------------------------------------------------------
# Graphical method
# ---------------------------------------------------------------------------

def _feasible_vertices(A, b):
    """
    Return every corner point of {x : A x <= b, x >= 0} as an array (one row per vertex).

    Corners are found by intersecting every choice of n boundary hyperplanes
    (the constraints plus the axes x_i = 0) and keeping only the points that
    satisfy *all* constraints.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape
    G = np.vstack([A, -np.eye(n)])
    h = np.concatenate([b, np.zeros(n)])

    vertices = []
    for idx in combinations(range(m + n), n):
        sub = G[list(idx)]
        if abs(np.linalg.det(sub)) < _TOL:
            continue  # parallel / dependent boundaries never meet in a single point
        x = np.linalg.solve(sub, h[list(idx)])
        x[np.abs(x) < 1e-12] = 0.0
        if np.all(G @ x <= h + 1e-7) and not any(np.allclose(x, v) for v in vertices):
            vertices.append(x)
    return np.array(vertices)


def graphical_method(c, A, b):
    """
    Solves a 2-variable or 3-variable LPP using the graphical method and visualizes it with Plotly.
    Problem form: maximize c.x subject to A x <= b, x >= 0 (assumes a bounded feasible region).

    Parameters:
    c : list - Coefficients of the objective function (e.g., [3, 2] or [3, 2, 1])
    A : 2D array or list - Coefficients of the constraints (e.g., [[2, 1], [1, 2]])
    b : list - Right-hand side values (e.g., [8, 6])

    Returns:
    (optimal_point, optimal_value)
    """
    # Convert A to a 2D NumPy array if it isn't already
    A = np.array(A)

    num_vars = len(c)

    if num_vars not in [2, 3]:
        raise ValueError("Graphical method supports only 2 or 3 variables.")
    if A.ndim != 2 or A.shape[1] != num_vars:
        raise ValueError(f"A must have {num_vars} columns for {num_vars} variables.")
    if A.shape[0] != len(b):
        raise ValueError(f"A must have {len(b)} rows to match the number of constraints.")

    if num_vars == 2:
        return solve_2d(c, A, b)
    else:
        return solve_3d(c, A, b)


def _best_vertex(c, corners):
    if len(corners) == 0:
        raise ValueError("The problem is infeasible: there is no feasible corner point.")
    z_values = corners @ c
    idx = int(np.argmax(z_values))
    return z_values, idx


def solve_2d(c, A, b):
    """Solve and visualize 2D LPP (any number of <= constraints)."""
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # Corner points of the feasible region: only vertices that satisfy every constraint
    corners = _feasible_vertices(A, b)
    z_values, optimal_idx = _best_vertex(c, corners)
    optimal_point = corners[optimal_idx]
    optimal_value = z_values[optimal_idx]

    x_max = max(1.0, corners[:, 0].max()) * 1.2
    y_max = max(1.0, corners[:, 1].max()) * 1.2
    x1 = np.linspace(0, x_max, 400)

    # Order the corners around their centre so the polygon outlines the region
    centre = corners.mean(axis=0)
    order = np.argsort(np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0]))
    polygon = corners[order]
    polygon = np.vstack([polygon, polygon[:1]])

    fig = go.Figure()
    for i in range(len(b)):
        label = f'{A[i, 0]:g}x₁ + {A[i, 1]:g}x₂ ≤ {b[i]:g}'
        if A[i, 1] != 0:
            fig.add_trace(go.Scatter(x=x1, y=(b[i] - A[i, 0] * x1) / A[i, 1], mode='lines', name=label))
        elif A[i, 0] != 0:  # vertical line x1 = const
            xv = b[i] / A[i, 0]
            fig.add_trace(go.Scatter(x=[xv, xv], y=[0, y_max], mode='lines', name=label))

    fig.add_trace(go.Scatter(
        x=polygon[:, 0], y=polygon[:, 1],
        fill='toself',
        fillcolor='rgba(255, 182, 193, 0.5)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='Feasible Region'
    ))
    fig.add_trace(go.Scatter(x=corners[:, 0], y=corners[:, 1], mode='markers', name='Corner Points',
                             marker=dict(size=10, color='red')))
    fig.add_trace(go.Scatter(
        x=[optimal_point[0]], y=[optimal_point[1]], mode='markers',
        name=f'Optimal ({optimal_point[0]:.1f}, {optimal_point[1]:.1f})',
        marker=dict(size=12, color='gold', symbol='star')
    ))
    if c[1] != 0:
        fig.add_trace(go.Scatter(x=x1, y=(optimal_value - c[0] * x1) / c[1], mode='lines',
                                 name=f'Z = {optimal_value:.1f}', line=dict(color='purple', dash='dash')))

    fig.update_layout(title='2D Graphical Method for LPP', xaxis_title='x₁', yaxis_title='x₂',
                      xaxis=dict(range=[0, x_max]), yaxis=dict(range=[0, y_max]), showlegend=True)
    fig.show()

    print(f"Corner Points and Z values: {list(zip(map(tuple, corners), z_values))}")
    print(f"Optimal Value: {optimal_value:.1f}")
    print(f"Optimal Solution: x1 = {optimal_point[0]:.1f}, x2 = {optimal_point[1]:.1f}")
    return optimal_point, optimal_value


def solve_3d(c, A, b):
    """Solve and visualize 3D LPP (any number of <= constraints)."""
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # Corner points: intersections of three boundary planes that satisfy every constraint
    corners = _feasible_vertices(A, b)
    z_values, optimal_idx = _best_vertex(c, corners)
    optimal_point = corners[optimal_idx]
    optimal_value = z_values[optimal_idx]

    fig = go.Figure()

    # Feasible region = convex hull of its corner points (needs a non-degenerate 3D shape)
    try:
        hull = ConvexHull(corners)
        s = hull.simplices
        fig.add_trace(go.Mesh3d(
            x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
            i=s[:, 0], j=s[:, 1], k=s[:, 2],
            opacity=0.4, color='lightpink', name='Feasible Region', showlegend=True
        ))
    except Exception:
        pass  # fewer than 4 corners or a flat region: just show the points

    fig.add_trace(go.Scatter3d(
        x=corners[:, 0], y=corners[:, 1], z=corners[:, 2], mode='markers', name='Corner Points',
        marker=dict(size=5, color='blue')
    ))
    fig.add_trace(go.Scatter3d(
        x=[optimal_point[0]], y=[optimal_point[1]], z=[optimal_point[2]],
        mode='markers',
        name=f'Optimal ({optimal_point[0]:.1f}, {optimal_point[1]:.1f}, {optimal_point[2]:.1f})',
        marker=dict(size=8, color='gold', symbol='diamond')
    ))
    fig.update_layout(
        title='3D Graphical Method for LPP',
        scene=dict(xaxis_title='x₁', yaxis_title='x₂', zaxis_title='x₃'),
        showlegend=True
    )
    fig.show()

    print(f"Corner Points and Z values: {list(zip(map(tuple, corners), z_values))}")
    print(f"Optimal Value: {optimal_value:.1f}")
    print(f"Optimal Solution: x1 = {optimal_point[0]:.1f}, x2 = {optimal_point[1]:.1f}, x3 = {optimal_point[2]:.1f}")
    return optimal_point, optimal_value


# ---------------------------------------------------------------------------
# Simplex core shared by simplex_method and big_m_method
# ---------------------------------------------------------------------------

def _run_simplex(tableau, basis, max_iter=100000):
    """
    Pivot `tableau` in place until the objective row has no negative entry.
    `basis[i]` is the column that is basic in row i; it is kept up to date.

    Returns True when optimal, False when the problem is unbounded.
    """
    m = tableau.shape[0] - 1
    for _ in range(max_iter):
        obj = tableau[-1, :-1]
        if np.all(obj >= -_TOL):
            return True

        pivot_col = int(np.argmin(obj))
        col = tableau[:m, pivot_col]
        ratios = np.full(m, np.inf)
        positive = col > _TOL
        ratios[positive] = tableau[:m, -1][positive] / col[positive]
        if np.all(np.isinf(ratios)):
            return False

        pivot_row = int(np.argmin(ratios))
        tableau[pivot_row] /= tableau[pivot_row, pivot_col]
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]
        basis[pivot_row] = pivot_col

    raise RuntimeError("Simplex did not converge (iteration limit reached).")


def _basic_solution(tableau, basis, n):
    """Values of the first n variables, read from the explicit basis (no float equality tests)."""
    solution = np.zeros(tableau.shape[1] - 1)
    for row, col in enumerate(basis):
        solution[col] = tableau[row, -1]
    solution = solution[:n]
    solution[np.abs(solution) < 1e-9] = 0.0
    return solution


# ---------------------------------------------------------------------------
# Integer programming (branch and bound)
# ---------------------------------------------------------------------------

def integer_simplex(c, A, b, Min=False, constraint_types=None, max_nodes=10000):
    """
    Solves an Integer LPP using Branch-and-Bound (depth first).

    Parameters:
    c : list - Objective function coefficients
    A : 2D array - Constraint coefficients
    b : list - Right-hand side values
    Min : bool - If True, minimize Z; if False (default), maximize Z
    constraint_types : list - '<=', '>=' or '=' per row (default: all '<=')
    max_nodes : int - safety limit on the number of subproblems explored

    Returns:
    best_value : float - Optimal integer objective value
    best_solution : array - Optimal integer solution
    """
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape
    types = list(constraint_types) if constraint_types is not None else ['<='] * m

    best_value = float('inf') if Min else float('-inf')
    best_solution = None
    stack = [(A, b, types)]
    nodes = 0

    while stack:
        nodes += 1
        if nodes > max_nodes:
            raise RuntimeError(f"Branch and bound exceeded {max_nodes} nodes (problem may be unbounded).")
        A_sub, b_sub, t_sub = stack.pop()

        # Branch constraints such as x_i >= ceil(.) need a '>=' row, which the plain
        # slack-basis simplex cannot start from; the Big-M solver handles all row types.
        opt_value, solution = big_m_method(c, A_sub, b_sub, t_sub, Min=Min)
        if solution is None:  # infeasible or unbounded relaxation
            continue

        # Bound: this subtree cannot beat the incumbent
        if (Min and opt_value >= best_value - _TOL) or (not Min and opt_value <= best_value + _TOL):
            continue

        fractional = np.abs(solution - np.round(solution)) > 1e-6
        if not fractional.any():
            best_value = opt_value
            best_solution = np.round(solution)
            continue

        # Branch on the first fractional variable
        i = int(np.argmax(fractional))
        floor_val = np.floor(solution[i])
        row = np.zeros((1, n))
        row[0, i] = 1
        A_new = np.vstack([A_sub, row])
        # x_i >= floor + 1 (pushed first) and x_i <= floor (explored first)
        stack.append((A_new, np.append(b_sub, floor_val + 1), t_sub + ['>=']))
        stack.append((A_new, np.append(b_sub, floor_val), t_sub + ['<=']))

    return best_value, best_solution


# ---------------------------------------------------------------------------
# Simplex / Big-M
# ---------------------------------------------------------------------------

def simplex_method(c, A, b, integer=False, Min=False):
    """
    Solves a Linear Programming Problem using the Simplex Method (constraints A x <= b, x >= 0).

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

    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape

    if np.any(b < 0):
        # The slack basis is not feasible when some b_i < 0; use the Big-M solver instead.
        return big_m_method(c, A, b, ['<='] * m, Min=Min)

    # Initial tableau: [A | I | b] with the objective row at the bottom
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :n] = A
    tableau[:m, n:n + m] = np.eye(m)  # Slack variables
    tableau[:m, -1] = b
    # Objective row: -c for maximization; for minimization we maximize -c.x, so the row is +c
    tableau[-1, :n] = c if Min else -c
    basis = list(range(n, n + m))

    if not _run_simplex(tableau, basis):
        return "Problem is unbounded", None

    solution = _basic_solution(tableau, basis, n)
    optimal_value = tableau[-1, -1]
    optimal_value = (-optimal_value if Min else optimal_value) + 0.0  # + 0.0 avoids -0.0
    return optimal_value, solution


def big_m_method(c, A, b, constraint_types, Min=False, M=None):
    """
    Solves an LPP using the Big M Method.

    Parameters:
    c : list - Objective function coefficients (e.g., [3, 2])
    A : 2D array - Constraint coefficients (e.g., [[2, 1], [1, 2]])
    b : list - Right-hand side values (e.g., [8, 6])
    constraint_types : list - Types of constraints ('<=', '>=', '=')
    Min : bool - If True, minimize Z; if False (default), maximize Z
    M : float - Penalty for artificial variables (default: scaled to the size of c)

    Returns:
    optimal_value : float - Optimal value of Z
    solution : array - Values of decision variables
    """
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    constraint_types = list(constraint_types)

    m, n = A.shape
    if len(constraint_types) != m:
        raise ValueError("constraint_types must have one entry per constraint.")

    # Make every right-hand side non-negative (multiply the row by -1 and flip its sense)
    flip = {'<=': '>=', '>=': '<=', '=': '='}
    for i in range(m):
        if constraint_types[i] not in flip:
            raise ValueError(f"Unknown constraint type {constraint_types[i]!r}; use '<=', '>=' or '='.")
        if b[i] < 0:
            A[i] = -A[i]
            b[i] = -b[i]
            constraint_types[i] = flip[constraint_types[i]]

    if M is None:
        M = 1e4 * max(1.0, float(np.abs(c).max()))

    num_slack = constraint_types.count('<=')
    num_surplus = constraint_types.count('>=')
    num_artificial = num_surplus + constraint_types.count('=')

    tableau = np.zeros((m + 1, n + num_slack + num_surplus + num_artificial + 1))
    basis = []

    # Fill constraint rows
    slack_idx, surplus_idx = n, n + num_slack
    first_artificial = artificial_idx = n + num_slack + num_surplus
    for i in range(m):
        tableau[i, :n] = A[i]
        if constraint_types[i] == '<=':
            tableau[i, slack_idx] = 1
            basis.append(slack_idx)
            slack_idx += 1
        elif constraint_types[i] == '>=':
            tableau[i, surplus_idx] = -1
            surplus_idx += 1
            tableau[i, artificial_idx] = 1
            basis.append(artificial_idx)
            artificial_idx += 1
        else:  # '='
            tableau[i, artificial_idx] = 1
            basis.append(artificial_idx)
            artificial_idx += 1
        tableau[i, -1] = b[i]

    # Objective row: maximize (c.x for Max, -c.x for Min) - M * (sum of artificials)
    tableau[-1, :n] = c if Min else -c
    tableau[-1, first_artificial:first_artificial + num_artificial] = M  # artificial penalty
    for i in range(m):
        if constraint_types[i] in ('>=', '='):
            tableau[-1, :] -= M * tableau[i, :]  # price out the artificial basis columns

    if not _run_simplex(tableau, basis):
        # "Unbounded" only makes sense for a feasible problem: check feasibility first
        # by solving with a zero objective (which can never be unbounded).
        if c.any():
            status, _ = big_m_method(np.zeros(n), A, b, constraint_types, Min=Min, M=M)
            if status == "Problem is infeasible":
                return status, None
        return "Problem is unbounded", None

    # Infeasible if an artificial variable is still in the basis at a positive level
    for row, col in enumerate(basis):
        if col >= first_artificial and tableau[row, -1] > 1e-6:
            return "Problem is infeasible", None

    solution = _basic_solution(tableau, basis, n)
    optimal_value = tableau[-1, -1]
    optimal_value = (-optimal_value if Min else optimal_value) + 0.0
    return optimal_value, solution
