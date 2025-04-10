import numpy as np
import pandas as pd
 
class TransportationProblem:
    def __init__(self, cost_matrix, supply, demand, auto_balance=True):
        self.cost_matrix = np.array(cost_matrix)
        self.supply = np.array(supply)
        self.demand = np.array(demand)
        self.allocation = None
        if auto_balance:
            self.balance_problem()
 
    def balance_problem(self):
        total_supply = np.sum(self.supply)
        total_demand = np.sum(self.demand)
        if total_supply == total_demand:
            return
        elif total_supply > total_demand:
            # Add dummy demand column
            dummy_demand = total_supply - total_demand
            self.demand = np.append(self.demand, dummy_demand)
            # Add a column of zeros to the cost matrix
            self.cost_matrix = np.hstack((self.cost_matrix, np.zeros((self.cost_matrix.shape[0], 1))))
        else:
            # Add dummy supply row
            dummy_supply = total_demand - total_supply
            self.supply = np.append(self.supply, dummy_supply)
            # Add a row of zeros to the cost matrix
            self.cost_matrix = np.vstack((self.cost_matrix, np.zeros((1, self.cost_matrix.shape[1]))))
 
    def NorthWestSolver(self):
        allocation = np.zeros_like(self.cost_matrix, dtype=float)
        supply = self.supply.copy()
        demand = self.demand.copy()
        i, j = 0, 0
        while i < len(supply) and j < len(demand):
            if supply[i] == 0:
                i += 1
            elif demand[j] == 0:
                j += 1
            else:
                amount = min(supply[i], demand[j])
                allocation[i, j] = amount
                supply[i] -= amount
                demand[j] -= amount
        return allocation
 
    @staticmethod
    def row_penalties(cost_matrix):
        penalties = np.zeros(cost_matrix.shape[0])
        for i in range(cost_matrix.shape[0]):
            finite_vals = cost_matrix[i, np.isfinite(cost_matrix[i, :])]
            if len(finite_vals) < 2:
                penalties[i] = 0  # Or choose an alternative handling
            else:
                two_min = np.partition(finite_vals, 1)[:2]
                penalties[i] = two_min[1] - two_min[0]
        return penalties
 
 
    @staticmethod
    def col_penalties(cost_matrix):
        two_min = np.partition(cost_matrix, 1, axis=0)[:2, :]
        penalties = two_min[1, :] - two_min[0, :]
        return penalties
 
    def VogelSolver(self):
        cost_mat = self.cost_matrix.copy().astype(float)
        supply = self.supply.copy()
        demand = self.demand.copy()
        n, m = cost_mat.shape
        allocation = np.zeros((n, m))
 
        while np.any(supply > 0) and np.any(demand > 0):
            row_pen = self.row_penalties(cost_mat)
            col_pen = self.col_penalties(cost_mat)
            max_row_pen = np.nanmax(row_pen) if not np.all(np.isnan(row_pen)) else -np.inf
            max_col_pen = np.nanmax(col_pen) if not np.all(np.isnan(col_pen)) else -np.inf
 
            if max_row_pen >= max_col_pen:
                row = np.nanargmax(row_pen)
                col = np.nanargmin(cost_mat[row])
            else:
                col = np.nanargmax(col_pen)
                row = np.nanargmin(cost_mat[:, col])
 
            alloc = min(supply[row], demand[col])
            allocation[row, col] = alloc
            supply[row] -= alloc
            demand[col] -= alloc
 
            if supply[row] == 0:
                cost_mat[row, :] = np.inf
            else:
                cost_mat[:, col] = np.inf
 
        return allocation
 
    def LeastCostMethod(self):
        cost_mat = self.cost_matrix.copy()
        supply = self.supply.copy()
        demand = self.demand.copy()
        allocation = np.zeros_like(cost_mat, dtype=float)
 
        while True:
            # Find the minimum cost cell that is not exhausted
            min_val = np.inf
            min_row, min_col = -1, -1
            for i in range(cost_mat.shape[0]):
                for j in range(cost_mat.shape[1]):
                    if cost_mat[i, j] < min_val and supply[i] > 0 and demand[j] > 0:
                        min_val = cost_mat[i, j]
                        min_row, min_col = i, j
            if min_val == np.inf:
                break  # No more cells to allocate
            alloc = min(supply[min_row], demand[min_col])
            allocation[min_row, min_col] = alloc
            supply[min_row] -= alloc
            demand[min_col] -= alloc
            if supply[min_row] == 0:
                cost_mat[min_row, :] = np.inf
            if demand[min_col] == 0:
                cost_mat[:, min_col] = np.inf
        return allocation
 
    def find_cycle(self, basic_cells, start):
        """
        Revised cycle finder using recursive backtracking.
        Given basic_cells (a list of (row, col) tuples) and a starting cell,
        find a closed loop (cycle) that alternates between horizontal and vertical moves.
        """
        basic_set = set(basic_cells)
 
        def search(path):
            current = path[-1]
            # When cycle is closed (length >= 4) and returns to start, we've found a cycle.
            if len(path) >= 4 and current == start:
                return path
            # Determine allowed move direction based on last move.
            # For the first move, both row and column moves are allowed.
            last_direction = None
            if len(path) >= 2:
                prev = path[-2]
                last_direction = 'row' if current[0] == prev[0] else 'col'
            for cell in basic_set:
                # Skip if same as current or if already in path (unless it is the start and eligible to close)
                if cell == current:
                    continue
                if cell in path and cell != start:
                    continue
                # Must be in same row or same column.
                if current[0] != cell[0] and current[1] != cell[1]:
                    continue
                # If a last move exists, alternate direction.
                if last_direction is not None:
                    if last_direction == 'row' and current[1] != cell[1]:
                        continue
                    if last_direction == 'col' and current[0] != cell[0]:
                        continue
                # Do not allow immediate return to start unless the path length is >=3 (to allow cycle length of 4)
                if cell == start and len(path) < 3:
                    continue
 
                new_path = path + [cell]
                result = search(new_path)
                if result is not None:
                    return result
            return None
 
        return search([start])
 
    def ModiMethod(self):
        # Get initial feasible solution using Vogel's method
        allocation = self.VogelSolver()
        m, n = allocation.shape
 
        # Iteratively improve the solution until optimality
        while True:
            # Identify basic cells (cells with positive allocation)
            basic_cells = [(i, j) for i in range(m) for j in range(n) if allocation[i, j] > 0]
            # --- Step 1: Compute dual variables u and v for basic cells ---
            u = np.full(m, np.nan)
            v = np.full(n, np.nan)
            u[0] = 0  # Set an arbitrary starting point
            updated = True
            while updated:
                updated = False
                for (i, j) in basic_cells:
                    if np.isnan(u[i]) and not np.isnan(v[j]):
                        u[i] = self.cost_matrix[i, j] - v[j]
                        updated = True
                    elif not np.isnan(u[i]) and np.isnan(v[j]):
                        v[j] = self.cost_matrix[i, j] - u[i]
                        updated = True
 
            # --- Step 2: Compute reduced costs for non-basic cells ---
            reduced_cost = np.full((m, n), np.nan)
            non_basic = []
            for i in range(m):
                for j in range(n):
                    if (i, j) not in basic_cells:
                        non_basic.append((i, j))
                        reduced_cost[i, j] = self.cost_matrix[i, j] - (u[i] + v[j])
 
            # --- Step 3: Check for optimality ---
            if all(reduced_cost[i, j] >= 0 for (i, j) in non_basic if not np.isnan(reduced_cost[i, j])):
                print("Optimal solution found.")
                return allocation
 
            # --- Step 4: Choose the entering cell (most negative reduced cost) ---
            min_val = 0
            entering_cell = None
            for (i, j) in non_basic:
                if not np.isnan(reduced_cost[i, j]) and reduced_cost[i, j] < min_val:
                    min_val = reduced_cost[i, j]
                    entering_cell = (i, j)
            if entering_cell is None:
                print("No entering cell found; solution is optimal.")
                return allocation
 
            # --- Step 5: Find a cycle (closed loop) including the entering cell ---
            temp_basic = basic_cells.copy()
            temp_basic.append(entering_cell)
            cycle = self.find_cycle(temp_basic, entering_cell)
            if cycle is None:
                print("No cycle found. Something went wrong.")
                return allocation
 
            # --- Step 6: Determine adjustment theta ---
            # In the cycle, the entering cell is assigned a '+' sign.
            # Alternate signs along the cycle.
            signs = [1 if idx % 2 == 0 else -1 for idx in range(len(cycle))]
            theta = np.inf
            leaving_index = None
            for idx, cell in enumerate(cycle):
                # Skip the entering cell (always with '+') when searching for theta
                if signs[idx] == -1:
                    i, j = cell
                    if allocation[i, j] < theta:
                        theta = allocation[i, j]
                        leaving_index = idx
 
            # --- Step 7: Update allocations along the cycle ---
            for idx, cell in enumerate(cycle):
                i, j = cell
                if idx == 0:
                    allocation[i, j] += theta  # entering cell gets theta added
                else:
                    allocation[i, j] += signs[idx] * theta
 
            # Remove the leaving cell from the basis by setting its allocation to zero
            if leaving_index is not None:
                leave = cycle[leaving_index]
                allocation[leave[0], leave[1]] = 0
 
    def solve(self, method='VAM'):
        if method == 'NWCR':
            self.allocation = self.NorthWestSolver()
        elif method == 'VAM':
            self.allocation = self.VogelSolver()
        elif method == 'LCM':
            self.allocation = self.LeastCostMethod()
        elif method == 'MODI':
            self.allocation = self.ModiMethod()
        else:
            raise ValueError("Invalid method. Choose NWCR, VAM, LCM, or MODI.")
        return self.allocation
 
# Example usage:
mat = [[4,8,8,0],[16,24,16,0],[8,16,24,0]]
supply = np.array([76,82,77])
demand = np.array([72,102,41,20])
tp = TransportationProblem(mat, supply, demand)
print("Vogel's Solution:")
print(tp.solve('VAM'))
print("\nModi Method Solution (Optimal):")
print(tp.solve('MODI'))