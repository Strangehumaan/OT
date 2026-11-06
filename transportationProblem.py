import numpy as np
 
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
    def _penalties(cost_matrix, axis):
        """Difference between the two smallest costs along `axis` (1 = per row, 0 = per column)."""
        sorted_costs = np.sort(cost_matrix, axis=axis)
        if sorted_costs.shape[axis] < 2:
            # Only one cell left in the line: use its cost so it is allocated first.
            return np.take(sorted_costs, 0, axis=axis).copy()
        diff = np.take(sorted_costs, 1, axis=axis) - np.take(sorted_costs, 0, axis=axis)
        return np.where(np.isnan(diff), 0.0, diff)  # inf - inf from forbidden routes

    @staticmethod
    def row_penalties(cost_matrix):
        return TransportationProblem._penalties(cost_matrix, axis=1)

    @staticmethod
    def col_penalties(cost_matrix):
        return TransportationProblem._penalties(cost_matrix, axis=0)

    def VogelSolver(self):
        cost_mat = self.cost_matrix.astype(float)
        supply = self.supply.astype(float)
        demand = self.demand.astype(float)
        allocation = np.zeros(cost_mat.shape)
        active_rows = supply > 0
        active_cols = demand > 0

        # Penalties are computed only over rows/columns that still have supply/demand.
        # Considering exhausted lines lets one with penalty 0 be re-selected forever.
        while active_rows.any() and active_cols.any():
            rows = np.flatnonzero(active_rows)
            cols = np.flatnonzero(active_cols)
            sub = cost_mat[np.ix_(rows, cols)]
            row_pen = self.row_penalties(sub)
            col_pen = self.col_penalties(sub)

            if row_pen.max() >= col_pen.max():
                r = int(np.argmax(row_pen))
                c = int(np.argmin(sub[r]))
            else:
                c = int(np.argmax(col_pen))
                r = int(np.argmin(sub[:, c]))
            row, col = rows[r], cols[c]

            alloc = min(supply[row], demand[col])
            allocation[row, col] = alloc
            supply[row] -= alloc
            demand[col] -= alloc
            if supply[row] <= 0:
                active_rows[row] = False
            if demand[col] <= 0:
                active_cols[col] = False

        return allocation

    def LeastCostMethod(self):
        cost_mat = self.cost_matrix.astype(float)  # np.inf can't be stored in an int array
        supply = self.supply.astype(float)
        demand = self.demand.astype(float)
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
        cells = set(basic_cells) | {start}

        def search(path, move_in_row):
            current = path[-1]
            for cell in cells:
                if cell == current:
                    continue
                # Move along the current row, or along the current column.
                if move_in_row and cell[0] != current[0]:
                    continue
                if not move_in_row and cell[1] != current[1]:
                    continue
                if cell == start:
                    # Close the loop: needs >= 4 cells and an even count so the
                    # closing move is perpendicular to the first move.
                    if len(path) >= 4 and len(path) % 2 == 0:
                        return path  # start is NOT repeated at the end
                    continue
                if cell in path:
                    continue
                result = search(path + [cell], not move_in_row)
                if result is not None:
                    return result
            return None

        # The loop may leave the start cell along its row or along its column.
        return search([start], True) or search([start], False)
 
    def _complete_basis(self, allocation):
        """
        Return a set of exactly m+n-1 basic cells forming a spanning tree of the
        transportation graph. Cells with positive allocation are always included;
        zero-allocation cells are added (cheapest first, skipping any that would
        close a loop) to repair a degenerate solution.
        """
        m, n = allocation.shape
        parent = list(range(m + n))  # union-find over rows 0..m-1 and columns m..m+n-1

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        basis = set()
        positive = [(i, j) for i in range(m) for j in range(n) if allocation[i, j] > 1e-12]
        zeros = sorted(
            ((i, j) for i in range(m) for j in range(n) if allocation[i, j] <= 1e-12),
            key=lambda c: self.cost_matrix[c],
        )
        for (i, j) in positive + zeros:
            ri, rj = find(i), find(m + j)
            if ri != rj:
                parent[ri] = rj
                basis.add((i, j))
            if len(basis) == m + n - 1:
                break
        return basis

    def ModiMethod(self, max_iter=1000):
        # Get initial feasible solution using Vogel's method
        allocation = self.VogelSolver()
        m, n = allocation.shape
        cost = self.cost_matrix.astype(float)
        eps = 1e-9

        # The basis must always hold exactly m+n-1 cells (a spanning tree), even when
        # some of them carry zero allocation (degenerate case). Deriving it from
        # "allocation > 0" would leave u/v undefined.
        basis = self._complete_basis(allocation)

        for _ in range(max_iter):
            # --- Step 1: Compute dual variables u and v from the basic cells ---
            u = np.full(m, np.nan)
            v = np.full(n, np.nan)
            u[0] = 0
            updated = True
            while updated:
                updated = False
                for (i, j) in basis:
                    if not np.isnan(u[i]) and np.isnan(v[j]):
                        v[j] = cost[i, j] - u[i]
                        updated = True
                    elif np.isnan(u[i]) and not np.isnan(v[j]):
                        u[i] = cost[i, j] - v[j]
                        updated = True

            # --- Step 2/4: Reduced costs of non-basic cells; pick most negative ---
            min_val = -eps
            entering_cell = None
            for i in range(m):
                for j in range(n):
                    if (i, j) in basis:
                        continue
                    d = cost[i, j] - (u[i] + v[j])
                    if d < min_val:
                        min_val = d
                        entering_cell = (i, j)

            # --- Step 3: Optimal if no negative reduced cost ---
            if entering_cell is None:
                return allocation

            # --- Step 5: Find the closed loop through the entering cell ---
            cycle = self.find_cycle(list(basis), entering_cell)
            if cycle is None:
                raise RuntimeError("MODI: no closed loop found for the entering cell.")

            # --- Step 6: theta = smallest allocation among the '-' cells ---
            # cycle[0] is the entering cell ('+'); signs alternate along the loop.
            minus_cells = [cycle[k] for k in range(1, len(cycle), 2)]
            leaving = min(minus_cells, key=lambda c: allocation[c])
            theta = allocation[leaving]

            # --- Step 7: Shift theta around the loop and update the basis ---
            for k, (i, j) in enumerate(cycle):
                allocation[i, j] += theta if k % 2 == 0 else -theta
            allocation[leaving] = 0.0  # remove float residue
            basis.remove(leaving)
            basis.add(entering_cell)

        raise RuntimeError(f"MODI did not converge within {max_iter} iterations.")

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
 
if __name__ == "__main__":
    # Example usage:
    mat = [[4,8,8,0],[16,24,16,0],[8,16,24,0]]
    supply = np.array([76,82,77])
    demand = np.array([72,102,41,20])
    tp = TransportationProblem(mat, supply, demand)
    print("Vogel's Solution:")
    print(tp.solve('VAM'))
    print("\nModi Method Solution (Optimal):")
    print(tp.solve('MODI'))