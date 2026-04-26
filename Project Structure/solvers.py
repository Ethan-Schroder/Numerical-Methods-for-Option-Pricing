import numpy as np

class SORSolver:
    """
    SOR and PSOR solvers for tridiagonal systems A x = d.

    SOR  : standard solver for European options
    PSOR : projected solver for American options (enforces V >= payoff)
    """

    def __init__(self, omega=1.2, tol=1e-8, max_iter=10_000):
        """
        omega    : relaxation factor (1 = Gauss-Seidel, 1 < omega < 2 = over-relaxation)
        tol      : convergence tolerance
        max_iter : maximum number of iterations
        """
        self.omega    = omega
        self.tol      = tol
        self.max_iter = max_iter

    def solve(self, a, b, c, d, x0=None) -> np.ndarray:
        """Solve A x = d and return solution vector x."""

        n = len(b)

        # Start from x0 if provided, otherwise start from zeros
        x = np.zeros(n) if x0 is None else x0.copy()

        for _ in range(self.max_iter):

            x_old = x.copy()  # snapshot of x before this iteration

            for i in range(n):

                # Contributions from left and right neighbours
                left  = a[i-1] * x[i-1]     if i > 0     else 0.0
                right = c[i]   * x_old[i+1] if i < n - 1 else 0.0

                # Gauss-Seidel update, then blend with omega (SOR step)
                x_gs = (d[i] - left - right) / b[i]
                x[i] = (1 - self.omega) * x_old[i] + self.omega * x_gs

            # Converged if largest change across all elements is below tolerance
            if np.max(np.abs(x - x_old)) < self.tol:
                return x

        raise RuntimeError("SOR did not converge within max_iter.")

    def solve_projected(self, a, b, c, d, payoff, x0=None) -> np.ndarray:
        """
        Solve A x = d with early exercise constraint x >= payoff.
        Used for American options.
        """

        n = len(b)

        # Start from payoff if no initial guess provided
        x = payoff.copy() if x0 is None else x0.copy()

        for _ in range(self.max_iter):

            x_old = x.copy()  # snapshot of x before this iteration

            for i in range(n):

                # Contributions from left and right neighbours
                left  = a[i-1] * x[i-1]     if i > 0     else 0.0
                right = c[i]   * x_old[i+1] if i < n - 1 else 0.0

                # Gauss-Seidel update, then blend with omega (SOR step)
                x_gs    = (d[i] - left - right) / b[i]
                relaxed = (1 - self.omega) * x_old[i] + self.omega * x_gs

                # Projection step — enforce early exercise constraint
                x[i] = max(payoff[i], relaxed)

            # Converged if largest change across all elements is below tolerance
            if np.max(np.abs(x - x_old)) < self.tol:
                return x

        raise RuntimeError("PSOR did not converge within max_iter.")
