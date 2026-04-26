from pricers import OptionPricer
import numpy as np
import time
from scipy.stats import norm
from solvers import SORSolver

class CrankNicolsonPricer(OptionPricer):
    """
    Crank-Nicolson finite difference pricer.
    Second-order accurate in both time and space.
    Supports European and American options.
    """

    def __init__(
        self,
        option_type:  str,
        option_style: str,
        S0:    float,
        K:     float,
        T:     float,
        r:     float,
        sigma: float,
        S_max: float,
        M:     int,
        N:     int,
        omega: float = 1.2,
    ):
        super().__init__(option_type, S0, K, T, r, sigma, S_max, M)

        if option_style.lower() not in ("european", "american"):
            raise ValueError("option_style must be 'european' or 'american'")

        self.option_style = option_style.lower()
        self.N            = N
        self._solver      = SORSolver(omega=omega)

    def _apply_boundary(self, V_new, n, dt) -> None:
        """Set boundary conditions at S=0 and S=S_max."""
        tau = self.T - n * dt
        if self.option_type == "call":
            V_new[0]  = 0.0
            V_new[-1] = self.S_max - self.K * np.exp(-self.r * tau)
        else:
            V_new[0]  = self.K * np.exp(-self.r * tau)
            V_new[-1] = 0.0

    def _build_lhs_rhs(self, V_old, S, dS, dt) -> tuple:
        """Build tridiagonal LHS and RHS for the CN system."""
        i  = np.arange(1, self.M)
        Si = S[i]

        # CN coefficients — average of explicit and implicit contributions
        alpha = 0.25 * dt * (self.sigma**2 * Si**2 / dS**2 - self.r * Si / dS)
        beta  = -0.5 * dt * (self.sigma**2 * Si**2 / dS**2 + self.r)
        gamma = 0.25 * dt * (self.sigma**2 * Si**2 / dS**2 + self.r * Si / dS)

        a = -alpha[1:]    # sub-diagonal
        b =  1 - beta     # main diagonal
        c = -gamma[:-1]   # super-diagonal

        rhs = (
            alpha * V_old[:-2]
            + (1 + beta) * V_old[1:-1]
            + gamma * V_old[2:]
        )
        return a, b, c, rhs

    def _compute_greeks_grid(self, V, S, dS) -> tuple:
        """Compute Greeks across full spatial grid."""
        delta_grid = np.zeros(self.M + 1)
        gamma_grid = np.zeros(self.M + 1)
        theta_grid = np.zeros(self.M + 1)

        for i in range(1, self.M):
            delta_grid[i] = (V[i + 1] - V[i - 1]) / (2.0 * dS)
            gamma_grid[i] = (V[i + 1] - 2.0 * V[i] + V[i - 1]) / dS**2
            # Theta from BS PDE (CN-consistent)
            theta_grid[i] = (
                -0.5 * self.sigma**2 * S[i]**2 * gamma_grid[i]
                - self.r * S[i] * delta_grid[i]
                + self.r * V[i]
            )
        return delta_grid, gamma_grid, theta_grid

    def solve(self) -> "CrankNicolsonPricer":
        start = time.time()

        dS    = self.S_max / self.M
        dt    = self.T / self.N
        S     = np.linspace(0, self.S_max, self.M + 1)
        V_old = self._payoff(S)
        V_new = np.zeros(self.M + 1)

        # Backward time-stepping
        for n in range(self.N):
            self._apply_boundary(V_new, n, dt)

            a, b, c, rhs = self._build_lhs_rhs(V_old, S, dS, dt)
            rhs[0]  -= a[0]  * V_new[0]
            rhs[-1] -= c[-1] * V_new[-1]

            if self.option_style == "european":
                V_new[1:-1] = self._solver.solve(a, b, c, rhs, x0=V_old[1:-1])
            else:
                payoff      = self._payoff(S[1:-1])
                V_new[1:-1] = self._solver.solve_projected(a, b, c, rhs, payoff, x0=V_old[1:-1])

            V_old[:] = V_new[:]  # roll forward

        # Point Greeks at S0
        idx       = np.clip(np.searchsorted(S, self.S0), 1, self.M - 1)
        price     = np.interp(self.S0, S, V_old)
        delta     = (V_old[idx + 1] - V_old[idx - 1]) / (2.0 * dS)
        gamma_val = (V_old[idx + 1] - 2.0 * V_old[idx] + V_old[idx - 1]) / dS**2
        theta     = (
            -0.5 * self.sigma**2 * S[idx]**2 * gamma_val
            - self.r * S[idx] * delta
            + self.r * V_old[idx]
        )

        delta_grid, gamma_grid, theta_grid = self._compute_greeks_grid(V_old, S, dS)

        # Wrap in 2D so base class plot() works consistently
        V_grid        = np.zeros((self.M + 1, 2))
        V_grid[:, -1] = V_old

        self.result = {
            "price":        price,
            "delta":        delta,
            "gamma":        gamma_val,
            "theta":        theta,
            "S_grid":       S,
            "V_grid":       V_grid,
            "delta_grid":   delta_grid,
            "gamma_grid":   gamma_grid,
            "theta_grid":   theta_grid,
            "dt":           dt,
            "dS":           dS,
            "T":            self.T,
            "M":            self.M,
            "N":            self.N,
            "scheme":       "crank_nicolson",
            "option_type":  self.option_type,
            "option_style": self.option_style,
            "time_taken":   time.time() - start,
        }
        return self
