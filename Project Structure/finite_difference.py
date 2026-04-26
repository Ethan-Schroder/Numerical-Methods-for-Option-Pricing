from pricers import OptionPricer
import numpy as np
import time
from scipy.stats import norm
from solvers import SORSolver

class FiniteDifferencePricer(OptionPricer):
    """
    Finite difference pricer for the Black-Scholes PDE.
    Supports explicit and implicit schemes, European and American styles.
    """

    def __init__(
        self,
        option_type:  str,
        option_style: str,
        scheme:       str,
        S0:    float,
        K:     float,
        T:     float,
        r:     float,
        sigma: float,
        S_max: float,
        M:     int,
        N:     int,
    ):
        super().__init__(option_type, S0, K, T, r, sigma, S_max, M)

        if option_style.lower() not in ("european", "american"):
            raise ValueError("option_style must be 'european' or 'american'")
        if scheme.lower() not in ("explicit", "implicit"):
            raise ValueError("scheme must be 'explicit' or 'implicit'")

        self.option_style = option_style.lower()
        self.scheme       = scheme.lower()
        self.N            = N
        self._solver      = SORSolver()  # reused across all time steps

    def _apply_boundary(self, V, k, dt) -> None:
        """Set boundary conditions at S=0 and S=S_max."""
        tau = (k + 1) * dt
        if self.option_type == "call":
            V[0, k + 1]      = 0.0
            V[self.M, k + 1] = self.S_max - self.K * np.exp(-self.r * tau)
        else:
            V[0, k + 1]      = self.K * np.exp(-self.r * tau)
            V[self.M, k + 1] = 0.0

    def _solve_explicit(self, V, S, dt, dS) -> None:
        """Explicit finite difference time stepping."""
        stability_limit = dS**2 / (self.sigma**2 * self.S_max**2)
        if dt > stability_limit:
            raise ValueError("Explicit scheme unstable: reduce dt or increase M.")

        for k in range(self.N):
            for i in range(1, self.M):
                a = 0.5 * dt * (self.sigma**2 * i**2 - self.r * i)
                b = 1.0 - dt * (self.sigma**2 * i**2 + self.r)
                c = 0.5 * dt * (self.sigma**2 * i**2 + self.r * i)

                V_cont = a * V[i - 1, k] + b * V[i, k] + c * V[i + 1, k]

                if self.option_style == "american":
                    intrinsic  = S[i] - self.K if self.option_type == "call" else self.K - S[i]
                    V[i, k+1] = max(V_cont, intrinsic)
                else:
                    V[i, k+1] = V_cont

            self._apply_boundary(V, k, dt)

    def _solve_implicit(self, V, S, dt) -> None:
        """Implicit (backward Euler) finite difference time stepping."""
        for k in range(self.N):
            self._apply_boundary(V, k, dt)

            i      = np.arange(1, self.M)
            a_full = -0.5 * dt * (self.sigma**2 * i**2 - self.r * i)
            b      =  1.0 + dt * (self.sigma**2 * i**2 + self.r)
            c_full = -0.5 * dt * (self.sigma**2 * i**2 + self.r * i)

            rhs = V[1:self.M, k].copy()
            rhs[0]  -= a_full[0]  * V[0, k + 1]
            rhs[-1] -= c_full[-1] * V[self.M, k + 1]

            if self.option_style == "european":
                V[1:self.M, k+1] = self._solver.solve(
                    a_full[1:], b, c_full[:-1], rhs, x0=V[1:self.M, k]
                )
            else:
                payoff = self._payoff(S[1:self.M])
                V[1:self.M, k+1] = self._solver.solve_projected(
                    a_full[1:], b, c_full[:-1], rhs, payoff, x0=V[1:self.M, k]
                )

    def _compute_greeks(self, V, S, dS, dt) -> tuple:
        """Compute point and grid Greeks at t=0."""
        i0    = np.clip(np.searchsorted(S, self.S0), 1, self.M - 1)
        price = np.interp(self.S0, S, V[:, -1])
        delta = (V[i0 + 1, -1] - V[i0 - 1, -1]) / (2.0 * dS)
        gamma = (V[i0 + 1, -1] - 2.0 * V[i0, -1] + V[i0 - 1, -1]) / dS**2
        theta = -(V[i0, -1] - V[i0, -2]) / dt

        delta_grid = np.zeros(self.M + 1)
        gamma_grid = np.zeros(self.M + 1)
        theta_grid = np.zeros(self.M + 1)

        for i in range(1, self.M):
            delta_grid[i] = (V[i + 1, -1] - V[i - 1, -1]) / (2.0 * dS)
            gamma_grid[i] = (V[i + 1, -1] - 2.0 * V[i, -1] + V[i - 1, -1]) / dS**2
            theta_grid[i] = -(V[i, -1] - V[i, -2]) / dt

        return price, delta, gamma, theta, delta_grid, gamma_grid, theta_grid

    def solve(self) -> "FiniteDifferencePricer":
        start = time.time()
        dS = self.S_max / self.M
        dt = self.T / self.N
        S  = np.linspace(0, self.S_max, self.M + 1)
        V  = np.zeros((self.M + 1, self.N + 1))

        V[:, 0] = self._payoff(S)   # terminal condition

        if self.scheme == "explicit":
            self._solve_explicit(V, S, dt, dS)
        else:
            self._solve_implicit(V, S, dt)

        price, delta, gamma, theta, dg, gg, tg = self._compute_greeks(V, S, dS, dt)

        self.result = {
            "price":        price,
            "delta":        delta,
            "gamma":        gamma,
            "theta":        theta,
            "S_grid":       S,
            "V_grid":       V,
            "delta_grid":   dg,
            "gamma_grid":   gg,
            "theta_grid":   tg,
            "dt":           dt,
            "dS":           dS,
            "T":            self.T,
            "M":            self.M,
            "N":            self.N,
            "scheme":       self.scheme,
            "option_type":  self.option_type,
            "option_style": self.option_style,
            "time_taken":   time.time() - start,
        }
        return self
