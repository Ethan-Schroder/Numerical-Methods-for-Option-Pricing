from pricers import OptionPricer
import numpy as np
import time


class MonteCarloPricer(OptionPricer):
    """
    Monte Carlo pricer for European options.

    Features:
        - Common Random Numbers (CRN) for stable Greeks
        - Pathwise Delta estimator
        - Central bump Gamma and Theta
        - Pseudo spatial grid for plotting
        - 6-panel plot output
    """

    def __init__(
        self,
        option_type: str,
        S0:       float,
        K:        float,
        T:        float,
        r:        float,
        sigma:    float,
        n_paths:  int,
        n_steps:  int   = 100,
        eps_S:    float = 1e-2,   # spot bump size for Gamma
        eps_T:    float = 1e-4,   # time bump size for Theta
    ):
        # Pass shared parameters up to OptionPricer base class
        super().__init__(option_type, S0, K, T, r, sigma)
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.eps_S   = eps_S
        self.eps_T   = eps_T

    # ================================================================
    # Private helpers
    # ================================================================

    def _simulate_terminal(self, S0: float, T: float, Z: np.ndarray) -> np.ndarray:
        """
        Simulate terminal asset prices S(T) using the GBM closed form.

        S(T) = S0 * exp( (r - 0.5σ²)T + σ√T Z )

        Accepts external Z so the same random numbers can be reused
        across bumped scenarios (Common Random Numbers).
        """
        return S0 * np.exp(
            (self.r - 0.5 * self.sigma**2) * T
            + self.sigma * np.sqrt(T) * Z
        )

    def _simulate_paths(self, Z_paths: np.ndarray) -> np.ndarray:
        """
        Simulate full asset price paths using Euler-Maruyama discretisation.
        Used only for the paths plot — not for pricing or Greeks.

        Returns S of shape (n_paths, n_steps + 1).
        """
        dt = self.T / self.n_steps
        S  = np.zeros((self.n_paths, self.n_steps + 1))
        S[:, 0] = self.S0

        for t in range(1, self.n_steps + 1):
            S[:, t] = S[:, t - 1] * np.exp(
                (self.r - 0.5 * self.sigma**2) * dt
                + self.sigma * np.sqrt(dt) * Z_paths[:, t - 1]
            )
        return S

    def _point_greeks(self, s: float, discount: float, Z: np.ndarray) -> tuple:
        """
        Compute price, Delta, Gamma, Theta at a single spot value s.
        Reuses Z (CRN) for all bump calculations so noise cancels cleanly.
        """
        # Simulate terminal prices at s, s+eps, s-eps using same Z
        ST_s      = self._simulate_terminal(s,              self.T, Z)
        ST_s_up   = self._simulate_terminal(s + self.eps_S, self.T, Z)
        ST_s_down = self._simulate_terminal(s - self.eps_S, self.T, Z)

        # Discounted expected payoffs
        price_s    = discount * np.mean(self._payoff(ST_s))
        price_s_up = discount * np.mean(self._payoff(ST_s_up))
        price_s_dn = discount * np.mean(self._payoff(ST_s_down))

        # Pathwise Delta — differentiates payoff through indicator function
        indicator = (ST_s > self.K).astype(float) if self.option_type == "call" \
                    else (ST_s < self.K).astype(float)
        delta_s = discount * np.mean(indicator * ST_s / s)
        if self.option_type == "put":
            delta_s *= -1

        # Gamma — central finite difference on price w.r.t. spot
        gamma_s = (price_s_up - 2 * price_s + price_s_dn) / self.eps_S**2

        # Theta — derived from BS PDE so it's consistent with Delta and Gamma
        # ∂V/∂t = -0.5σ²S²Γ - rSΔ + rV
        theta_s = (
            -0.5 * self.sigma**2 * s**2 * gamma_s
            - self.r * s * delta_s
            + self.r * price_s
        )

        return price_s, delta_s, gamma_s, theta_s

    # ================================================================
    # solve()
    # ================================================================

    def solve(self) -> "MonteCarloPricer":
        start = time.time()

        discount = np.exp(-self.r * self.T)

        # ----------------------------------------------------------------
        # ONE set of random numbers shared across ALL calculations (CRN)
        # This ensures bumped scenarios differ only in S0/T — not in noise
        # ----------------------------------------------------------------
        Z = np.random.randn(self.n_paths)

        # ----------------------------------------------------------------
        # Point estimates at S0
        # ----------------------------------------------------------------

        # Terminal prices at base spot — renamed ST_base to protect
        # from being overwritten inside the pseudo-grid loop below
        ST_base = self._simulate_terminal(self.S0, self.T, Z)
        price   = discount * np.mean(self._payoff(ST_base))

        # Delta — pathwise estimator
        indicator = (ST_base > self.K).astype(float) if self.option_type == "call" \
                    else (ST_base < self.K).astype(float)
        delta = discount * np.mean(indicator * ST_base / self.S0)
        if self.option_type == "put":
            delta *= -1

        # Gamma — central bump using CRN
        ST_up      = self._simulate_terminal(self.S0 + self.eps_S, self.T, Z)
        ST_down    = self._simulate_terminal(self.S0 - self.eps_S, self.T, Z)
        price_up   = discount * np.mean(self._payoff(ST_up))
        price_down = discount * np.mean(self._payoff(ST_down))
        gamma      = (price_up - 2 * price + price_down) / self.eps_S**2

        # Theta — central bump in time using CRN
        ST_T_plus     = self._simulate_terminal(self.S0, self.T + self.eps_T, Z)
        ST_T_minus    = self._simulate_terminal(self.S0, self.T - self.eps_T, Z)
        price_T_plus  = np.exp(-self.r * (self.T + self.eps_T)) * np.mean(self._payoff(ST_T_plus))
        price_T_minus = np.exp(-self.r * (self.T - self.eps_T)) * np.mean(self._payoff(ST_T_minus))
        theta         = (price_T_minus - price_T_plus) / (2 * self.eps_T)

        # ----------------------------------------------------------------
        # Pseudo spatial grid — price and Greeks across range of spot values
        # Used for plotting only, not for the point estimates above
        # Same Z reused across all grid points (CRN)
        # ----------------------------------------------------------------
        S_grid     = np.linspace(1, self.S_max, 100)
        V_grid     = []
        delta_grid = []
        gamma_grid = []
        theta_grid = []

        for s in S_grid:
            price_s, delta_s, gamma_s, theta_s = self._point_greeks(s, discount, Z)
            V_grid.append(price_s)
            delta_grid.append(delta_s)
            gamma_grid.append(gamma_s)
            theta_grid.append(theta_s)

        # ----------------------------------------------------------------
        # Simulate full paths — uses fresh randomness (for realistic plot)
        # Separate from Z so the paths aren't constrained to CRN structure
        # ----------------------------------------------------------------
        Z_paths = np.random.randn(self.n_paths, self.n_steps)
        S_paths = self._simulate_paths(Z_paths)

        self.result = {
            # Point estimates at S0
            "price":      float(price),
            "delta":      float(delta),
            "gamma":      float(gamma),
            "theta":      float(theta),

            # Spatial grid (for plotting)
            "S_grid":     S_grid,
            "V_grid":     np.array(V_grid).reshape(-1, 1),
            "delta_grid": np.array(delta_grid),
            "gamma_grid": np.array(gamma_grid),
            "theta_grid": np.array(theta_grid),

            # Path data (for plotting)
            "S_paths":    S_paths,
            "S_terminal": ST_base,

            "time_taken": time.time() - start,
        }
        return self

    # ================================================================
    # 6-panel plot
    # ================================================================

    def plot(self) -> None:
        """
        6-panel plot:
            [0,0] Option Value      [0,1] Delta
            [1,0] Gamma             [1,1] Theta
            [2,0] Simulated Paths   [2,1] Terminal Distribution
        """
        import matplotlib.pyplot as plt

        if self.result is None:
            raise RuntimeError("Call .solve() before .plot()")

        r         = self.result
        pad       = min(5, len(r["S_grid"]) // 10)
        S_int     = r["S_grid"][pad:-pad]
        S_paths   = r["S_paths"]
        ST        = r["S_terminal"]
        time_grid = np.linspace(0, self.T, self.n_steps + 1)

        fig, axs = plt.subplots(3, 2, figsize=(12, 14))
        fig.suptitle(
            f"MonteCarloPricer — {self.option_type.capitalize()} Option",
            fontsize=14
        )

        # Panel definitions — (ax, x data, y data, title, ylabel)
        panels = [
            (axs[0, 0], r["S_grid"], r["V_grid"][:, -1],          "Option Value", "Value"),
            (axs[0, 1], S_int,       r["delta_grid"][pad:-pad],    "Delta",        "Delta"),
            (axs[1, 0], S_int,       r["gamma_grid"][pad:-pad],    "Gamma",        "Gamma"),
            (axs[1, 1], S_int,       r["theta_grid"][pad:-pad],    "Theta",        "Theta (per year)"),
        ]
        for ax, x, y, title, ylabel in panels:
            ax.plot(x, y)
            ax.set_title(title)
            ax.set_xlabel("Asset Price S")
            ax.set_ylabel(ylabel)
            ax.grid(True)

        # Simulated paths — plot first 50 to avoid clutter
        for i in range(min(50, self.n_paths)):
            axs[2, 0].plot(time_grid, S_paths[i], alpha=0.1, linewidth=0.8)
        axs[2, 0].axhline(self.K, linestyle="--", color="black", label=f"Strike K={self.K}")
        axs[2, 0].set_title("Simulated Paths")
        axs[2, 0].set_xlabel("Time")
        axs[2, 0].set_ylabel("Asset Price S")
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        # Terminal price distribution
        axs[2, 1].hist(ST, bins=50, density=True, color="steelblue", alpha=0.7)
        axs[2, 1].axvline(self.K, linestyle="--", color="black", label=f"Strike K={self.K}")
        axs[2, 1].set_title("Terminal Distribution")
        axs[2, 1].set_xlabel("S(T)")
        axs[2, 1].set_ylabel("Density")
        axs[2, 1].legend()
        axs[2, 1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
