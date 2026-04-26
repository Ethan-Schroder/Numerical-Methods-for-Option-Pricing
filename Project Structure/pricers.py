import numpy as np
import time
from abc import ABC, abstractmethod
from scipy.stats import norm
from solvers import SORSolver

class OptionPricer(ABC):
    """
    Abstract base class for all option pricers.
    Holds shared inputs, payoff helper, plotting, and repr.
    All subclasses must implement solve().
    """

    def __init__(
        self,
        option_type:  str,
        S0:    float,
        K:     float,
        T:     float,
        r:     float,
        sigma: float,
        S_max: float = 200.0,
        M:     int   = 500,
    ):
        if option_type.lower() not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

        self.option_type = option_type.lower()
        self.S0    = S0
        self.K     = K
        self.T     = T
        self.r     = r
        self.sigma = sigma
        self.S_max = S_max
        self.M     = M

        self.result = None   # populated after .solve()

    def _payoff(self, S: np.ndarray) -> np.ndarray:
        """Intrinsic value across asset grid — shared by all pricers."""
        if self.option_type == "call":
            return np.maximum(S - self.K, 0.0)
        return np.maximum(self.K - S, 0.0)

    @abstractmethod
    def solve(self) -> "OptionPricer":
        """Price the option. Must be implemented by every subclass."""
        pass

    def price_at(self, S):
        if self.result is None:
            raise RuntimeError("Call .solve() first")
        return np.interp(
            S,
            self.result["S_grid"],
            self.result["V_grid"][:, -1]
        )

    def plot(self) -> None:
        """Plot option value and Greeks — shared by FD and BS pricers."""
        import matplotlib.pyplot as plt

        if self.result is None:
            raise RuntimeError("Call .solve() before .plot()")

        r   = self.result
        pad = 5
        S_int = r["S_grid"][pad:-pad]

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(
            f"{self.__class__.__name__} — {self.option_type.capitalize()} at t=0",
            fontsize=14
        )

        plots = [
            (axs[0, 0], r["S_grid"],  r["V_grid"][:, -1],          "Option Value", "Value"),
            (axs[0, 1], S_int, r["delta_grid"][pad:-pad], "Delta",  "Delta"),
            (axs[1, 0], S_int, r["gamma_grid"][pad:-pad], "Gamma",  "Gamma"),
            (axs[1, 1], S_int, r["theta_grid"][pad:-pad], "Theta",  "Theta (per year)"),
        ]
        for ax, x, y, title, ylabel in plots:
            ax.plot(x, y)
            ax.set_title(title)
            ax.set_xlabel("Asset Price S")
            ax.set_ylabel(ylabel)
            ax.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
    def greeks_at(self, S):
        if self.result is None:
            raise RuntimeError("Call .solve() first")

        S_grid = self.result["S_grid"]

        return {
            "delta": np.interp(S, S_grid, self.result["delta_grid"]),
            "gamma": np.interp(S, S_grid, self.result["gamma_grid"]),
            "theta": np.interp(S, S_grid, self.result["theta_grid"]),
        }

    def __repr__(self) -> str:
        status = f"price={self.result['price']:.4f}" if self.result else "unsolved"
        return (
            f"{self.__class__.__name__}("
            f"{self.option_type}, S0={self.S0}, K={self.K}, "
            f"T={self.T}, r={self.r}, sigma={self.sigma} | {status})"
        )
