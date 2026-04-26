from pricers import OptionPricer
import numpy as np
import time
from scipy.stats import norm

class BlackScholesPricer(OptionPricer):
    """
    Analytical closed-form Black-Scholes pricer.
    European options only.
    No __init__ needed — base class handles all inputs.
    """

    def solve(self) -> "BlackScholesPricer":
        start = time.time()

        S  = np.linspace(0.0001, self.S_max, self.M + 1)  # avoid log(0)
        dS = S[1] - S[0]
        V  = np.zeros((self.M + 1, 2))
        delta_grid = np.zeros(self.M + 1)
        gamma_grid = np.zeros(self.M + 1)
        theta_grid = np.zeros(self.M + 1)

        if self.T <= 0:
            # At expiry — value equals payoff
            V[:, -1] = self._payoff(S)
        else:
            d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) \
                 / (self.sigma * np.sqrt(self.T))
            d2     = d1 - self.sigma * np.sqrt(self.T)
            pdf_d1 = norm.pdf(d1)

            if self.option_type == "call":
                V[:, -1]   = S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
                delta_grid = norm.cdf(d1)
                theta_grid = (
                    -S * pdf_d1 * self.sigma / (2 * np.sqrt(self.T))
                    - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
                )
            else:
                V[:, -1]   = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                delta_grid = norm.cdf(d1) - 1
                theta_grid = (
                    -S * pdf_d1 * self.sigma / (2 * np.sqrt(self.T))
                    + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
                )

            gamma_grid = pdf_d1 / (S * self.sigma * np.sqrt(self.T))

        self.result = {
            "price":        float(np.interp(self.S0, S, V[:, -1])),
            "delta":        float(np.interp(self.S0, S, delta_grid)),
            "gamma":        float(np.interp(self.S0, S, gamma_grid)),
            "theta":        float(np.interp(self.S0, S, theta_grid)),
            "S_grid":       S,
            "V_grid":       V,
            "delta_grid":   delta_grid,
            "gamma_grid":   gamma_grid,
            "theta_grid":   theta_grid,
            "dt":           None,
            "dS":           dS,
            "T":            self.T,
            "M":            self.M,
            "N":            None,
            "scheme":       "closed_form",
            "option_type":  self.option_type,
            "option_style": "european",
            "time_taken":   time.time() - start,
        }
        return self
