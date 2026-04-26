# Numerical Methods for Option Pricing

A Python library for pricing vanilla options and computing their Greeks (sensitivities) using five distinct methods. The library is built with object-oriented design and handles both European and American contracts.

> Developed as part of a dissertation on quantitative finance and numerical analysis.

---

## Methods Implemented

| Method | European | American | Accuracy |
|---|---|---|---|
| Black–Scholes (Closed-Form) | ✅ | ❌ | Exact |
| Explicit Finite Difference | ✅ | ✅ | O(Δt, ΔS²) |
| Implicit Finite Difference | ✅ | ✅ | O(Δt, ΔS²) |
| Crank–Nicolson | ✅ | ✅ | O(Δt², ΔS²) |
| Monte Carlo Simulation | ✅ | ❌ | O(1/√N) |

---

## Mathematical Background

### Asset Price Dynamics

Asset prices are modelled as **Geometric Brownian Motion (GBM)**:

```
dS = μS dt + σS dX
```

where `μ` is the drift, `σ` is the volatility, and `dX` is a Wiener process increment.

### The Black–Scholes PDE

Using Itô's Lemma and a delta-hedging argument to eliminate stochastic risk, the option price `V(S, t)` satisfies:

```
∂V/∂t + ½σ²S² ∂²V/∂S² + rS ∂V/∂S - rV = 0
```

For European options, this has an analytical solution. For American options — where early exercise is sometimes optimal — numerical methods are required.

### Closed-Form Solution (European Call)

```
C = S·N(d₁) - K·e^{-rT}·N(d₂)

d₁ = [ln(S/K) + (r + ½σ²)T] / (σ√T)
d₂ = d₁ - σ√T
```

European puts follow from put–call parity: `P = C - S + Ke^{-rT}`.

---

## Project Structure

```
├── pricers.py           # Abstract base class (OptionPricer)
├── black_scholes.py     # Closed-form analytical pricer
├── finite_difference.py # Explicit and implicit FD schemes
├── crank_nicolson.py    # Crank–Nicolson scheme
├── monte_carlo.py       # Monte Carlo simulation
└── solvers.py           # SOR and PSOR iterative solvers
```

---

## Quickstart

### European Call — Black–Scholes

```python
from black_scholes import BlackScholesPricer

pricer = BlackScholesPricer(
    option_type="call",
    S0=100, K=100, T=1.0, r=0.05, sigma=0.2
)
pricer.solve()

print(pricer)
# BlackScholesPricer(call, S0=100, K=100, T=1.0, r=0.05, sigma=0.2 | price=10.4506)

print(pricer.result["delta"])  # 0.6368
print(pricer.result["gamma"])  # 0.0188
```

### American Put — Crank–Nicolson

```python
from crank_nicolson import CrankNicolsonPricer

pricer = CrankNicolsonPricer(
    option_type="put",
    option_style="american",
    S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
    S_max=200, M=500, N=500
)
pricer.solve()
pricer.plot()
```

### European Put — Monte Carlo

```python
from monte_carlo import MonteCarloPricer

pricer = MonteCarloPricer(
    option_type="put",
    S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
    n_paths=100_000, n_steps=100
)
pricer.solve()
pricer.plot()  # 6-panel output including simulated paths and terminal distribution
```

### American Put — Explicit Finite Difference

```python
from finite_difference import FiniteDifferencePricer

pricer = FiniteDifferencePricer(
    option_type="put",
    option_style="american",
    scheme="explicit",
    S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
    S_max=200, M=200, N=5000
)
pricer.solve()
```

---

## Greeks

All pricers expose the following Greeks at the target spot price `S0`, as well as across the full spatial grid for plotting:

| Greek | Definition | Method |
|---|---|---|
| **Delta** | `∂V/∂S` | Analytical (BS) / Pathwise (MC) / FD central difference |
| **Gamma** | `∂²V/∂S²` | Analytical (BS) / Central bump (MC) / FD second difference |
| **Theta** | `∂V/∂t` | Analytical (BS) / BS PDE identity (CN, MC) / Backward difference (FD) |

```python
# Point Greeks at S0
greeks = pricer.greeks_at(S=100)
print(greeks)
# {'delta': 0.637, 'gamma': 0.019, 'theta': -6.41}

# Price at an arbitrary spot value
price = pricer.price_at(S=105)
```

---

## Numerical Schemes

### Explicit Finite Difference
Straightforward time-stepping using known values at each step. Subject to a **stability constraint**: `Δt ≤ ΔS² / (σ²S_max²)`. Unstable if this is violated — an error is raised automatically.

### Implicit Finite Difference (Backward Euler)
Unconditionally stable. Requires solving a tridiagonal linear system at each time step using **SOR** (Successive Over-Relaxation) for European options, or **PSOR** (Projected SOR) for American options to enforce the early exercise constraint.

### Crank–Nicolson
Averages the explicit and implicit operators, achieving **second-order accuracy in time**. Uses PSOR with a configurable relaxation factor `ω`. The recommended default is `ω = 1.2`.

### Monte Carlo
Simulates terminal asset prices using the GBM closed form `S(T) = S₀ · exp((r - ½σ²)T + σ√T · Z)`. **Common Random Numbers (CRN)** are used across all bumped scenarios so that noise cancels when computing Greeks by finite difference.

---

## Solvers (`solvers.py`)

The `SORSolver` class implements two iterative methods for tridiagonal systems `Ax = d`:

- `solve(a, b, c, d)` — standard SOR for European options
- `solve_projected(a, b, c, d, payoff)` — PSOR for American options, enforcing `V ≥ intrinsic value` at every node

Default parameters: `ω = 1.2`, `tol = 1e-8`, `max_iter = 10,000`.

---

## Dependencies

```
numpy
scipy
matplotlib
```

Install with:
```bash
pip install numpy scipy matplotlib
```

---

## Key Parameters

| Parameter | Description |
|---|---|
| `S0` | Current asset price |
| `K` | Strike price |
| `T` | Time to expiry (years) |
| `r` | Risk-free interest rate |
| `sigma` | Volatility (annualised) |
| `S_max` | Upper boundary for asset price grid |
| `M` | Number of spatial grid steps |
| `N` | Number of time steps (FD/CN only) |
| `n_paths` | Number of simulation paths (MC only) |

---

## References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy.
- Merton, R. C. (1973). *Theory of Rational Option Pricing.* Bell Journal of Economics.
- Hull, J. C. (2018). *Options, Futures, and Other Derivatives.* Pearson.
- Shreve, S. E. (2004). *Stochastic Calculus for Finance II.* Springer.
