import numpy as np
from scipy.stats import norm

def black_scholes(S, K, r, sigma, T, option_type="call"):
    """
    S: spot price
    K: strike price
    r: risk-free rate
    sigma: volatility
    T: time to maturity (years)
    option_type: "call" or "put"
    """

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


# Example usage
if __name__ == "__main__":
    S = 100     # Spot
    K = 100     # Strike
    r = 0.05    # Rate
    sigma = 0.2 # Vol
    T = 1       # 1 year

    c = black_scholes(S, K, r, sigma, T, "call")
    p = black_scholes(S, K, r, sigma, T, "put")

    print("Call price:", c)
    print("Put price:", p)
