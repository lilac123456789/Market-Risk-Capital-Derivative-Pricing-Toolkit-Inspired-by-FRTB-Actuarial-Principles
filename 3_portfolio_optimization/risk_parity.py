import numpy as np

def risk_parity(cov, tol=1e-8, max_iter=1000):
    """
    Solve Risk Parity (Equal Risk Contribution) portfolio.
    """

    n = len(cov)
    w = np.ones(n) / n  # start with equal weights

    for _ in range(max_iter):
        marginal_risk = np.dot(cov, w)
        risk_contrib = w * marginal_risk

        diff = risk_contrib - np.mean(risk_contrib)

        if np.linalg.norm(diff) < tol:
            break

        w -= 0.01 * diff
        w = np.abs(w)
        w /= w.sum()

    return w


# Example usage
if __name__ == "__main__":
    cov = np.array([
        [0.04, 0.01, 0.008],
        [0.01, 0.09, 0.006],
        [0.008, 0.006, 0.025]
    ])

    w = risk_parity(cov)
    print("Risk parity portfolio weights:", w)
