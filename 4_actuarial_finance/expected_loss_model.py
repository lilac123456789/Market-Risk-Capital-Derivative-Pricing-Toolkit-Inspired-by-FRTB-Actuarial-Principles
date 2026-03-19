import numpy as np

def expected_loss(PD, LGD, EAD):
    """
    PD: probability of default
    LGD: loss given default
    EAD: exposure at default
    """
    return PD * LGD * EAD

def lifetime_expected_loss(PD_curve, LGD, EAD):
    """
    PD_curve: array of marginal PDs for each year
    """
    cumulative_EL = 0
    for year, pd in enumerate(PD_curve, start=1):
        cumulative_EL += expected_loss(pd, LGD, EAD)
    return cumulative_EL


# Example usage
if __name__ == "__main__":
    PD_curve = [0.02, 0.03, 0.04, 0.05]
    LGD = 0.45
    EAD = 100000

    EL = lifetime_expected_loss(PD_curve, LGD, EAD)
    print("Lifetime expected loss:", EL)
