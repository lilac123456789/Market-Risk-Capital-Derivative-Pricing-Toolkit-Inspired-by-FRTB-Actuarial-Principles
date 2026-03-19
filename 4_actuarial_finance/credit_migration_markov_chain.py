import numpy as np

class CreditMarkovModel:
    def __init__(self, transition_matrix, states):
        """
        transition_matrix: Square matrix (NxN)
        states: List of rating labels
        """
        self.P = np.array(transition_matrix)
        self.states = states

    def n_step_transition(self, n):
        """
        Compute n-step transition matrix via matrix power.
        """
        return np.linalg.matrix_power(self.P, n)

    def migrate(self, initial_state, n):
        """
        Probability distribution after n periods.
        """
        idx = self.states.index(initial_state)
        e0 = np.zeros(len(self.states))
        e0[idx] = 1
        return np.dot(e0, self.n_step_transition(n))


# Example usage
if __name__ == "__main__":
    # Simple 4-state rating system
    states = ["AAA", "AA", "A", "Default"]
    P = [
        [0.90, 0.08, 0.01, 0.01],
        [0.02, 0.92, 0.05, 0.01],
        [0.01, 0.03, 0.94, 0.02],
        [0.00, 0.00, 0.00, 1.00]  # absorbing
    ]

    model = CreditMarkovModel(P, states)
    dist_3_years = model.migrate("A", 3)
    print("Rating distribution after 3 years from A:", dist_3_years)
