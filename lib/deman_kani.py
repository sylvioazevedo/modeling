import numpy as np

class DemanKaniModel:

    def __init__(self, S0, K, rf, T, vol):
        self.S0 = S0
        self.K = K
        self.rf = rf
        self.T = T
        self.vol = vol    

    def build_tree(self, num_steps):
        dt = self.T / num_steps
        u = np.exp(self.vol * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(self.rf * dt) - d) / (u - d)  # Risk-neutral probability

        # Initialize the tree
        tree = np.zeros((num_steps + 1, num_steps + 1))
        tree[0, 0] = self.S0

        # Build the tree
        for i in range(1, num_steps + 1):
            for j in range(i + 1):
                tree[j, i] = self.S0 * (u ** (i - j)) * (d ** j)

        return tree, p

    def price_option(self, num_steps, option_type="call"):
        
        tree, p = self.build_tree(num_steps)
        dt = self.T / num_steps
        discount = np.exp(-self.rf * dt)

        # Initialize option values at maturity
        option_values = np.zeros((num_steps + 1, num_steps + 1))
        for j in range(num_steps + 1):
            if option_type == "call":
                option_values[j, num_steps] = max(0, tree[j, num_steps] - self.K)
            elif option_type == "put":
                option_values[j, num_steps] = max(0, self.K - tree[j, num_steps])

        # Backward induction
        for i in range(num_steps - 1, -1, -1):
            for j in range(i + 1):
                option_values[j, i] = discount * (p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1])

        return option_values[0, 0]
    

    def build_tree_with_probabilities(self, num_steps):

        dt = self.T / num_steps
        u = np.exp(self.vol * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(self.rf * dt) - d) / (u - d)  # Risk-neutral probability

        # Initialize the tree
        tree = np.zeros((num_steps + 1, num_steps + 1))
        probabilities = np.zeros((num_steps + 1, num_steps + 1))
        arrow_debreu_probs = np.zeros((num_steps + 1, num_steps + 1))
        tree[0, 0] = self.S0
        probabilities[0, 0] = 1.0  # Root node has probability 1

        # Build the tree, probabilities, and Arrow-Debreu probabilities
        for i in range(1, num_steps + 1):
            for j in range(i + 1):
                tree[j, i] = self.S0 * (u ** (i - j)) * (d ** j)
                if j == 0:
                    probabilities[j, i] = probabilities[j, i - 1] * p
                elif j == i:
                    probabilities[j, i] = probabilities[j - 1, i - 1] * (1 - p)
                else:
                    probabilities[j, i] = probabilities[j - 1, i - 1] * (1 - p) + probabilities[j, i - 1] * p
                
                # Calculate Arrow-Debreu probabilities
                arrow_debreu_probs[j, i] = probabilities[j, i] * np.exp(-self.rf * i * dt)

        return tree, probabilities, arrow_debreu_probs, p
    

if __name__ == '__main__':

    # Example usage
    model = DemanKaniModel(S0=100, K=100, rf=0.05, T=1, vol=0.2)
    num_steps = 50
    call_price = model.price_option(num_steps, option_type="call")
    put_price = model.price_option(num_steps, option_type="put")

    print(f"Call Option Price: {call_price}")
    print(f"Put Option Price: {put_price}")


    # Example usage
    model = DemanKaniModel(S0=100, K=100, rf=0.02, T=1, vol=0.4)
    num_steps = 4  # Use a small number of steps for easier visualization

    # Build the tree
    tree, p = model.build_tree(num_steps)

    # Print the tree
    print("Recombining Binomial Tree:")
    for i in range(num_steps + 1):
        print(f"Step {i}: {tree[:i + 1, i]}")

    # Print the risk-neutral probability
    print(f"\nRisk-neutral probability (p): {p}")


    # Build the tree with probabilities
    tree, probabilities, arrow_debreu_probs, p = model.build_tree_with_probabilities(num_steps)

    # Print the asset price tree
    print("Recombining Binomial Tree (Asset Prices):")
    for i in range(num_steps + 1):
        print(f"Step {i}: {tree[:i + 1, i]}")

    # Print the probability tree
    print("\nProbability Tree:")
    for i in range(num_steps + 1):
        print(f"Step {i}: {probabilities[:i + 1, i]}")

    # Print the Arrow-Debreu probability tree
    print("\nArrow-Debreu Probability Tree:")
    for i in range(num_steps + 1):
        print(f"Step {i}: {arrow_debreu_probs[:i + 1, i]}")

    # Print the risk-neutral probability
    print(f"\nRisk-neutral probability (p): {p}")