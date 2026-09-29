#Fraud Detection

import numpy as np
import matplotlib.pyplot as plt

n_transactions = 50
p_flag = 0.04
n_hours = 20000
capacity = 4

# Simulate flagged transactions for each hour
flagged_transactions = np.random.binomial(
    n=n_transactions,
    p=p_flag,
    size=n_hours
)

print("Fraud Detection Simulation")
print("Number of simulated hours:", n_hours)
print("Transactions per hour:", n_transactions)
print("Flagging probability:", p_flag)

print("\nFirst 20 simulated hours:")
print(flagged_transactions[:20])

# Empirical statistics
empirical_mean = np.mean(flagged_transactions)
empirical_variance = np.var(flagged_transactions)

print("\nEmpirical Statistics")
print("Mean:", empirical_mean)
print("Variance:", empirical_variance)
print("Minimum:", np.min(flagged_transactions))
print("Maximum:", np.max(flagged_transactions))

# Theoretical statistics
theoretical_mean = n_transactions * p_flag
theoretical_variance = (
    n_transactions
    * p_flag
    * (1 - p_flag)
)

print("\nTheoretical Statistics")
print("Mean:", theoretical_mean)
print("Variance:", theoretical_variance)

# Investigation capacity
hours_over_capacity = np.sum(
    flagged_transactions > capacity
)

hours_within_capacity = np.sum(
    flagged_transactions <= capacity
)

probability_over_capacity = (
    hours_over_capacity / n_hours
)

probability_within_capacity = (
    hours_within_capacity / n_hours
)

print("\nInvestigation Capacity")
print("Capacity:", capacity)
print("Hours over capacity:", hours_over_capacity)
print("Hours within capacity:", hours_within_capacity)
print("Empirical P(X > 4):", probability_over_capacity)
print("Empirical P(X <= 4):", probability_within_capacity)

# Empirical probability distribution
values, counts = np.unique(
    flagged_transactions,
    return_counts=True
)

probabilities = counts / n_hours

plt.figure(figsize=(9, 5))

plt.bar(
    values,
    probabilities,
    color="purple",
    alpha=0.75
)

plt.axvline(
    capacity + 0.5,
    color="red",
    linestyle="--",
    label="Capacity = 4"
)

plt.xlabel("Flagged Transactions per Hour")
plt.ylabel("Empirical Probability")
plt.title("Distribution of Fraud Flags per Hour")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.show()
