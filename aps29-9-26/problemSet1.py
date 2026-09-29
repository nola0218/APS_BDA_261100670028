#Bernoulli Random Variable

import numpy as np
import matplotlib.pyplot as plt

p = 0.30
sample_size = 20

# Theoretical PMF
theoretical_pmf = {
    0: 1 - p,
    1: p
}

print("Theoretical PMF")
print("P(X=0):", theoretical_pmf[0])
print("P(X=1):", theoretical_pmf[1])

# Generate Bernoulli observations
samples = np.random.binomial(
    n=1,
    p=p,
    size=sample_size
)

print("\nGenerated observations:")
print(samples)
print("Number of observations:", len(samples))

# Count outcomes
number_of_clicks = np.sum(samples == 1)
number_of_non_clicks = np.sum(samples == 0)

print("\nCounts")
print("Number of clicks:", number_of_clicks)
print("Number of non-clicks:", number_of_non_clicks)
print("Total:", number_of_clicks + number_of_non_clicks)

# Empirical PMF
empirical_pmf = {
    0: number_of_non_clicks / sample_size,
    1: number_of_clicks / sample_size
}

print("\nEmpirical PMF")
print("P(X=0):", empirical_pmf[0])
print("P(X=1):", empirical_pmf[1])

# Expectations
theoretical_mean = p
empirical_mean = np.mean(samples)

print("\nExpectation")
print("Theoretical:", theoretical_mean)
print("Empirical:", empirical_mean)

# Variance
theoretical_variance = p * (1 - p)
empirical_variance = np.var(samples)

print("\nVariance")
print("Theoretical:", theoretical_variance)
print("Empirical:", empirical_variance)

# Final results
results = [
    [
        "P(X=0)",
        theoretical_pmf[0],
        empirical_pmf[0],
        abs(theoretical_pmf[0] - empirical_pmf[0])
    ],
    [
        "P(X=1)",
        theoretical_pmf[1],
        empirical_pmf[1],
        abs(theoretical_pmf[1] - empirical_pmf[1])
    ],
    [
        "Expectation",
        theoretical_mean,
        empirical_mean,
        abs(theoretical_mean - empirical_mean)
    ],
    [
        "Variance",
        theoretical_variance,
        empirical_variance,
        abs(theoretical_variance - empirical_variance)
    ]
]

print("\nFinal Result Table")
print(
    f"{'Measure':<15}"
    f"{'Theoretical':<15}"
    f"{'Empirical':<15}"
    f"{'Absolute Difference':<20}"
)

print("-" * 65)

for result in results:
    print(
        f"{result[0]:<15}"
        f"{result[1]:<15.4f}"
        f"{result[2]:<15.4f}"
        f"{result[3]:<20.4f}"
    )

# Compare PMFs
x = np.array([0, 1])
theoretical_values = np.array([
    theoretical_pmf[0],
    theoretical_pmf[1]
])
empirical_values = np.array([
    empirical_pmf[0],
    empirical_pmf[1]
])

width = 0.35

plt.figure(figsize=(7, 5))

plt.bar(
    x - width / 2,
    theoretical_values,
    width,
    label="Theoretical",
    color="steelblue"
)

plt.bar(
    x + width / 2,
    empirical_values,
    width,
    label="Empirical",
    color="orange"
)

plt.xlabel("Outcome")
plt.ylabel("Probability")
plt.title("Theoretical vs Empirical PMF")
plt.xticks([0, 1], ["X=0", "X=1"])
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.show()

# Effect of sample size
sample_sizes = [10, 100, 1000, 10000]

print("\nEffect of Sample Size")
print(
    f"{'Sample Size':<15}"
    f"{'P(X=0)':<15}"
    f"{'P(X=1)':<15}"
    f"{'Mean':<15}"
    f"{'Variance':<15}"
)

print("-" * 75)

for n in sample_sizes:
    samples_n = np.random.binomial(
        n=1,
        p=p,
        size=n
    )

    pmf_0 = np.mean(samples_n == 0)
    pmf_1 = np.mean(samples_n == 1)
    mean = np.mean(samples_n)
    variance = np.var(samples_n)

    print(
        f"{n:<15}"
        f"{pmf_0:<15.4f}"
        f"{pmf_1:<15.4f}"
        f"{mean:<15.4f}"
        f"{variance:<15.4f}"
    )
