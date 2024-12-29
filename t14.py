
import numpy as np
import matplotlib.pyplot as plt

# Datele de intrare ale proiectului 2 - 14
mu = 0.25
sigma = 0.6

print("Mean: ", mu)
print("Var:  ", sigma)
print("------------------\n")

sigma = np.sqrt(0.6)
sigma_stripes_x = [
    mu - 3 * sigma,
    mu - 2 * sigma,
    mu - 1 * sigma,
    mu,
    mu + 1 * sigma,
    mu + 2 * sigma,
    mu + 3 * sigma,
]

# Metoda polară
def generate_normal_polar(n):
    samples = []
    while len(samples) < n:
        u1 = np.random.uniform(0, 1)
        u2 = np.random.uniform(0, 1)

        v1 = 2 * u1 - 1
        v2 = 2 * u2 - 1

        s = v1 ** 2 + v2 ** 2
        if s < 1:
            z1 = v1 * np.sqrt((-2 * np.log(s))/s)
            z2 = v2 * np.sqrt((-2 * np.log(s))/s)

            n1 = mu + sigma * z1
            n2 = mu + sigma * z2

            samples.append(n1)
            samples.append(n2)
    return np.array(samples)

# Metoda de compunere-respingere
def generate_normal_rejection(n):
    samples = []
    while len(samples) < n:
        # Generate uniformly distributed values
        u = np.random.uniform(0, 1)
        y = np.random.exponential(1)

        cond = np.exp(-0.5 * y ** 2 + y - 0.5)
        if u <= cond:
            u2 = np.random.uniform(0, 1)
            if u2 <= 0.5:
                s = 1
            else:
                s = -1

            ############################ x
            samples.append(mu + sigma * (s * y))
    return np.array(samples)

# Generăm <n_samples> valori
n_samples = 10_000
samples_polar = generate_normal_polar(n_samples)
samples_rejection = generate_normal_rejection(n_samples)

print(samples_polar)
print("Polar Mean: ", np.mean(samples_polar))
print("Polar Var:  ", np.var(samples_polar))
print("Polar min:  ", np.min(samples_polar))
print("Polar max:  ", np.max(samples_polar))
print("------------------\n")

print(samples_rejection)
print("Rej Mean: ", np.mean(samples_rejection))
print("Rej Var:  ", np.var(samples_rejection))
print("Rej min:  ", np.min(samples_rejection))
print("Rej max:  ", np.max(samples_rejection))
print("------------------\n")

rng = np.random.default_rng()
in_python_alg = rng.normal(mu, sigma, n_samples)

print(in_python_alg)
print("Numpy Mean: ", np.mean(in_python_alg))
print("Numpy Var:  ", np.var(in_python_alg))
print("Numpy min:  ", np.min(in_python_alg))
print("Numpy max:  ", np.max(in_python_alg))

# Plotting histograms
plt.figure(figsize=(21, 6))

plt.subplot(1, 3, 1)
plt.hist(samples_polar, bins=50, color='blue', alpha=0.7)
plt.title('Histogram (Polar Method)')
plt.xlabel('Value')
plt.ylabel('Frequency')
for x in sigma_stripes_x:
    plt.axvline(x=x, color='red', linestyle='--', linewidth=1)


plt.subplot(1, 3, 2)
plt.hist(samples_rejection, bins=50, color='green', alpha=0.7)
plt.title('Histogram (Rejection Method)')
plt.xlabel('Value')
plt.ylabel('Frequency')
for x in sigma_stripes_x:
    plt.axvline(x=x, color='blue', linestyle='-.', linewidth=1)

plt.subplot(1, 3, 3)
plt.hist(in_python_alg, bins=50, color='red', alpha=0.7)
plt.title('Numpy alg')
plt.xlabel('Value')
plt.ylabel('Frequency')
for x in sigma_stripes_x:
    plt.axvline(x=x, color='green', linestyle=':', linewidth=1)

plt.tight_layout()
plt.show()
