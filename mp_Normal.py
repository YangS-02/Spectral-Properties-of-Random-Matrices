import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


tol = 2 * 9e-15
v2 = 2
c = 0.5
a = v2 * (1 - np.sqrt(c)) ** 2
b = v2 * (1 + np.sqrt(c)) ** 2
p = 1000
n = int(p / c)


X = np.random.normal(0.0, np.sqrt(v2), (p, n))
if c <= 1:
    lams = np.linalg.eigvalsh(X @ X.T / n)
else:
    lams = np.linalg.eigvalsh(X.T @ X / n)


def mc(x):
    mp = np.where((x >= a) & (x <= b) & (x != 0), np.sqrt((b - x) * (x - a)) / (2 * v2 * np.pi * c * x), 0)
    mp = np.where((x == 0) & (c > 1), mp + ((c - 1) / c), mp)
    return mp


pp = len(lams)
lams = np.append(np.zeros(p - pp), lams)


I = integrate.quad(mc, a, b, epsabs=tol)
mass = I[0] + np.maximum(1.0 - 1.0 / c, 0)
print(f"mass = {mass}")
if mass < 1.0 - p * tol:
    print("Warning! -- mass missing")


x = np.arange(0, b, 0.001)
plt.hist(lams, bins=66, density=True)
if c <= 1:
    lmax = max(mc(x))
else:
    range_x = np.arange(a, b, 0.1)
    lmax = max(mc(range_x))
plt.ylim(0, 1.25 * lmax)
plt.plot(x, mc(x), '-', linewidth=2)
plt.xlabel("Eigenvalues")
plt.ylabel("Density")
plt.title(f"MP-Gaussian (Var = {v2}, q = {c}, p = {p}, n = {n})", fontsize=12)
if c > 1:
    plt.text(a/3, 1.15 * lmax, f"The pole at zero has density {(c - 1)/c:.2f}", fontsize=8)
plt.show()
ftype = "pdf"
plt.savefig("img/mp." + ftype,
            transparent=True, format=ftype)
