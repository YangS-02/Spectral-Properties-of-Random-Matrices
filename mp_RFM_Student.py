import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.stats import t


tol = 2 * 9e-15
df = 3.5
c = 0.6
p = 1000
n = int(p / c)

X = t.rvs(df=df, loc=0, scale=np.sqrt(2523.04 * ((df-2)/df)), size=(p, n))
a = 2523.04 * (1 - np.sqrt(c)) ** 2
b = 2523.04 * (1 + np.sqrt(c)) ** 2
if c <= 1:
    lams = np.linalg.eigvalsh(X @ X.T / n)
else:
    lams = np.linalg.eigvalsh(X.T @ X / n)
print(lams)


def mc(x):
    mp = np.where((x >= a) & (x <= b) & (x != 0), np.sqrt((b - x) * (x - a)) / (2 * np.var(X) * np.pi * c * x), 0)
    mp = np.where((x == 0) & (c > 1), mp + ((c - 1) / c), mp)
    return mp


I = integrate.quad(mc, a, b, epsabs=tol)
mass = I[0] + np.maximum(1.0 - 1.0 / c, 0)
print(f"mass = {mass}")
if mass < 1.0 - p * tol:
    print("Warning! -- mass missing")

x = np.arange(0, b, 0.01)
plt.hist(lams, bins=66, density=True)
if c <= 1:
    lmax = max(mc(x))
else:
    range_x = np.arange(a, b, 0.01)
    lmax = max(mc(range_x))
plt.ylim(0, 1.25 * lmax)
plt.plot(x, mc(x), '-', linewidth=2)
plt.ylim(0, 0.0005)
# plt.xlim(0, 2 * b)
plt.xlabel("Eigenvalues")
plt.ylabel("Density")
plt.title(f"MP-Student-scaled (df = {df}, q = {c}, p = {p}, n = {n})", fontsize=12)
if c > 1:
    plt.text(a / 3, 1.15 * lmax, f"The pole at zero has density {(c - 1) / c:.2f}", fontsize=8)
plt.show()
ftype = "pdf"
plt.savefig("img/mp." + ftype,
            transparent=True, format=ftype)