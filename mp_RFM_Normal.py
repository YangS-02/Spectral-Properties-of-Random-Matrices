import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


tol = 2 * 9e-15
c = 0.6
p = 1000
n = int(p / c)

sd_beta = 0.3
sd_x = 16
sd_X = 50
v2 = (sd_x ** 2) * (sd_beta ** 2) + sd_X ** 2

# Y = (beta * x^T) + X
# beta: a p-vector draw from N(0, s^2), s = 0.2
beta = np.random.normal(0.0, sd_beta, size=(p, 1))
# x:    a n-vector draw from N(0, sigma^2), sigma = 16
x = np.random.normal(0.0, sd_x, size=(n, 1))
# X:    a p * n matrix draw from N(0, d^2), d = 50
X = np.random.normal(0.0, sd_X, size=(p, n))

Y = np.dot(beta, np.transpose(x)) + X
print("Theoretical:", v2)
print("Sample Variance:", np.var(Y))
print("Sample Mean:", np.mean(Y))
a = v2 * (1 - np.sqrt(c)) ** 2
b = v2 * (1 + np.sqrt(c)) ** 2
print(f'a = {a}, b = {b}')
if c <= 1:
    lams = np.linalg.eigvalsh(Y @ Y.T / n)
else:
    lams = np.linalg.eigvalsh(Y.T @ Y / n)
print(lams)


def mc(x):
    x = np.asarray(x)
    mp = np.where((x >= a) & (x <= b) & (x != 0), np.sqrt((b - x) * (x - a)) / (2 * np.var(X) * np.pi * c * x), 0)
    mp = np.where((x == 0) & (c > 1), mp + ((c - 1) / c), mp)
    return mp


pp = len(lams)
lams = np.append(np.zeros(p - pp), lams)


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
    range_x = np.arange(a, b, 0.1)
    lmax = max(mc(range_x))
plt.ylim(0, 1.25 * lmax)
plt.plot(x, mc(x), '-', linewidth=2)
plt.xlabel("Eigenvalues")
plt.ylabel("Density")
plt.title(f"MP-Classical-Model-Normal (Var = {v2}, q = {c}, p = {p}, n = {n})", fontsize=12)
if c > 1:
    plt.text(a/3, 1.15 * lmax, f"The pole at zero has density {(c - 1)/c:.2f}", fontsize=8)
plt.show()
ftype = "pdf"
plt.savefig("img/mp." + ftype,
            transparent=True, format=ftype)

I = integrate.quad(mc, a, b, epsabs=tol)
mass = I[0] + np.maximum(1.0 - 1.0 / c, 0)
print(f"mass = {mass}")
if mass < 1.0 - p * tol:
    print("Warning! -- mass missing")
