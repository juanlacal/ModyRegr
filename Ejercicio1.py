import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(15)   # semilla fija para que sea reproducible
THETA = -1/3
A, B = -0.999, 0.999              # intervalo de búsqueda (evita log(0) en ±1)


def simular(n, theta=THETA):
    u = rng.uniform(0, 1, n)
    if theta == 0:
        return 2*u - 1
    return (-1 + np.sqrt((1 - theta)**2 + 4*theta*u)) / theta


x_big = simular(100_000)
print(f"Media simulada: {x_big.mean():.4f}   teórica θ/3: {THETA/3:.4f}")

grid_x = np.linspace(-1, 1, 200)
plt.figure()
plt.hist(x_big, bins=50, density=True, alpha=0.6, label="simulado")
plt.plot(grid_x, 0.5*(1 + THETA*grid_x), "r", label="densidad teórica")
plt.xlabel("x"); plt.ylabel("densidad"); plt.title("Parte 1: muestra simulada (θ = -1/3)")
plt.legend()


def log_ver(theta, x):
    return -len(x)*np.log(2) + np.sum(np.log(1 + theta*x))


def golden(f, a, b, tol=1e-6):
    """Minimiza f en [a, b]; requiere f unimodal."""
    phi = (np.sqrt(5) - 1) / 2
    c, d = b - phi*(b - a), a + phi*(b - a)
    while b - a > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - phi*(b - a), a + phi*(b - a)
    return (a + b) / 2


def estimar(x):
    return golden(lambda t: -log_ver(t, x), A, B)


x10 = simular(10)
theta_hat = estimar(x10)
print(f"\nMuestra n=10: {np.round(np.sort(x10), 3)}")
print(f"θ̂ (golden) = {theta_hat:.4f}")

M = 50
tamanios = [10, 100, 1000]

resultados = {}
for n in tamanios:
    est = np.array([estimar(simular(n)) for _ in range(M)])
    sesgo = est.mean() - THETA
    var = est.var()
    mse = np.mean((est - THETA)**2)
    bordes = np.mean((est < A + 1e-3) | (est > B - 1e-3))
    resultados[n] = est
    print(f"\nn = {n}")
    print(f"  media θ̂ = {est.mean():.4f}")
    print(f"  sesgo    = {sesgo:.5f}")
    print(f"  varianza = {var:.5f}")
    print(f"  MSE      = {mse:.5f}   (sesgo² + var = {sesgo**2 + var:.5f})")
    print(f"  % de estimaciones en el borde ±1: {100*bordes:.0f}%")

# Distribución de θ̂ según n
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
for a, n in zip(ax, tamanios):
    a.hist(resultados[n], bins=15, alpha=0.7)
    a.axvline(THETA, color="g", ls=":", label="θ real")
    a.axvline(resultados[n].mean(), color="r", ls="--", label="media θ̂")
    a.set_title(f"n = {n}"); a.set_xlabel("θ̂"); a.legend()
fig.suptitle(f"Partes 3-4: distribución de θ̂ en {M} muestras")
fig.tight_layout()

# MSE, sesgo² y varianza en función de n (escala log)
sesgos2 = [(resultados[n].mean() - THETA)**2 for n in tamanios]
vars_ = [resultados[n].var() for n in tamanios]
mses = [np.mean((resultados[n] - THETA)**2) for n in tamanios]
plt.figure()
plt.loglog(tamanios, mses, "o-", label="MSE")
plt.loglog(tamanios, vars_, "s--", label="varianza")
plt.loglog(tamanios, sesgos2, "^:", label="sesgo²")
plt.xlabel("n"); plt.title("Evolución de MSE, varianza y sesgo²")
plt.legend()

plt.show()