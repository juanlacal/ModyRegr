import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline


url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
df.columns = df.columns.str.upper()

print(df.shape)
print("Faltantes:", df.isna().sum().sum())
print(df.describe().T)
print("Casas con MEDV = 50 (censuradas):", (df["MEDV"] == 50).sum())

# Distribución de la variable objetivo
plt.figure()
plt.hist(df["MEDV"], bins=30)
plt.xlabel("MEDV (miles de USD)"); plt.title("Distribución de MEDV")
plt.savefig("p2_hist_medv.png", dpi=200, bbox_inches="tight")

# Correlación de cada variable con MEDV
corr = df.corr()["MEDV"].drop("MEDV").sort_values()
print("\nCorrelaciones con MEDV:\n", corr)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(df["RM"], df["MEDV"], s=10, alpha=0.6)
ax[0].set_xlabel("RM (habitaciones promedio)"); ax[0].set_ylabel("MEDV")
ax[1].scatter(df["LSTAT"], df["MEDV"], s=10, alpha=0.6)
ax[1].set_xlabel("LSTAT (% población de bajo nivel socioeconómico)")
fig.tight_layout()
plt.savefig("p2_scatter_rm_lstat.png", dpi=200, bbox_inches="tight")


X = df.drop(columns="MEDV")
y = df["MEDV"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nEntrenamiento:", X_train.shape, " Validación:", X_val.shape)

# ---------------------------------------------------------------
# Parte 3: modelo lineal con todas las variables
# ---------------------------------------------------------------
lin = LinearRegression()
lin.fit(X_train, y_train)               # solo con entrenamiento

pred_train = lin.predict(X_train)
pred_val = lin.predict(X_val)

print(f"\nLineal  MSE train: {mean_squared_error(y_train, pred_train):.2f}   "
      f"MSE val: {mean_squared_error(y_val, pred_val):.2f}")
print(f"        R² train:  {r2_score(y_train, pred_train):.3f}   "
      f"R² val:  {r2_score(y_val, pred_val):.3f}")

coefs = pd.Series(lin.coef_, index=X.columns).sort_values()
print("\nCoeficientes del modelo lineal:\n", coefs)

plt.figure()
plt.scatter(y_val, pred_val, s=15, alpha=0.7)
plt.plot([5, 50], [5, 50], "r--", label="predicción perfecta")
plt.xlabel("MEDV real"); plt.ylabel("MEDV predicho")
plt.title("Modelo lineal: validación"); plt.legend()
plt.savefig("p2_lineal_pred.png", dpi=200, bbox_inches="tight")


grados = [1, 2, 3]
mse_tr, mse_va = [], []

print()
for g in grados:
    modelo = make_pipeline(
        PolynomialFeatures(degree=g, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )
    modelo.fit(X_train, y_train)
    mse_tr.append(mean_squared_error(y_train, modelo.predict(X_train)))
    mse_va.append(mean_squared_error(y_val, modelo.predict(X_val)))
    n_var = modelo.named_steps["polynomialfeatures"].n_output_features_
    print(f"Grado {g} ({n_var} variables): MSE train = {mse_tr[-1]:.2f}   MSE val = {mse_va[-1]:.2f}")

plt.figure()
plt.plot(grados, mse_tr, "o-", label="entrenamiento")
plt.plot(grados, mse_va, "s-", label="validación")
plt.yscale("log")
plt.xticks(grados)
plt.xlabel("Grado del polinomio"); plt.ylabel("MSE (escala log)")
plt.title("MSE en función del grado"); plt.legend()
plt.savefig("p2_mse_grado.png", dpi=200, bbox_inches="tight")


lambdas = np.logspace(-3, 4, 30)          # de 0.001 a 10000
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mejores = {}      # λ óptimo por grado
cv_min = {}       # MSE de CV en el λ óptimo

print()
plt.figure()
for g in [2, 3]:
    cv_mse = []
    for lam in lambdas:
        m = make_pipeline(
            PolynomialFeatures(degree=g, include_bias=False),
            StandardScaler(),
            Ridge(alpha=lam)               # en sklearn λ se llama alpha
        )
        s = cross_val_score(m, X_train, y_train, cv=kf, scoring="neg_mean_squared_error")
        cv_mse.append(-s.mean())

    mejores[g] = lambdas[np.argmin(cv_mse)]
    cv_min[g] = min(cv_mse)

    final = make_pipeline(
        PolynomialFeatures(degree=g, include_bias=False),
        StandardScaler(),
        Ridge(alpha=mejores[g])
    )
    final.fit(X_train, y_train)
    print(f"Ridge grado {g}: λ óptimo = {mejores[g]:.3f}   MSE CV = {cv_min[g]:.2f}   "
          f"MSE train = {mean_squared_error(y_train, final.predict(X_train)):.2f}   "
          f"MSE val = {mean_squared_error(y_val, final.predict(X_val)):.2f}")
    plt.semilogx(lambdas, cv_mse, "o-", ms=3, label=f"grado {g}")

plt.xlabel("λ (escala log)"); plt.ylabel("MSE de validación cruzada")
plt.title("Ridge: MSE en función de λ"); plt.legend()
plt.savefig("p2_ridge_lambda.png", dpi=200, bbox_inches="tight")

g_best = min(cv_min, key=cv_min.get)
print(f"\nGrado elegido por CV: {g_best}")

sin_reg = make_pipeline(PolynomialFeatures(g_best, include_bias=False),
                        StandardScaler(), LinearRegression()).fit(X_train, y_train)
con_reg = make_pipeline(PolynomialFeatures(g_best, include_bias=False),
                        StandardScaler(), Ridge(alpha=mejores[g_best])).fit(X_train, y_train)

fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for a, m, titulo in [(ax[0], sin_reg, "Sin regularización"),
                     (ax[1], con_reg, f"Ridge (λ = {mejores[g_best]:.2f})")]:
    pred = m.predict(X_val)
    a.scatter(y_val, pred, s=15, alpha=0.7)
    a.plot([5, 50], [5, 50], "r--")
    a.set_title(f"{titulo}\nMSE val = {mean_squared_error(y_val, pred):.2f}")
    a.set_xlabel("MEDV real")
ax[0].set_ylabel("MEDV predicho")
fig.suptitle(f"Polinomio de grado {g_best}: validación")
fig.tight_layout()
plt.savefig("p2_mejor_modelo.png", dpi=200, bbox_inches="tight")

plt.show()