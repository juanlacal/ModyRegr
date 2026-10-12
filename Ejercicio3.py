"""
Entrega 1 - Problema 3: Alturas
Modelo: altura = b0 + b1 * (promedio padres, centrado) + b2 * Hombre + error
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

df = pd.read_csv("Alturas_1.csv")
df.columns = ["id", "altura", "sexo", "padre", "madre"]
print("Filas originales:", len(df))
print("Filas con datos faltantes:\n", df[df.isna().any(axis=1)])


d = df.dropna().copy()
print("Filas usadas:", len(d))
print(d["sexo"].value_counts())

d["hombre"] = (d["sexo"] == "Masculino").astype(int)
d["medio"] = (d["padre"] + d["madre"]) / 2
media_medio = d["medio"].mean()
d["medio_c"] = d["medio"] - media_medio
print(f"Promedio de 'altura media de los padres': {media_medio:.2f} cm")

print("\nMedias por sexo:\n", d.groupby("sexo")[["altura", "padre", "madre"]].mean().round(1))

formulas = {
    "padre + madre":            "altura ~ padre + madre",
    "sexo":                     "altura ~ hombre",
    "padre + madre + sexo":     "altura ~ padre + madre + hombre",
    "promedio padres + sexo":   "altura ~ medio_c + hombre",
}
print("\nModelo                     R2     R2 aj.  sigma")
for nombre, f in formulas.items():
    m = smf.ols(f, d).fit()
    print(f"{nombre:25s} {m.rsquared:.3f}  {m.rsquared_adj:.3f}   {np.sqrt(m.scale):.2f}")

m_sep = smf.ols("altura ~ padre + madre + hombre", d).fit()
print("\nTest H0: coef. padre = coef. madre")
print(m_sep.t_test("padre = madre"))

modelo = smf.ols("altura ~ medio_c + hombre", d).fit()
print(modelo.summary())

b0, b1, b2 = modelo.params
print(f"\naltura = {b0:.2f} + {b1:.3f}*(promedio padres - {media_medio:.1f}) + {b2:.2f}*hombre")
print(f"sigma = {np.sqrt(modelo.scale):.2f} cm")

m_sin39 = smf.ols("altura ~ medio_c + hombre", d[d["id"] != 39]).fit()
print("\nSin la observación 39:", m_sin39.params.round(3).to_dict(), f"R2 = {m_sin39.rsquared:.3f}")


plt.figure(figsize=(6, 4.5))
grid = np.linspace(d["medio"].min() - 1, d["medio"].max() + 1, 50)
for sexo, h, marker in [("Femenino", 0, "o"), ("Masculino", 1, "s")]:
    sub = d[d["hombre"] == h]
    plt.scatter(sub["medio"], sub["altura"], marker=marker, alpha=0.7, label=sexo)
    plt.plot(grid, b0 + b1 * (grid - media_medio) + b2 * h, lw=2)
plt.plot(grid, grid, "k:", lw=1, label="altura = promedio padres")
plt.xlabel("Promedio de alturas de los padres (cm)")
plt.ylabel("Altura del estudiante (cm)")
plt.title("Modelo ajustado por sexo")
plt.legend()
plt.tight_layout()
plt.savefig("p3_modelo.png", dpi=200, bbox_inches="tight")
plt.show()