import numpy as np
import pandas as pd
import utile.LinearRegression as l
import utile.Scaler as Scaler
import matplotlib.pyplot as plt

X = np.array([1936, 1946, 1954, 1962, 1968, 1975, 1982, 1990, 1999, 2005]).reshape(-1,1)
Y = np.array([41183000, 39848000, 42781000, 46459000, 49655000, 52599000, 54296000, 56652000, 58521000, 60825000]).reshape(-1,1)

df = pd.DataFrame(X, columns=["annee"])
df["annee2"] = df['annee'] ** 2
scaler = Scaler.Scaler()
scaler.fit_transform(df)

X_norm = df.to_numpy()

# Mon modèle
model_polynomial = l.LinearRegression()
model_polynomial.fit(X_norm, Y)
Y_pred_polynomial = model_polynomial.predict(X_norm)

# Modèle scikit-learn
model_simple = l.LinearRegression()
model_simple.fit(X, Y)
Y_pred_simple = model_simple.predict(X)

# Indicateurs de performance
mse_polynomial = model_polynomial.MSE(Y, Y_pred_polynomial)
r2_polynomial = model_polynomial.get_coef_determination()

mse_simple = model_simple.MSE(Y, Y_pred_simple)
r2_simple = model_simple.get_coef_determination()

print("polynomial :" + str(mse_polynomial) + "    simple : " + str(mse_simple))
print("polynomial :" + str(r2_polynomial) + "    simple : " + str(r2_simple))

annees_futures = np.array([1999, 2005, 2010, 2015, 2020]).reshape(-1,1)
df_annees_futures = pd.DataFrame({ "annee" : [1999, 2005, 2010, 2015, 2020]})
df_annees_futures["annee2"] = df_annees_futures["annee"] ** 2

annees_futures_norm = df_annees_futures.copy()
scaler.transform(annees_futures_norm)

predictions_polynomial = model_polynomial.predict(annees_futures_norm.to_numpy())

predictions_simple = model_simple.predict(annees_futures)



# Visualisation
plt.figure(figsize=(12, 6))
plt.scatter(X, Y, color='blue', label='Données réelles')
plt.plot(X, Y_pred_polynomial, color='red', label='Régression polynomial')
plt.plot(X, Y_pred_simple, color='green', linestyle='--', label='Régression simple')
plt.scatter(df_annees_futures["annee"], predictions_polynomial, color='red', marker='x', label='Projections polynomial')
plt.scatter(df_annees_futures["annee"], predictions_simple, color='green', marker='x', label='Projections simple')
plt.title('Régression Linéaire : Population')
plt.xlabel('Année')
plt.ylabel('Population')
plt.legend()
plt.savefig('rendu_3/img/regression_lineaire_population.png')

plt.show()
