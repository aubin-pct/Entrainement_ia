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
# Recherche du meilleur degré de polynôme
best_degree = 1
best_model_polynomial = l.LinearRegression()
best_model_polynomial.fit(X_norm, Y)
best_Y_pred_polynomial = best_model_polynomial.predict(X_norm)
best_mse_polynomial = best_model_polynomial.MSE(Y, best_Y_pred_polynomial)
best_r2_polynomial = best_model_polynomial.get_coef_determination()

for degree in range(2,6):
    df = pd.DataFrame(X, columns=["annee"])
    df["annee2"] = df['annee'] ** degree
    scaler.fit_transform(df)

    X_norm = df.to_numpy()

    model_polynomial = l.LinearRegression()
    model_polynomial.fit(X_norm, Y)
    Y_pred_polynomial = model_polynomial.predict(X_norm)
    mse_polynomial = model_polynomial.MSE(Y, Y_pred_polynomial)
    r2_polynomial = model_polynomial.get_coef_determination()
    print(str(degree) + "  " + str(round(r2_polynomial, 5)) + "   " + str(mse_polynomial))

    if (mse_polynomial < best_mse_polynomial):
        best_degree = degree
        best_model_polynomial = model_polynomial
        best_mse_polynomial = mse_polynomial
        best_r2_polynomial = r2_polynomial
        best_Y_pred_polynomial = Y_pred_polynomial

# Modèle scikit-learn
model_simple = l.LinearRegression()
model_simple.fit(X, Y)
Y_pred_simple = model_simple.predict(X)

# Indicateurs de performance
mse_simple = model_simple.MSE(Y, Y_pred_simple)
r2_simple = model_simple.get_coef_determination()

# Affichage indicateurs
print("MSE -> polynomial : " + str(best_mse_polynomial) + "    simple : " + str(mse_simple))
print("r2 -> polynomial : " + str(best_r2_polynomial) + "    simple : " + str(r2_simple))

# annees_futures mise en forme
annees_futures = np.array([1999, 2005, 2010, 2015, 2020]).reshape(-1,1)
df_annees_futures = pd.DataFrame({ "annee" : [1999, 2005, 2010, 2015, 2020]})
df_annees_futures["annee2"] = df_annees_futures["annee"] ** best_degree

# Fit scaler pour best_modele
df = pd.DataFrame(X, columns=["annee"])
df["annee2"] = df['annee'] ** best_degree
scaler.fit(df)

# Normalisation annees_futures
annees_futures_norm = df_annees_futures.copy()
scaler.transform(annees_futures_norm)

# Prédictions
predictions_polynomial = best_model_polynomial.predict(annees_futures_norm.to_numpy())
predictions_simple = model_simple.predict(annees_futures)


# Visualisation
plt.figure(figsize=(12, 6))
plt.scatter(X, Y, color='blue', label='Données réelles')
plt.plot(X, best_Y_pred_polynomial, color='red', label='Régression polynomial')
plt.plot(X, Y_pred_simple, color='green', linestyle='--', label='Régression simple')
plt.scatter(df_annees_futures["annee"], predictions_polynomial, color='red', marker='x', label='Projections polynomial')
plt.scatter(df_annees_futures["annee"], predictions_simple, color='green', marker='x', label='Projections simple')
plt.title('Régression Linéaire : Population')
plt.xlabel('Année')
plt.ylabel('Population')
plt.legend()
plt.savefig('rendu_3/img/regression_lineaire_population.png')

plt.show()
