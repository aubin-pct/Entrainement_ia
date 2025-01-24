import numpy as np
import pandas as pd
import utile.LinearRegression as l
import utile.Scaler as Scaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

X = np.array([1936, 1946, 1954, 1962, 1968, 1975, 1982, 1990, 1999, 2005]).reshape(-1,1)
Y = np.array([41183000, 39848000, 42781000, 46459000, 49655000, 52599000, 54296000, 56652000, 58521000, 60825000]).reshape(-1,1)

df = pd.DataFrame(X, columns=["annee"])
df["annee2"] = df['annee'] ** 2
scaler = Scaler.Scaler()
scaler.fit_transform(df)

X_norm = df.to_numpy()

# Mon modèle
model = l.LinearRegression()
model.fit(X_norm, Y)
Y_pred_moi = model.predict(X_norm)

# Modèle scikit-learn
model_sklearn = LinearRegression()
model_sklearn.fit(X_norm, Y)
Y_pred_sklearn = model_sklearn.predict(X_norm)

# Indicateurs de performance
mse_moi = model.MSE(Y, Y_pred_moi)
r2_moi = model.get_coef_determination()

mse_sklearn = mean_squared_error(Y, Y_pred_sklearn)
r2_sklearn = r2_score(Y, Y_pred_sklearn)

print(str(mse_moi) + "    sklearn : " + str(mse_sklearn))
print(str(r2_moi) + "    sklearn : " + str(r2_sklearn))


annees_futures = pd.DataFrame({ "annee" : [1999, 2005, 2010, 2015, 2020]})
annees_futures["annee2"] = annees_futures["annee"] ** 2

annees_futures_norm = annees_futures.copy()
scaler.transform(annees_futures_norm)

predictions_moi = model.predict(annees_futures_norm.to_numpy())

predictions_sklearn = model_sklearn.predict(annees_futures_norm.to_numpy())



# Visualisation
plt.figure(figsize=(12, 6))
plt.scatter(X, Y, color='blue', label='Données réelles')
plt.plot(X, Y_pred_moi, color='red', label='Régression manuelle')
plt.plot(X, Y_pred_sklearn, color='green', linestyle='--', label='Régression scikit-learn')
plt.scatter(annees_futures["annee"], predictions_moi, color='red', marker='x', label='Projections manuelles')
plt.scatter(annees_futures["annee"], predictions_sklearn, color='green', marker='x', label='Projections scikit-learn')
plt.title('Régression Linéaire : Population')
plt.xlabel('Année')
plt.ylabel('Population')
plt.legend()
plt.savefig('rendu_3/img/regression_lineaire_population.png')

plt.show()
