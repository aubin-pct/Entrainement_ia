import numpy as np
import utile.LinearRegression as l
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

X = np.array([1936, 1946, 1954, 1962, 1968, 1975, 1982, 1990, 1999, 2005]).reshape(-1,1)
Y = np.array([41183000, 39848000, 42781000, 46459000, 49655000, 52599000, 54296000, 56652000, 58521000, 60825000]).reshape(-1,1)


# Mon modèle
model = l.LinearRegression()
model.fit(X, Y)
Y_pred_moi = model.predict(X)

# Modèle scikit-learn
model_sklearn = LinearRegression()
model_sklearn.fit(X, Y)
Y_pred_sklearn = model_sklearn.predict(X)

# Indicateurs de performance
mse_moi = model.MSE(Y, Y_pred_moi)
r2_moi = model.get_coef_determination()

mse_sklearn = mean_squared_error(Y, Y_pred_sklearn)
r2_sklearn = r2_score(Y, Y_pred_sklearn)

print(str(mse_moi) + "    sklearn : " + str(mse_sklearn))
print(str(r2_moi) + "    sklearn : " + str(r2_sklearn))

annees_futures = np.array([1999, 2005, 2010, 2015, 2020]).reshape(-1,1)
predictions_moi = model.predict(annees_futures)
predictions_sklearn = model_sklearn.predict(annees_futures)

# Visualisation
plt.figure(figsize=(12, 6))
plt.scatter(X, Y, color='blue', label='Données réelles')
plt.plot(X, Y_pred_moi, color='red', label='Régression manuelle')
plt.plot(X, Y_pred_sklearn, color='green', linestyle='--', label='Régression scikit-learn')
plt.scatter(annees_futures, predictions_moi, color='red', marker='x', label='Projections manuelles')
plt.scatter(annees_futures, predictions_sklearn, color='green', marker='x', label='Projections scikit-learn')
plt.title('Régression Linéaire : Population')
plt.xlabel('Année')
plt.ylabel('Population')
plt.legend()
plt.show()
