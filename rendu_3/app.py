import numpy as np
import pandas as pd
import utile.LinearRegression as l
import utile.PolynomialRegression as p
import utile.Scaler as Scaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



df_csv = pd.read_csv("rendu_3/csv_files/Ice_cream_selling_data.csv")

x_train = df_csv["Temperature (°C)"].to_numpy().reshape(-1,1)
y_train = df_csv["Ice Cream Sales (units)"].to_numpy().reshape(-1,1)

scaler = Scaler.Scaler()


model_polynomial = p.PolynomialRegression()
model_polynomial.fit(x_train, y_train)
Y_pred_polynomial = model_polynomial.predict(x_train)

# Modèle lineaire
model_simple = l.LinearRegression()
model_simple.fit(x_train, y_train)
Y_pred_simple = model_simple.predict(x_train)

# Indicateurs de performance
mse_simple = model_simple.MSE(y_train, Y_pred_simple)
r2_simple = model_simple.get_coef_determination()

# Affichage indicateurs
print("MSE -> polynomial : " + str(model_polynomial.MSE(y_train, Y_pred_polynomial)) + "    simple : " + str(mse_simple))
print("r2 -> polynomial : " + str(model_polynomial.get_coef_determination()) + "    simple : " + str(r2_simple))




sorted_indices = np.argsort(x_train.flatten())  # Indices pour trier x_train
x_train_sorted = x_train[sorted_indices]
Y_pred_polynomial_sorted = Y_pred_polynomial[sorted_indices]
Y_pred_simple_sorted = Y_pred_simple[sorted_indices]


# Visualisation
plt.figure(figsize=(12, 6))
plt.scatter(x_train, y_train, color='blue', label='Données réelles')
plt.plot(x_train_sorted, Y_pred_polynomial_sorted, color='red', label='Régression polynomial')
plt.plot(x_train_sorted, Y_pred_simple_sorted, color='green', linestyle='--', label='Régression simple')
plt.title('Régression Linéaire : Ventes')
plt.xlabel('Température')
plt.ylabel('Ventes')
plt.legend()
plt.savefig('rendu_3/img/regression_polynomial_Vente_Glaces.png')

plt.show()
