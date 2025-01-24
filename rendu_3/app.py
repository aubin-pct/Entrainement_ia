import numpy as np
import pandas as pd
import utile.LinearRegression as l
import utile.PolynomialRegression as p
import utile.Scaler as Scaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



df_csv = pd.read_csv("rendu_3/csv_files/Ice_cream_selling_data.csv")

temperatures = df_csv["Temperature (°C)"].to_numpy().reshape(-1,1)
ventes = df_csv["Ice Cream Sales (units)"].to_numpy().reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(temperatures, ventes, test_size=0.2, random_state=42)

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


df = pd.DataFrame(x_test, columns=["temperature"])
for i in range(2, model_polynomial.degree + 1):
        df["temperature" + str(i)] = df['temperature'] ** i
scaler.fit(df)

# Normalisation annees_futures
futures_norm = df.copy()
scaler.transform(futures_norm)

# Prédictions
predictions_polynomial = model_polynomial.predict(futures_norm.to_numpy())
predictions_simple = model_simple.predict(x_test)


# Visualisation
plt.figure(figsize=(12, 6))
plt.scatter(x_train, y_train, color='blue', label='Données réelles')
plt.plot(x_train, Y_pred_polynomial, color='red', label='Régression polynomial')
plt.plot(x_train, Y_pred_simple, color='green', linestyle='--', label='Régression simple')
plt.scatter(df["temperature"], predictions_polynomial, color='red', marker='x', label='Projections polynomial')
plt.scatter(df["temperature"], predictions_simple, color='green', marker='x', label='Projections simple')
plt.scatter(df["temperature"], y_test, color='blue', marker='x', label='valeurs réelles')
plt.title('Régression Linéaire : Ventes')
plt.xlabel('Température')
plt.ylabel('Ventes')
plt.legend()
plt.savefig('rendu_3/img/regression_polynomial_Vente_Glaces.png')

plt.show()
