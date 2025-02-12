import numpy as np
import pandas as pd
import test
import utile.LinearRegression as l
import utile.PolynomialRegression as p
import utile.Scaler as Scaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold



#df_csv = pd.read_csv("rendu_3/csv_files/Ice_cream_selling_data.csv")

#x_train = df_csv["Temperature (°C)"].to_numpy().reshape(-1,1)
#y_train = df_csv["Ice Cream Sales (units)"].to_numpy().reshape(-1,1)

# Extraction
df_csv = pd.read_csv("rendu_3/csv_files/Position_Salaries.csv")


# Mise en forme, à modifier selon le fichier donné
x_train = df_csv.iloc[:, 1].values.reshape(-1,1)
y_train = df_csv.iloc[:, -1].values.reshape(-1,1)

kf = KFold(n_splits=10, shuffle=True, random_state=42)


mse = np.array([])
mse_mean = np.array([])
r2 = np.array([])

# Graphique des ordre 1 à 6
plt.scatter(x_train, y_train, label="Données réelles", color='black')
for degree in range(1,len(x_train)):
    # Création de la transformation polynomiale
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(x_train)

    # Ajustement du modèle de régression linéaire
    lin_reg = l.LinearRegression()
    lin_reg.fit(X_poly, y_train)
    
    # Prédictions
    y_pred = lin_reg.predict(X_poly)

    # Calcul des métriques
    mse = np.append(mse, mean_squared_error(y_train, y_pred))
    r2 = np.append(r2, r2_score(y_train, y_pred))

    # Génération des points de prédiction pour le graphique
    X_grid = np.arange(np.min(x_train), np.max(x_train)+1.2, 0.2)
    X_grid = X_grid.reshape((len(X_grid), 1))
    X_grid_poly = poly_reg.transform(X_grid)
    y_grid_pred = lin_reg.predict(X_grid_poly)
    
    # Tracer la courbe pour ce degré
    plt.plot(X_grid, y_grid_pred, label=f"Degré {degree}")
    mse_temp = np.array([])
    for train_index, test_index in kf.split(x_train):
        X_Train, X_Test = X_poly[train_index], X_poly[test_index]
        Y_Train, Y_Test = y_train[train_index], y_train[test_index]

        lin_reg.fit(X_Train, Y_Train)  # Entraînement du modèle
        y_pred = lin_reg.predict(X_Test) 
        mse_temp = np.append(mse_temp, lin_reg.MSE(Y_Test, y_pred))
    mse_mean = np.append(mse_mean, np.mean(mse_temp))
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.savefig("rendu_3/img/regression_poly_degres.png", format="png")
plt.show()

# Graphique mean squared error
plt.scatter(range(1,len(x_train)), mse, label="MSE", color='black')
plt.scatter(range(1,len(x_train)), mse_mean, label="MSE Cross Validation test", color='red')
plt.title("Régression Polynomial : Evolution de l'erreur quadratique moyenne")
plt.xlabel("Ordre")
plt.ylabel("Y")
plt.legend()
plt.show()

# Graphique coefficient de détermination
plt.scatter(range(1,len(x_train)), r2, label="R2", color='blue')
plt.title('Régression Polynomial : Evolution du coefficient de détermination')
plt.xlabel("Ordre")
plt.ylabel("Y")
plt.legend()
plt.show()

scaler = Scaler.Scaler()

# Modèle polynomial
model_polynomial = p.PolynomialRegression()
degree = input("Quel degré choisir ?")
model_polynomial.fit(x_train, y_train, degree=int(degree))
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
x_grid_train = np.arange(np.min(x_train_sorted), np.max(x_train_sorted)+1.2, 0.2)
x_grid_train = x_grid_train.reshape((len(x_grid_train), 1))
Y_pred_polynomial_sorted = model_polynomial.predict(x_grid_train)
Y_pred_simple_sorted = Y_pred_simple[sorted_indices]


# Visualisation
plt.figure(figsize=(12, 6))
plt.scatter(x_train, y_train, color='blue', label='Données réelles')
plt.plot(x_grid_train, Y_pred_polynomial_sorted, color='red', label=f'Régression polynomial ordre : {model_polynomial.degree}')
plt.plot(x_train_sorted, Y_pred_simple_sorted, color='green', linestyle='--', label='Régression simple')
plt.xticks(np.arange(np.min(x_train), np.max(x_train) + 2, 1))
plt.title('Régression Linéaire : Salaires')
plt.xlabel('Level')
plt.ylabel('Salaire')
plt.legend()
plt.savefig('rendu_3/img/regression_polynomial_Salary.png')

plt.show()
