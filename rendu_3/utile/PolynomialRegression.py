import pandas as pd
import utile.Scaler as Scaler
import numpy as np

class PolynomialRegression():


    def __init__(self, normalised = True):
        self.degree = None
        self.normalised = normalised
        self.scaler = Scaler.Scaler()
    
    def fit(self, X, Y):
        best_degree = 1
        best_B = None
        best_mse_polynomial = float('inf')
        ridge = 1e-6
        for degree in range(1,6):
            scaler = Scaler.Scaler()
            df = pd.DataFrame(X, columns=["temperature"])
            self.degree = degree
            for i in range(2, degree + 1):
                df["temperature" + str(i)] = df['temperature'] ** i
            if self.normalised:
                scaler.fit_transform(df)
            X_norm = df.to_numpy()
            X_norm = np.c_[np.ones(X_norm.shape[0]), X_norm]
            self.B = np.linalg.inv(X_norm.T @ X_norm + ridge * np.eye(X_norm.shape[1])) @ (X_norm.T @ Y)
            
            Y_pred_polynomial = self.predict(X)
            mse_polynomial = self.MSE(Y, Y_pred_polynomial)

            if (mse_polynomial < best_mse_polynomial):
                best_degree = degree
                best_mse_polynomial = mse_polynomial
                self.__set_coef_determination(Y_pred_polynomial, Y)
                self.scaler = scaler
                best_B = self.B
        self.B = best_B
        self.degree = best_degree


    def predict(self, X, col_1_presente=False):
        df = pd.DataFrame(X, columns=["temperature"])
        for i in range(2, self.degree + 1):
            df[f"temperature{i}"] = df['temperature'] ** i
        if self.normalised:
            self.scaler.transform(df)
        X_poly = df.to_numpy()
        if not col_1_presente:
            X_poly = np.c_[np.ones(X_poly.shape[0]), X_poly]
        return X_poly @ self.B
    
    def MSE(self, YR, YP):
        return np.mean((YR - YP)**2)
    
    def __set_coef_determination(self, Y_pred, Y):
        SCE = np.sum((Y_pred - np.mean(Y)) ** 2)
        SCT = np.sum((Y - np.mean(Y)) ** 2)
        self.coef_determination = SCE / SCT


    def get_coef_determination(self):
        return self.coef_determination