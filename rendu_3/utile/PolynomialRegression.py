import utile.LinearRegression as l
import pandas as pd
import utile.Scaler as Scaler

class PolynomialRegression(l.LinearRegression):


    def __init__(self):
        super().__init__()
        self.degree = None
    
    def fit(self, X, Y, normalised = True):
        scaler = Scaler.Scaler()
        
        best_B = None
        best_mse_polynomial = float('inf')

        for degree in range(2,6):
            df = pd.DataFrame(X, columns=["temperature"])
            for i in range(2, degree + 1):
                df["temperature" + str(i)] = df['temperature'] ** i
            if normalised:
                scaler.fit_transform(df)
                X_norm = df.to_numpy()
            else:
                X_norm = df.to_numpy()
            super().fit(X_norm, Y)
            Y_pred_polynomial = self.predict(X_norm)
            mse_polynomial = self.MSE(Y, Y_pred_polynomial)

            if (mse_polynomial < best_mse_polynomial):
                self.degree = degree
                best_mse_polynomial = mse_polynomial
                best_B = self.B
        self.B = best_B


    def predict(self, X):
        if X.shape[1] > 1:
            return super().predict(X)
        df = pd.DataFrame(X, columns=["temperature"])
        for i in range(2, self.degree + 1):
            df[f"temperature{i}"] = df['temperature'] ** i
        X_poly = df.to_numpy()
        return super().predict(X_poly)