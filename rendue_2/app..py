import numpy as np
import LinearRegression as l

X = np.array([2, 4, 6, 8, 10])
Y = np.array([[50, 100, 150, 200, 250]])

model = l.LinearRegression()

print(X.shape)
print(Y.shape)

model.fit(X, Y)

print(model.predict(12))

