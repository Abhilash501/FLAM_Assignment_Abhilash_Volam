import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

#Load data
data = pd.read_csv("xy_data.csv")
x_data = data["x"].values
y_data = data["y"].values
t_data = np.linspace(6, 60, len(x_data))

#Define model
def model(params, t):
    theta, M, X = params
    x = t * np.cos(theta) - np.exp(M * t) * np.sin(0.3 * t) * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + np.exp(M * t) * np.sin(0.3 * t) * np.cos(theta)
    return x, y

#Define residuals
def residuals(params):
    x_pred, y_pred = model(params, t_data)
    return np.concatenate([(x_pred - x_data), (y_pred - y_data)])

#Initial guess and bounds
theta0 = np.radians(25)
M0 = 0.0
X0 = 50.0
bounds = ([np.radians(0), -0.05, 0], [np.radians(50), 0.05, 100])

#Fit parameters
result = least_squares(residuals, [theta0, M0, X0], bounds=bounds)
theta_opt, M_opt, X_opt = result.x

#Print results
print(f"Optimized θ = {np.degrees(theta_opt):.4f}°")
print(f"Optimized M = {M_opt:.5f}")
print(f"Optimized X = {X_opt:.5f}")

#Compute predicted curve
x_fit, y_fit = model(result.x, t_data)

#Plot results
plt.plot(x_data, y_data, 'o', label='Data')
plt.plot(x_fit, y_fit, '-', label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Parametric Curve Fitting')
plt.show()
