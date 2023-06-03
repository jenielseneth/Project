import numpy as np
from scipy.optimize import minimize

def lppl(t, A, B, tc, m, C1, C2, omega):
    """
    LPPL function for fitting.
    
    t: Time array
    A, B, tc, m, C1, C2, omega: Parameters of the LPPL function
    """
    return A + B * (tc - t) ** m * (1 + C1 * np.cos(omega * np.log(tc - t)) + C2 * np.sin(omega * np.log(tc - t)))

def objective(params, t, y):
    """
    Objective function to minimize using LPPL parameters.
    
    params: Array of LPPL parameters to optimize
    t: Time array
    y: S&P 500 price array
    """
    A, B, tc, m, C1, C2, omega = params
    y_pred = lppl(t, A, B, tc, m, C1, C2, omega)
    error = np.sum((y_pred - y) ** 2)  # Sum of squared errors
    return error

def fit_lppl(t, y):
    """
    Fit LPPL function to S&P 500 data using downhill simplex optimization.
    
    t: Time array
    y: S&P 500 price array
    """
    # Initial guess for LPPL parameters
    initial_params = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
    
    # Minimize objective function using downhill simplex optimization
    result = minimize(objective, initial_params, args=(t, y), method='Nelder-Mead')
    
    # Extract optimized LPPL parameters
    A, B, tc, m, C1, C2, omega = result.x
    
    return A, B, tc, m, C1, C2, omega

# Example usage
t = np.array([1, 2, 3, 4, 5])  # Time array (replace with actual time data)
y = np.array([100, 120, 130, 140, 150])  # S&P 500 price array (replace with actual price data)

A, B, tc, m, C1, C2, omega = fit_lppl(t, y)
print("Optimized LPPL parameters:")
print("A:", A)
print("B:", B)
print("tc:", tc)
print("m:", m)
print("C1:", C1)
print("C2:", C2)
print("omega:", omega)
