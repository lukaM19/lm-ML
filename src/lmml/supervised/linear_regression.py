import numpy as np
import pandas as pd

class LinearRegression():
    def __init__(self, lr = 0.01, n_iters = 1000, grad_descent = False):
        self.learning_rate = lr
        self.n_iters = n_iters
        self.intercept = None
        self.coefficients = None
        self.grad_descent = grad_descent

        if grad_descent:
            self.fit = self._fit_gd
            self.predict = self._predict_gd
        else:
            self.fit = self._fit_cf
            self.predict = self._predict_cf

    def _fit_gd(self, X, y):
        
        n_samples, n_features = X.shape
        self.coefficients = np.zeros(n_features)
        self.intercept = 0
        
        # Gradient descent
        for i in range(self.n_iters):
            y_pred = np.dot(X, self.coefficients) + self.intercept
            coef_grad = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            intercept_grad = (1 / n_samples) * np.sum(y_pred - y)

            self.coefficients -= self.learning_rate * coef_grad
            self.intercept -= self.learning_rate * intercept_grad

    def _predict_gd(self, X):
        return np.dot(X, self.coefficients) + self.intercept

    
    def _fit_cf(self, X, y):

        if self.grad_descent :
            self.fit_gd(X,y)
        else:
            
            # Add a column of ones to the input features to account for the intercept
            X_b = np.c_[np.ones((X.shape[0], 1)), X]  # X_b is X with a column of ones

            # Compute the optimal values for the coefficients using the normal equation
            # theta_best = (X_b.T @ X_b)^{-1} @ X_b.T @ y
            theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

            # The intercept (theta_0) is the first element, and the coefficients are the rest
            self.intercept = theta_best[0]
            self.coefficients = theta_best[1:]
    
    def _predict_cf(self, X):

        if self.grad_descent :
            return self.predict_gd(X)
        else:

            # Add a column of ones to the input features to account for the intercept
            X_b = np.c_[np.ones((X.shape[0], 1)), X]

            # Compute the predictions using the learned coefficients
            y_pred = X_b.dot(np.r_[self.intercept, self.coefficients])
            return y_pred
