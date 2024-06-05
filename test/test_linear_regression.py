import numpy as np
import pytest
from src.lmml.supervised.linear_regression import LinearRegression

@pytest.fixture
def data():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    return X, y

def test_fit_normal(data):
    X, y = data
    model = LinearRegression(grad_descent=False)
    model.fit(X, y)
    y_pred = model.predict(X)
    np.testing.assert_almost_equal(y_pred, y, decimal=5, err_msg="Normal equation method failed.")

def test_fit_gd(data):
    X, y = data
    model = LinearRegression(lr=0.1, n_iters=2000, grad_descent=True)
    model.fit(X, y)
    y_pred = model.predict(X)
    np.testing.assert_almost_equal(y_pred, y, decimal=1, err_msg=f"Gradient descent method failed. y_pred: {y_pred}, y: {y}")

def test_predict_normal(data):
    X, y = data
    model = LinearRegression(grad_descent=False)
    model.fit(X, y)
    y_pred = model.predict(X)
    assert len(y_pred) == len(y), "Predict method for normal equation failed."

def test_predict_gd(data):
    X, y = data
    model = LinearRegression(lr=0.1, n_iters=2000, grad_descent=True)
    model.fit(X, y)
    y_pred = model.predict(X)
    assert len(y_pred) == len(y), "Predict method for gradient descent failed."

if __name__ == "__main__":
    pytest.main()
