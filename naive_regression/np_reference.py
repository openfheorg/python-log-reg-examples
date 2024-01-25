################################################
# Linear Regression
################################################

import numpy as np


def predict(X, weights):
    return np.dot(X, weights)

def calculate_loss(prediction, y, scaling):
    residuals = y - prediction
    return residuals, np.sum(residuals ** 2) * scaling

def apply_gradient(X, weights, residuals, scaling, alpha):
    df_dw = -2 * scaling * (X.T @ residuals)
    df_dw = df_dw.reshape(len(df_dw), -1)
    df_dw *= alpha

    weights = weights - df_dw
    return weights


def train(X, y, weights, lr, epochs):
    for epoch in range(epochs):
        y_pred = predict(X, weights)
        residuals, loss = calculate_loss(y_pred, y, 1 / len(y))
        weights = apply_gradient(X, weights, residuals, 1 / len(X), alpha=lr)

        print(f"epoch: {epoch} ----> MSE: {round(loss, ndigits=5)}")

    return weights