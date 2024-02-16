from ematrix import EMatrix
from typing import Optional, List, Tuple, Union
import copy
import numpy as np


def predict(regression_type: str,
            X:EMatrix,
            weights: EMatrix,
            nonlinearity_stats: Optional[List] = None) -> Tuple[EMatrix, Union[List, None]]:
    """
    nonlinearity_stats are for in case you want to store the data going into the
        approximations. This is particularly useful if you're finding that the data
        is blowing up or has stopped making sense. One source of issues is typically
        that your computations have accumulated too much noise, but it can also be
        the result of your data exceeding the sigmoid obunds
    """
    # compute estimates as dot product between features and weights

    dot_prod = X.dot(weights, "vertical")
    if regression_type == "normal" or regression_type == "linear":
        return dot_prod, nonlinearity_stats
    elif regression_type == "logistic" or regression_type == "sigmoid":
        if nonlinearity_stats is not None:
            nonlinearity_stats.append(dot_prod)
        eVv_y_hat = EMatrix.sigmoid(dot_prod)
    else:
        print(f"Unsupported regression type {regression_type}")
        exit(-1)

    return eVv_y_hat, nonlinearity_stats


def calculate_loss(prediction: EMatrix, label: EMatrix,
                   inverse_num_samples_scale: float,
                   ) -> tuple[EMatrix, EMatrix]:
    residuals = label - prediction

    # compute error (difference between estimate y_hat and true value y)
    sq_error = residuals.hprod(residuals)
    enc_SSE = sq_error.sum()

    enc_SSE *= inverse_num_samples_scale
    return residuals, enc_SSE


def apply_gradient(
        X: EMatrix,
        weights: EMatrix,
        residuals: EMatrix,
        scaling: float,
        alpha: float,
        repeat_weights_N_times: int,
) -> Tuple[EMatrix, EMatrix]:
    """
    We return the new weights and the gradients to generate these weights
        this is to allow us to inspect if we need to.
    """
    # Internally, the dot product handles the need for the transpose.
    # Note: taking the transpose of an encrypted vector, without special tricks,
    #   is expensive
    grad = X.dot(residuals, "vertical")
    grad = grad * -2 * scaling

    grad_alpha = grad * alpha
    repeated_grad_alpha = grad_alpha.vecConv2Hrep(repeat_weights_N_times)
    weights = weights - repeated_grad_alpha
    return weights, grad


def recryption(
        grad: EMatrix,
        weights: EMatrix,
) -> Tuple[EMatrix, EMatrix]:
    """
    Recrypt to refresh the noise
    """
    eVv_old_grad = grad.recrypt()
    weights.recrypt_self()
    return weights, eVv_old_grad


def train(
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        alpha: float,
        n_epochs: int,
        run_bootstrap_mode = False
):
    inverse_scale = 1 / len(y)

    ####################################################################
    # We need to repeat the weights N-times bc we do the hadamard product then sum
    #   when we're doing the dot product
    weights = np.squeeze(weights, axis=1).tolist()
    repeated_weights = []
    for i in range(len(X)):
        repeated_weights.append(weights)
    weights = EMatrix.fromList(repeated_weights, packing="vertical", repeated=True)
    weights.encryptSelf()

    ####################################################################
    # We encrypt all at once. NOTE: this is not a true SGD - we're not shuffling
    #   between each epoch. Having said that, this is WAY faster
    e_X = EMatrix.fromList(X.tolist())
    e_y = EMatrix.fromList(y.tolist())
    e_X.encryptSelf()
    e_y.encryptSelf()
    # errors = []
    for epoch in range(n_epochs):
        y_pred, _ = predict("linear", e_X, weights, [])
        residuals, loss = calculate_loss(y_pred, label=e_y, inverse_num_samples_scale=inverse_scale)
        weights, grads = apply_gradient(e_X, weights, residuals, inverse_scale, alpha, len(X))

        ################################################
        # The following isn't realistic - we don't always know the loss. However,
        #   this is primarily for demo purposes
        ################################################
        loss.decryptSelf()
        print(f"epoch: {epoch} ----> MSE: {loss[0]}")

        # Things accumulate noise as we do computations. These following are options to handle the noise:
        #   - bootstrapping, which is expensive
        #   - decrypting and re-encrypting, which comes with its own tradeoffs
        if run_bootstrap_mode:
            weights.bootstrap_self()
        else:
            weights.recrypt_self()