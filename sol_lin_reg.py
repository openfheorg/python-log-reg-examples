from pprint import pprint

from typing import Tuple
import yaml
from naive_regression.crypto_utils import setup_crypto
from naive_regression.ematrix import EMatrix
import numpy as np

from naive_regression.np_reference import train

def predict(
        X:EMatrix,
        weights: EMatrix,
) -> EMatrix:
    ################################################
    # Exe: implement the prediction via. a dot-product.
    #       think carefully about what the out-packing might be
    ################################################
    return X.dot(weights, "vertical")


def calculate_loss(prediction: EMatrix, label: EMatrix,
                   inverse_num_samples_scale: float,
                   ) -> Tuple[EMatrix, EMatrix]:
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

    grad = X.dot(residuals, "vertical")
    grad = grad * -2 * scaling

    grad_alpha = grad * alpha
    repeated_grad_alpha = grad_alpha.vecConv2Hrep(repeat_weights_N_times)
    weights = weights - repeated_grad_alpha
    return weights, grad


np.random.seed(42)

if __name__ == '__main__':

    with open("naive_regression/config.yml", "r") as f:
        config = yaml.safe_load(f)
    print("ML Config:")
    pprint(config["ml_params"])
    print("Crypto Params:")
    pprint(config["crypto_params"])
    if config["crypto_params"]["run_bootstrap"]:
        print("Running with bootstrap")
        pprint(config["crypto_bootstrap_params"])
    ml_conf = config["ml_params"]
    batch_size = ml_conf["batch_size"]
    lr = ml_conf["lr"]
    epochs = ml_conf["epochs"]

    ################################################
    # Generate data
    ################################################

    X = np.random.rand(batch_size * 5, 5)
    y = (np.dot(X, np.random.rand(5, 1))) + np.random.rand(1)
    noise = np.random.randn(y.shape[0], y.shape[1])
    y = y + noise

    weights = np.random.rand(5, 1)
    print("#" * 10)
    print("Plaintext Performance")
    m_stat = train(X, y, weights, lr, epochs)

    print("#" * 10)
    print("Encrypted Performance")

    setup_crypto(
        num_data_points=-1 if config["crypto_params"]["run_bootstrap"] else len(X),
        c_params=config["crypto_params"],
        bootstrap_params=config["crypto_bootstrap_params"]
    )

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
    run_bootstrap_mode = config["crypto_params"]["run_bootstrap"]
    for epoch in range(epochs):
        y_pred = predict(e_X, weights)
        residuals, loss = calculate_loss(y_pred, label=e_y, inverse_num_samples_scale=inverse_scale)
        weights, grads = apply_gradient(e_X, weights, residuals, inverse_scale, lr, len(X))

        ################################################
        # Exe: it's not always realistic, but you may wish to displaty the loss
        ################################################

        print(f"epoch: {epoch} ----> MSE: {loss.decryptSelf()[0]}")

        ################################################
        # Exe: Our ciphertexts accumulate noise as we do computations. We have two options to handle the noise:
        #   - bootstrapping, which is expensive
        #   - decrypting and re-encrypting, which comes with its own tradeoffs
        #   Benchmark the two to get a feel for the timing difference
        ################################################


        if run_bootstrap_mode:
            weights.bootstrap_self()
        else:
            weights.recrypt_self()