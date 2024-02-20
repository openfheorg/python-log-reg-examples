import openfhe
from typing import Dict, List, Optional, Tuple

import yaml

from efficient_regression.crypto_utils import create_crypto
from efficient_regression.lr_train_funcs import matrix_vector_product_row
import logging
import numpy as np
import pandas as pd

from efficient_regression.utils import next_power_of_2, encrypt_weights, mat_to_ct_mat_row_major, get_raw_value_from_ct, \
    one_d_mat_to_vec_col_cloned_ct

CT = openfhe.Ciphertext
CC = openfhe.CryptoContext


def predict(
        cc: CC,
        ct_X: CT,
        ctThetas,
        row_size: int,
        col_sum_keymap: Dict,
        cheb_range_start: float,
        cheb_range_end: float,
        cheb_poly_degree: int,
) -> List:
    ################################################
    # Exe:
    #     implement the dot-product to generate the logits via:
    #     - hadamard product
    #     - EvalSumCols
    #     cc.EvalLogistic() to generate the prediction
    ################################################
    raise NotImplementedError("Implement the prediction")


def repeat_and_encrypt_weights(
        cc: CC,
        trained_weights: List[float],
        padded_row_size: int,
        num_slots: int,
        kp: openfhe.KeyPair
):
    ################################################
    # Exe: test your understanding of the repeated packing!
    #      1) pad the trained_weights vector
    #      2) repeat the weight across the number-of-slots
    #      3) pack the plaintext
    #      4) encrypt
    ################################################
    raise NotImplementedError("Implement the repeat_and_encrypt_weights")


def load_data(x_file, y_file) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xs = pd.read_csv(x_file).to_numpy()
    ys = pd.read_csv(y_file).to_numpy()
    return Xs, ys, Xs, ys


if __name__ == '__main__':
    # Trained in-the-clear using the `logreg_reference.ipynb`

    trained_weights = [-0.83946494, 0.1006747, -0.86173275, 0.41098421, -0.55124025,
                       -0.09287871, -0.03976215, -0.20657445, 0.06133055, 0.24880721]

    with open("efficient_regression/inference_config.yml", "r") as f:
        config = yaml.safe_load(f)

    logging.basicConfig(format="[%(filename)s:%(lineno)s - %(funcName)s] %(message)s",
                        level=getattr(logging, config["logging_level"]))
    logger = logging.getLogger(__name__)

    logger.debug("ML Params")
    logger.debug(config["ml_params"])
    logger.debug("Crypto Params")
    logger.debug(config["crypto_params"])
    logger.debug("Chebyshev Params")
    logger.debug(config["chebyshev_params"])
    if config["crypto_params"]["run_bootstrap"]:
        logger.info("Running with Bootstrap")
        logger.debug(config["crypto_bootstrap_params"])
    ml_conf = config["ml_params"]
    lr_gamma = ml_conf["lr_gamma"]
    lr_eta = ml_conf["lr_eta"]
    epochs = ml_conf["epochs"]

    x_train, y_train, x_test, y_test = load_data(ml_conf["x_file"], ml_conf["y_file"])

    original_num_samples, original_num_features = x_train.shape

    logger.debug("Generating crypto objects")
    cc, kp, num_slots = create_crypto(
        crypto_hparams=config["crypto_params"],
        bootstrap_hparams=config["crypto_bootstrap_params"]
    )
    logger.debug("Generating crypto objects")
    padded_row_size = next_power_of_2(original_num_features)
    padded_col_size = num_slots / padded_row_size

    logger.debug("Generating the Sum keys")
    eval_sum_col_keys = cc.EvalSumColsKeyGen(kp.secretKey)

    # Encrypt the weights
    logger.debug("Generating Weights ciphertext")
    ct_x_train = mat_to_ct_mat_row_major(
        cc,
        x_train.tolist(),
        padded_row_size,
        num_slots,
        kp
    )

    ct_weights = repeat_and_encrypt_weights(
        cc,
        trained_weights,
        padded_row_size,
        num_slots,
        kp
    )

    predictions = predict(cc, ct_x_train, ct_weights, padded_row_size,
                          col_sum_keymap=eval_sum_col_keys,
                          cheb_range_start=config["chebyshev_params"]["lower_bound"],
                          cheb_range_end=config["chebyshev_params"]["upper_bound"],
                          cheb_poly_degree=config["chebyshev_params"]["polynomial_degree"],
                          )

    packed_preds: openfhe.Plaintext = cc.Decrypt(predictions, kp.secretKey)
    clear_preds = []
    packed_predictions = packed_preds.GetRealPackedValue()
    for idx in range(0, len(packed_predictions), padded_row_size):
        clear_preds.append(packed_predictions[idx])

    for i, (y_hat, y) in enumerate(zip(clear_preds, y_train)):
        if i > 10:
            break
        print(f"Prediction: {y_hat}, Rounded: {np.round(y_hat)}, Label: {y}")
