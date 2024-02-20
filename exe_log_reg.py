################################################
# Solution for logistic regression exercise
################################################

from pprint import pprint

import openfhe
import pandas as pd
from typing import Tuple, List
import yaml
import numpy as np

from efficient_regression.crypto_utils import create_crypto
from efficient_regression.lr_train_funcs import sol_logreg_calculate_grad, compute_loss, exe_logreg_calculate_grad
from efficient_regression.utils import next_power_of_2, collate_one_d_mat_to_ct, mat_to_ct_mat_row_major, \
    one_d_mat_to_vec_col_cloned_ct, get_raw_value_from_ct, encrypt_weights

np.random.seed(42)

CT = openfhe.Ciphertext
CC = openfhe.CryptoContext

import logging


def load_data(x_file, y_file) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xs = pd.read_csv(x_file).to_numpy()
    ys = pd.read_csv(y_file).to_numpy()
    return Xs, ys, Xs, ys


def update_weights(cc: CC, ct_weights: CT, grads: CT, lr: float):
    ################################################
    # Implement a gradient step.
    # Functions you may find useful:
    #   - cc.EvalSub
    #   - cc.EvalMult
    ################################################
    pass


def reduce_noise(
        cc,
        ct_weights,
        should_run_bootstrap,
        num_slots_boot,
        kp
):
    ################################################
    # Exe: handle noise refreshing for both the bootstrap and iterative
    #       mode.
    #      See what happens if you forget to set the number-of-iterations in EvalBootstrap
    ################################################
    pass




if __name__ == '__main__':

    with open("efficient_regression/config.yml", "r") as f:
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
    beta = [[0.0] for _ in range(original_num_features)]

    logger.debug("Generating crypto objects")
    cc, kp, num_slots = create_crypto(
        crypto_hparams=config["crypto_params"],
        bootstrap_hparams=config["crypto_bootstrap_params"]
    )
    logger.debug("Generating crypto objects")
    padded_row_size = next_power_of_2(original_num_features)
    padded_col_size = num_slots / padded_row_size

    # Exe: reduces the mult depth by 1
    # NOTE: we don't actually do the transpose. This is because when we use it later on
    #   we treat it as a col matrix, as opposed to a row matrix.
    neg_x_train_T = -1 * x_train * (1 / len(x_train))

    logger.debug("Generating the Sum keys")
    eval_sum_row_keys = cc.EvalSumRowsKeyGen(kp.secretKey, rowSize=padded_row_size)
    eval_sum_col_keys = cc.EvalSumColsKeyGen(kp.secretKey)

    # Encrypt the weights
    logger.debug("Generating Weights ciphertext")
    ct_weights = encrypt_weights(cc, kp, beta)

    logger.debug("Generating X-ciphertext")
    ct_x_train = mat_to_ct_mat_row_major(
        cc,
        x_train.tolist(),
        padded_row_size,
        num_slots,
        kp
    )
    ct_neg_x_train_T = mat_to_ct_mat_row_major(
        cc,
        neg_x_train_T.tolist(),
        padded_row_size,
        num_slots,
        kp
    )

    logger.debug("Generating y-ciphertext")
    ct_y = one_d_mat_to_vec_col_cloned_ct(
        cc,
        y_train.tolist(),
        padded_row_size,
        num_slots,
        kp
    )

    num_features_enc = next_power_of_2(original_num_features)
    num_slots_boot = num_features_enc * 8
    if config["crypto_params"]["run_bootstrap"]:
        logger.info("Enabling FHE features for bootstrap")
        bootstrap_hparams = config["crypto_bootstrap_params"]
        level_budget = bootstrap_hparams["level_budget"]
        bsgs_dim = bootstrap_hparams["bsgs_dim"]
        cc.Enable(openfhe.PKESchemeFeature.FHE)
        cc.EvalBootstrapSetup(level_budget, bsgs_dim, num_slots_boot)
        cc.EvalBootstrapKeyGen(kp.secretKey, num_slots_boot)
        logger.debug("Bootstrap set up")

    for curr_epoch in range(epochs):

        # print(f"************************************************************\nIteration: {curr_epoch}")

        if curr_epoch > 0:
            ct_weights = reduce_noise(
                cc=cc,
                ct_weights=ct_weights,
                should_run_bootstrap=config["crypto_params"]["run_bootstrap"],
                num_slots_boot=num_slots_boot,
                kp = kp
            )

        ################################################
        # Extract the weights
        ################################################

        # Exe: Navigate to the exercise function for an extra difficult problem. If for time constraints you want to
        #       skip this (or come back to this later), comment out the first line and uncomment the second.

        # Exe: Navigate to the exercise function for an extra difficult problem. If for time constraints you want to
        #       skip this (or come back to this later), comment out the first line and uncomment the second.
        ct_gradient = exe_logreg_calculate_grad(
        # ct_gradient = sol_logreg_calculate_grad(
            cc,
            ct_x_train,
            ct_neg_x_train_T,
            ct_y,
            ct_weights,
            row_size=padded_row_size,
            row_sum_keymap=eval_sum_row_keys,
            col_sum_keymap=eval_sum_col_keys,
            cheb_range_start=config["chebyshev_params"]["lower_bound"],
            cheb_range_end=config["chebyshev_params"]["upper_bound"],
            cheb_poly_degree=config["chebyshev_params"]["polynomial_degree"],
            kp=kp
        )

        if ct_gradient is None:
            raise Exception("You either "
                            "\ni) have not implemented exe_logreg_calculate_grad or "
                            "\nii) forgot to flip the function call to sol_logreg_calculate_grad")

        ct_weights = update_weights(cc, ct_weights, ct_gradient, lr_eta)

        if config["RUN_IN_DEBUG"]:
            clear_theta = get_raw_value_from_ct(cc, ct_weights, kp, original_num_features)
            loss = compute_loss(beta=clear_theta, X=x_train, y=y_train)

            clear_grads = get_raw_value_from_ct(cc, ct_gradient, kp, original_num_features)
            logger.info(f"Grad: {clear_grads}")
            logger.info(f"Theta: {clear_theta}")
            logger.info(f"Iteration: {curr_epoch} Loss: {loss}")