################################################
# Runner code for the naive implementation of regression
#   Note: this only runs on a minimal dataset to serve as a proof-of-concept
#   the correctness is benchmarked against a numpy implementation
#
#   This should not be misconstrued as a:
#       representative example
#       performance benchmark
################################################

from pprint import pprint

import openfhe
import pandas as pd
from typing import Tuple, List
import yaml
import numpy as np

from efficient_regression.crypto_utils import create_crypto
from efficient_regression.lr_train_funcs import encrypted_log_reg_calculate_gradient, compute_loss
from efficient_regression.utils import next_power_of_2, collate_one_d_mat_to_ct, mat_to_ct_mat_row_major, \
    one_d_mat_to_vec_col_cloned_ct, get_raw_value_from_ct

np.random.seed(42)

CT = openfhe.Ciphertext

import logging


def load_data(x_file, y_file, pct_train_on=0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xs = pd.read_csv(x_file).to_numpy()
    ys = pd.read_csv(y_file).to_numpy()
    # TODO: replace this....
    return Xs, ys, Xs, ys

    # total_size = len(Xs)
    #
    # idx_arr = np.arange(total_size)
    # np.random.shuffle(idx_arr)
    # train_idxs = idx_arr[:int(total_size * pct_train_on)]
    # test_idxs = idx_arr[int(total_size * pct_train_on):]

    # return Xs[train_idxs], ys[train_idxs], Xs[test_idxs], ys[test_idxs]


def generate_nag_mask(
        original_num_features,
        padded_row_size,
        padded_col_size,
        num_slots: int,
        cc: openfhe.CryptoContext,
        keys: openfhe.KeyPair,
) -> Tuple[openfhe.Plaintext, openfhe.Plaintext]:
    """
    Sets up all the relevant ciphertexts for machine learning including:
        - dataset
        - weights
        - optimizations
    """
    if padded_row_size * padded_col_size != num_slots:
        raise Exception("Padded row and col size must equal to number of slots")
    rotation_indices = [-padded_row_size, padded_row_size]
    cc.EvalRotateKeyGen(keys.secretKey, rotation_indices)

    # For the nesterov-accelerated gradients
    theta_mask = [0 for _ in range(num_slots)]
    phi_mask = [0 for _ in range(num_slots)]

    for i in range(num_slots):
        if (i / padded_row_size) % 2 == 0:
            theta_mask[i] = 1
        else:
            phi_mask[i] = 1

    return cc.MakeCKKSPackedPlaintext(theta_mask), cc.MakeCKKSPackedPlaintext(phi_mask)


if __name__ == '__main__':

    with open("config.yml", "r") as f:
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
    batch_size = ml_conf["batch_size"]
    lr_gamma = ml_conf["lr_gamma"]
    lr_eta = ml_conf["lr_eta"]
    epochs = ml_conf["epochs"]

    x_train, y_train, x_test, y_test = load_data(ml_conf["x_file"], ml_conf["y_file"], ml_conf["data_pct"])

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
    theta_mask, phi_mask = generate_nag_mask(
        original_num_features,
        padded_row_size,
        padded_col_size,
        num_slots,
        cc,
        kp
    )

    # Optimization: reduces the mult depth by 1
    # NOTE: we don't actually do the transpose. This is because when we use it later on
    #   we treat it as a col matrix, as opposed to a row matrix.
    neg_x_train_T = -1 * x_train * (lr_gamma / len(x_train))

    logger.debug("Generating the Sum keys")
    eval_sum_row_keys = cc.EvalSumRowsKeyGen(kp.secretKey, rowSize=padded_row_size)
    eval_sum_col_keys = cc.EvalSumColsKeyGen(kp.secretKey)

    # Encrypt the weights
    # https://github.com/openfheorg/openfhe-logreg-training-examples/blob/main/lr_nag.cpp#L302
    logger.debug("Generating Weights ciphertext")
    ct_weights = collate_one_d_mat_to_ct(cc, beta, beta, padded_row_size, num_slots, kp)

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

    #   https://github.com/openfheorg/openfhe-logreg-training-examples/blob/main/lr_nag.cpp#L311
    num_features_enc = next_power_of_2(original_num_features)
    num_slots_boot= num_features_enc * 8
    if config["crypto_params"]["run_bootstrap"]:
        logger.debug("Enabling FHE features for bootstrap")
        bootstrap_hparams = config["crypto_bootstrap_params"]
        level_budget = bootstrap_hparams["level_budget"]
        bsgs_dim = bootstrap_hparams["bsgs_dim"]
        cc.Enable(openfhe.PKESchemeFeature.FHE)
        cc.EvalBootstrapSetup(level_budget, bsgs_dim, num_slots_boot)
        cc.EvalBootstrapKeyGen(kp.secretKey, num_slots_boot)
        logger.debug("Bootstrap set up")


    for curr_epoch in range(epochs):

        print(f"************************************************************\nIteration: {curr_epoch}")

        if curr_epoch > 0:
            # Bootstrapping
            if config["crypto_params"]["run_bootstrap"]:
                logger.debug(f"Bootstrapping weights for iter: {curr_epoch}")
                ct_weights.SetSlots(num_slots_boot)
                if openfhe.get_native_int() == "128":
                    ct_weights = cc.EvalBootstrap(ct_weights)
                else:
                    ct_weights = cc.EvalBootstrap(ct_weights, 2)
            else:
                logger.debug(f"CT Refreshing for iter: {curr_epoch}")
                _pt_weights = cc.Decrypt(
                    kp.secretKey,
                    ct_weights
                )
                _raw_weights = _pt_weights.GetRealPackedValue()

                ct_weights = cc.Encrypt(
                    kp.publicKey,
                    cc.MakeCKKSPackedPlaintext(_raw_weights)
                )

        ################################################
        # Extract the weights
        ################################################
        _ct_theta = cc.EvalMult(ct_weights, theta_mask)
        ct_theta = cc.EvalAdd(
            cc.EvalRotate(_ct_theta, padded_row_size),
            _ct_theta
        )

        _ct_phi = cc.EvalMult(ct_weights, phi_mask)
        ct_phi = cc.EvalAdd(
            cc.EvalRotate(_ct_phi, -padded_row_size),
            _ct_phi
        )

        ct_gradient = encrypted_log_reg_calculate_gradient(
            cc,
            ct_x_train,
            ct_neg_x_train_T,
            ct_y,
            ct_theta,
            row_size=padded_row_size,
            row_sum_keymap=eval_sum_row_keys,
            col_sum_keymap=eval_sum_col_keys,
            cheb_range_start=config["chebyshev_params"]["lower_bound"],
            cheb_range_end=config["chebyshev_params"]["upper_bound"],
            cheb_poly_degree=config["chebyshev_params"]["polynomial_degree"],
        )
        ################################################
        # Note: Formulation of NAG update based on
        #   https://eprint.iacr.org/2018/462.pdf, Algorithm 1 and
        #   https://jlmelville.github.io/mize/nesterov.html
        ################################################

        ct_phi_prime = cc.EvalSub(
            ct_theta,
            ct_gradient
        )

        if (curr_epoch == 0):
            ct_theta = ct_phi_prime
        else:
            ct_theta = cc.EvalAdd(
                ct_phi_prime,
                cc.EvalMult(
                    lr_eta,
                    cc.EvalSub(ct_phi_prime, ct_phi)
                )
            )

        ct_phi = ct_phi_prime

        if config["RUN_IN_DEBUG"]:
            clear_theta = get_raw_value_from_ct(cc, ct_theta, kp, original_num_features)
            loss = compute_loss(beta=clear_theta, X=x_train, y=y_train)

            clear_phi = get_raw_value_from_ct(cc, ct_phi, kp, original_num_features)
            print(f"Theta: {clear_theta}")
            print(f"Phi: {clear_phi}")
            logger.debug(f"Loss: {loss}")

        # Repacking the two ciphertexts back into a single ciphertext
        ct_theta = cc.EvalMult(ct_theta, theta_mask)
        ct_phi = cc.EvalMult(ct_phi, phi_mask)
        ct_weights = cc.EvalAdd(ct_theta, ct_phi)
