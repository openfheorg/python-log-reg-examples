from typing import Dict, List

import numpy as np
from efficient_regression.enc_matrix import matrix_vector_product_col, matrix_vector_product_row
import openfhe

CT = openfhe.Ciphertext
CC = openfhe.CryptoContext


def optimize_x(X: np.ndarray, scaling_factor) -> np.ndarray:
    """
    Take the transpose, multiply by -1 / scaling factor. This optimizes the crypto-side
        of things in terms of run-time and mult depth. This also gets around the issue
        of needing to take an inverse
    """

    return X.T * (-1 / scaling_factor)


def encrypted_log_reg_calculate_gradient(
        cc: CC,
        ct_X: CT,
        ct_neg_Xt: CT,
        ct_y: CT,
        ctThetas,
        row_size: int,
        row_sum_keymap: Dict,
        col_sum_keymap: Dict,
        cheb_range_start: float,
        cheb_range_end: float,
        cheb_poly_degree: int
) -> List:
    """
    We use the same notation and setup as in https://eprint.iacr.org/2018/662.pdf
    """

    # Line 4. Generate the logits
    logits = matrix_vector_product_row(cc, col_sum_keymap, ct_X, ctThetas, row_size)
    # Line 5/6
    preds = cc.EvalLogistic(logits, cheb_range_start, cheb_range_end, cheb_poly_degree)
    # Line 8 - see page 9 for their notation
    residual = cc.EvalSub(ct_y, preds)

    gradients = matrix_vector_product_col(cc, row_sum_keymap, ct_neg_Xt,
                                          residual, row_size)
    return gradients


def re_encrypt(cc: CC, ct: CT, kp: openfhe.KeyPair):
    decrypted_res = cc.Decrypt(ct, kp.secretKey)
    return cc.Encrypt(
        kp.publicKey,
        cc.MakeCKKSPackedPlaintext(decrypted_res)
    )


def return_depth(ct: CT):
    # Ciphertext refreshing, the alternative to ciphertext-bootstrapping
    # TODO: implement if needed
    # https://github.com/openfheorg/openfhe-logreg-training-examples/blob/main/lr_train_funcs.cpp#L187
    mult_depth = ct.GetLevel()


# TODO: convert compute_loss https://github.com/openfheorg/openfhe-logreg-training-examples/blob/main/lr_train_funcs.cpp#L194
def compute_loss(
        beta: List,
        X: List,
        y: List
):
    # Plaintext loss computation that is based on https://stackoverflow.com/a/47798689/18031872
    num_samples = len(X)
    pass
