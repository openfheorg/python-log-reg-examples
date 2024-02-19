from typing import Dict, List, Optional

import numpy as np
import openfhe

from efficient_regression.utils import get_raw_value_from_ct

CT = openfhe.Ciphertext
CC = openfhe.CryptoContext


def sol_logreg_calculate_grad(
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
        cheb_poly_degree: int,
        kp: Optional[openfhe.KeyPair] = None
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


def exe_logreg_calculate_grad(
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
        cheb_poly_degree: int,
        kp: Optional[openfhe.KeyPair] = None
) -> List:
    ################################################
    # Exe: extra hard!! Feel free to reference the sol_logreg_calculate_grad function above
    #      In this we will implement the following steps:
    #      1) logit calculation using dot-product
    #      2) applying the logistic function
    #      3) residual calculation
    #      4) gradient calculation (without using the X.T) via a dot-product
    # Two functions that will be useful for the dot-product are matrix_vector_product_row and matrix_vector_product_col
    # That you might want to look at
    ################################################
    pass



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

def logistic_function(x):    
    return 1/ (1 + np.exp(-x))

def compute_loss(
        beta: List,
        X: np.ndarray,
        y: np.ndarray
):
    # Plaintext loss computation that is based on https://stackoverflow.com/a/47798689/18031872
    beta = np.asarray(beta)
    num_samples = len(X)
    y_pred = logistic_function(X.dot(beta))
    _y = y.squeeze()
    error = (_y * np.log(y_pred)) + ((1 - _y) * np.log(1 - y_pred))
    cost = -1 / num_samples * sum(error)
    return cost


def matrix_vector_product_row(cc: CC, eval_sum_col_map, c_mat, c_vec_row_cloned, row_size):
    c_mult = cc.EvalMult(c_mat, c_vec_row_cloned)
    return cc.EvalSumCols(c_mult, row_size, eval_sum_col_map)


def matrix_vector_product_col(cc: CC, eval_sum_row_map, c_mat, c_vec_col_cloned, row_size):
    c_mult = cc.EvalMult(c_mat, c_vec_col_cloned)
    return cc.EvalSumRows(c_mult, row_size, eval_sum_row_map)
