import math
from typing import Dict, Optional, Tuple, List

import openfhe

from efficient_regression.clone_vec import get_vec_col_cloned, get_vec_row_cloned

CT = openfhe.Ciphertext
CC = openfhe.CryptoContext

def get_raw_value_from_ct(cc, ct, kp, length):
    loss_pt_beta: openfhe.Plaintext = cc.Decrypt(ct, kp.secretKey)
    loss_pt_beta.SetLength(length)
    return loss_pt_beta.GetRealPackedValue()


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def is_power_of_two(n):
    return math.log2(n).is_integer()


def matrix_to_row_major_vec(in_mat: List[List]) -> List:
    num_rows = len(in_mat)
    num_cols = len(in_mat[0])
    container = []
    for i in range(num_rows):
        for j in range(num_cols):
            container.append(in_mat[i][j])
    return container


def one_d_mat_to_vec(in_mat: List[List]) -> List:
    num_rows = len(in_mat)
    num_cols = len(in_mat[0])

    # Some simple error checking
    b1 = (num_rows == 1 and num_cols > 1)  # Checks that it's a (1, X) mat
    b2 = (num_rows > 1 and num_cols == 1)  #
    assert b1 or b2, f"Expected a mat of shape (1, X) or (X, 1). Received: ({num_rows, num_cols})"
    return matrix_to_row_major_vec(in_mat)


def one_d_mat_to_vec_col_cloned_ct(cc: CC, in_mat: List[List], row_size: int, num_slots: int,
                                    kp: openfhe.KeyPair) -> CT:
    in_vec = one_d_mat_to_vec(in_mat)
    orig_largest_size = len(in_vec)
    col_size = num_slots // row_size
    if orig_largest_size > col_size:
        raise Exception("Input vector largest dimension exceeds col_size (num_slots // row_size)")
    for i in range(orig_largest_size, col_size):
        in_vec.append(0.0)
    if not is_power_of_two(len(in_vec)):
        raise Exception("Post-padding, the input vector must be of length power of two")

    cloned_col_vec = get_vec_col_cloned(in_vec, num_slots)
    return cc.Encrypt(kp.publicKey, cc.MakeCKKSPackedPlaintext(cloned_col_vec))

def clone_vec_rc(in_mat: List[List], row_size: int, num_slots: int) -> List:
    in_vec = one_d_mat_to_vec(in_mat)
    original_num_rows = len(in_vec)
    if original_num_rows > row_size:
        raise Exception("Input vector number of rows exceeds the row_size param")

    for i in range(original_num_rows, row_size):
        in_vec.append(0.0)

    if not is_power_of_two(len(in_vec)):
        raise Exception("Input vector is not a power of two")

    return get_vec_row_cloned(in_vec, num_slots, 0.0)

def collate_one_d_mat_to_ct(
        cc: CC,
        m1: List[List],
        m2: List[List],
        row_size,
        num_slots,
        kp: openfhe.KeyPair
) -> CT:
    in_vec_rc = clone_vec_rc(m1, row_size, num_slots)
    in_vec_rc2 = clone_vec_rc(m2, row_size, num_slots)
    container = []
    for i in range(num_slots):
        if (i // row_size) % 2 == 0:
            container.append(in_vec_rc[i])
        else:
            container.append(in_vec_rc2[i])

    return cc.Encrypt(kp.publicKey, cc.MakeCKKSPackedPlaintext(container))

def mat_to_ct_mat_row_major(
    cc: CC,
        mat: List[List],
        row_size: int,
        num_slots: int,
        kp: openfhe.KeyPair
) -> CT:
    orig_num_rows = len(mat)
    orig_num_cols = len(mat[0])
    num_cols = row_size
    num_rows = num_slots // num_cols

    if orig_num_rows > num_rows:
        raise Exception("Error: input matrix # rows exceeds numRows")
    if orig_num_cols > num_cols:
        raise Exception("Error: input matrix # cols exceeds numCols")

    in_rmzp = [0.0] * num_slots  # Row major zero-padded, initialized with zeros

    k = 0  # Index into vector to write
    for i in range(orig_num_rows):
        for j in range(orig_num_cols):
            in_rmzp[k] = mat[i][j]  # Fill in a row with data
            k += 1
        k += num_cols - orig_num_cols  # Skip the rest of the row (already zero)

    return cc.Encrypt(
        kp.publicKey,
        cc.MakeCKKSPackedPlaintext(in_rmzp)
    )
