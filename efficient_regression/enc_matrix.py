from typing import List
import openfhe

CT = openfhe.Ciphertext
CC = openfhe.CryptoContext


def get_vec_col_cloned(in_vec: List, num_slots: int) -> List:
    n = len(in_vec)
    if num_slots < n:
        raise Exception("num_clones * vec_size > num_slots")
    if num_slots == n:
        return in_vec
    num_clones = num_slots // n
    container = []
    for i in range(n):
        for j in range(num_clones):
            container.append(in_vec[i])
    return container


def get_vec_row_cloned(in_vec: List, num_slots, padding_val) -> List:
    n = len(in_vec)

    if num_slots < n:
        raise Exception("num_clones * vec_size > num_slots")
    if num_slots == n:
        return in_vec
    num_clones = num_slots // n

    container = []
    for i in range(num_clones):
        container.extend(in_vec)

    # Pad the remaining vals with padding_val
    container.extend([padding_val] * (num_slots - len(container)))
    return container


def matrix_vector_product_row(cc: CC, eval_sum_col_map, c_mat, c_vec_row_cloned, row_size):
    c_mult = cc.EvalMult(c_mat, c_vec_row_cloned)
    return cc.EvalSumCols(c_mult, row_size, eval_sum_col_map)


def matrix_vector_product_col(cc: CC, eval_sum_row_map, c_mat, c_vec_col_cloned, row_size):
    c_mult = cc.EvalMult(c_mat, c_vec_col_cloned)
    return cc.EvalSumRows(c_mult, row_size, eval_sum_row_map)
