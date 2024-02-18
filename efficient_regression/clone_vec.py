from typing import List

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


