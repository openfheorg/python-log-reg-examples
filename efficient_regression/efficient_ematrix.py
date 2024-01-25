import copy
import random
import unittest
from collections import deque
import numpy as np

import openfhe

import operator
from typing import Union, List, Tuple, Dict
from enum import Enum
from typing import Optional


CT = openfhe.Ciphertext

def clone_vec_row(in_vec: List, num_slots, padding_val):
    """
    Convert a 1d vector to col-cloned (input must be zero padded to power of two)
    Returns:
    """
    pass

def clone_vec_col():
    """
    Convert a 1d vector to col-cloned (input must be zero padded to power of two)
    Returns:

    """
    pass