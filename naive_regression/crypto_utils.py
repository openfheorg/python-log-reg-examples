from typing import Dict, Tuple, Optional, List

import openfhe
from openfhe import PKESchemeFeature

from naive_regression.ematrix import EMatrix


def next_pow_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def generate_rotation_keys(
        cc: openfhe.CryptoContext,
        pub_key: openfhe.PrivateKey,
        max_rotation: int) -> int:
    """
    Generate the rotation keys using baby, giant steps
    Args:
        cc:
        max_rotation:

    Returns:

    """
    import math

    ring_dimension = cc.GetRingDimension()
    minval = next_pow_of_2(min(ring_dimension / 2, max_rotation))
    rot_k = int(math.ceil(math.sqrt(minval)))
    rotations = list(range(1, rot_k + 1))
    for i in range(2 * rot_k, minval - 1, rot_k):
        rotations.append(i)
    neg_rotations = []
    for i in rotations:
        neg_rotations.append(-i)
    cc.EvalRotateKeyGen(pub_key, neg_rotations)
    return rot_k


def setup_crypto(
        num_data_points: int,
        c_params: Dict,
        bootstrap_params: Optional[Dict] = None
):
    # CKKS cryptographic parameters
    if openfhe.get_native_int() == 128:
        print("Running in 128-bit mode")
        rescale_tech = openfhe.ScalingTechnique.FIXEDAUTO
        scaling_mod_size = c_params["128_scaling_mod_size"]
        first_mod = c_params["128_first_mod"]
    else:
        print("Running in 64-bit mode")
        rescale_tech = openfhe.ScalingTechnique.FLEXIBLEAUTO
        scaling_mod_size = c_params["64_scaling_mod_size"]
        first_mod = c_params["64_first_mod"]

    if c_params["run_bootstrap"]:
        level_budget = bootstrap_params["level_budget"]
        levels_after_bootstrap = bootstrap_params["levels_after_bootstrap"]
        secret_key_dist = openfhe.SecretKeyDist.UNIFORM_TERNARY
        mult_depth = levels_after_bootstrap + openfhe.FHECKKSRNS.GetBootstrapDepth(level_budget, secret_key_dist)
    else:
        mult_depth = c_params["mult_depth"]

    ################################################
    # Setting the parameters
    ################################################
    parameters = openfhe.CCParamsCKKSRNS()
    parameters.SetScalingModSize(scaling_mod_size)
    parameters.SetScalingTechnique(rescale_tech)
    parameters.SetFirstModSize(first_mod)

    parameters.SetMultiplicativeDepth(mult_depth)

    cc = openfhe.GenCryptoContext(parameters)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)
    cc.Enable(PKESchemeFeature.FHE)

    if c_params["num_slots"]:
        num_slots = c_params["num_slots"]
    else:
        num_slots = int(cc.GetRingDimension() / 2)

    if c_params["run_bootstrap"]:
        cc.EvalBootstrapSetup(level_budget)
    keypair: openfhe.KeyPair = cc.KeyGen()
    rot_k = generate_rotation_keys(cc, keypair.secretKey, num_data_points)
    cc.EvalMultKeyGen(keypair.secretKey)
    cc.EvalSumKeyGen(keypair.secretKey)
    if c_params["run_bootstrap"]:
        cc.EvalBootstrapKeyGen(keypair.secretKey, num_slots)

    eMatrix = EMatrix(-1, -1, init=False)
    eMatrix.set_crypto(cc, rot_k, keypair.publicKey, keypair.secretKey)
