from typing import Dict, Tuple, Optional, List

import openfhe


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
        mult_depth,
        scale_mod_size,
        ring_dimensionality,
        crypto_batch_size,
        num_entries,  # Determines the ring dimensionality
        level_budget: Optional[List[int]] = None,
        levels_after_bootstrap: Optional[int] = None
) -> Tuple[openfhe.CryptoContext, openfhe.PublicKey, openfhe.PrivateKey, int, bool]:
    ################################################
    # Checking that the batch size isn't larger than what we support
    # if next_pow_of_2(num_entries) > crypto_batch_size:
    #     err_str = f"User-specified crypto batch ({crypto_batch_size}) size must be greater than the next-power-of-two of the number of entries({next_pow_of_2(num_entries)})"
    #     raise RuntimeError(err_str)
    if crypto_batch_size > ring_dimensionality / 2:
        err_str = "max batch size must be at most half of ring-dimensionality"
        raise RuntimeError(err_str)

    ################################################
    parameters = openfhe.CCParamsCKKSRNS()

    is_running_bootstrap =level_budget is not None and levels_after_bootstrap is not None
    if is_running_bootstrap:

        if len(level_budget) != 2:
            err_str = "level budget is specified by a tuple of two elements"
            raise RuntimeError(err_str)
        print("Running in bootstrap mode")
        secret_key_dist = openfhe.SecretKeyDist.UNIFORM_TERNARY
        parameters.SetSecretKeyDist(secret_key_dist)
        mult_depth = levels_after_bootstrap + openfhe.FHECKKSRNS.GetBootstrapDepth(
            level_budget, secret_key_dist)

    parameters.SetMultiplicativeDepth(mult_depth)
    parameters.SetScalingModSize(scale_mod_size)
    # parameters.SetBatchSize(crypto_batch_size)
    parameters.SetBatchSize(ring_dimensionality)
    # parameters.SetRingDim(ring_dimensionality)

    cc: openfhe.CryptoContext = openfhe.GenCryptoContext(parameters)
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.ADVANCEDSHE)
    cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)

    if is_running_bootstrap:
        cc.Enable(openfhe.PKESchemeFeature.FHE)

    keypair: openfhe.KeyPair = cc.KeyGen()

    rot_k = generate_rotation_keys(cc, keypair.secretKey, num_entries)
    cc.EvalMultKeyGen(keypair.secretKey)
    cc.EvalSumKeyGen(keypair.secretKey)
    if is_running_bootstrap:
        cc.EvalBootstrapSetup(level_budget)
        cc.EvalBootstrapKeyGen(keypair.secretKey, crypto_batch_size)

    return cc, keypair.publicKey, keypair.secretKey, rot_k, is_running_bootstrap