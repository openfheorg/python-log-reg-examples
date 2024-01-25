import logging
import math
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)
import openfhe

CT = openfhe.Ciphertext
CC = openfhe.CryptoContext

def create_crypto(
        crypto_hparams: Dict,
        bootstrap_hparams: Optional[Dict] = None
) -> Tuple[openfhe.CryptoContext, openfhe.KeyPair, int]:
    ################################################
    # Setting the crypto parameters
    ################################################
    # CKKS cryptographic parameters
    if openfhe.get_native_int() == 128:
        logging.info("Running in 128-bit mode")
        scaling_mod_size = crypto_hparams["128_scaling_mod_size"]
        first_mod = crypto_hparams["128_first_mod"]
    else:
        logging.info("Running in 64-bit mode")
        scaling_mod_size = crypto_hparams["64_scaling_mod_size"]
        first_mod = crypto_hparams["64_first_mod"]

    # Define the parameter object that will be used in the cryptocontext creation
    cc_params = openfhe.CCParamsCKKSRNS()
    # If running in bootstrap mode, this value gets overwritten!
    mult_depth = crypto_hparams["mult_depth"]

    if crypto_hparams["run_bootstrap"]:
        sk_dist = getattr(openfhe, bootstrap_hparams["secret_key_dist"] )
        level_budget = bootstrap_hparams["level_budget"]
        bsgs_dim = bootstrap_hparams["bsgs_dim"]

        if openfhe.get_native_int() == 128:
            levels_before_bootstrap = bootstrap_hparams["levels_before_bootstrap_128bit"]
        else:
            levels_before_bootstrap = bootstrap_hparams["levels_before_bootstrap_64bit"]

        approx_bootstrap_depth = bootstrap_hparams["approx_bootstrap_depth"]
        mult_depth = levels_before_bootstrap + openfhe.FHECKKSRNS.GetBootstrapDepth(
            approx_bootstrap_depth, level_budget, sk_dist
        )
        logger.info(f"Bootstrap final depth: {mult_depth}")
        cc_params.SetSecretKeyDist(sk_dist)
    # Setting is ordered based on the reference C++ implementation
    # https://github.com/openfheorg/openfhe-logreg-training-examples/blob/main/lr_nag.cpp#L239
    cc_params.SetMultiplicativeDepth(mult_depth)
    cc_params.SetScalingModSize(scaling_mod_size)
    cc_params.SetBatchSize(crypto_hparams["batch_size"])
    cc_params.SetSecurityLevel(getattr(openfhe, crypto_hparams["security_level"]))
    cc_params.SetRingDim(crypto_hparams["ring_dimensionality"])
    cc_params.SetScalingTechnique(getattr(openfhe, crypto_hparams["rescale_method"]))
    cc_params.SetKeySwitchTechnique(getattr(openfhe, crypto_hparams["keyswitch_method"]))
    cc_params.SetNumLargeDigits(crypto_hparams["num_large_digits"])
    cc_params.SetFirstModSize(first_mod)
    cc_params.SetMaxRelinSkDeg(crypto_hparams["max_relin_sk_deg"])

    logger.info("Generating CC")
    cc: openfhe.CryptoContext = openfhe.GenCryptoContext(cc_params)
    logger.info("Enabling Crypto Features")
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(openfhe.PKESchemeFeature.ADVANCEDSHE)

    logger.info("Generating Keypair")
    keys: openfhe.KeyPair = cc.KeyGen()
    logger.info("Generating Mult Key and Sum Key")
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    # Based on the number of occupied slots, we do some manipulation of the
    #   data
    num_slots = crypto_hparams["batch_size"]

    return cc, keys, num_slots

