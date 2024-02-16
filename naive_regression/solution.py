from pprint import pprint

import openfhe
from openfhe import PKESchemeFeature
from typing import Optional, Dict
import yaml
from naive_regression.crypto_utils import generate_rotation_keys
from naive_regression.ematrix import EMatrix
import numpy as np

np.random.seed(42)
from naive_regression.np_reference import train
from naive_regression.regression_code import train as enc_train


def create_crypto(
        num_data_points: int,
        c_params: Dict,
        bootstrap_params: Optional[Dict] = None
):
    ################################################
    # Setting the crypto parameters
    ################################################
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


if __name__ == '__main__':

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    print("ML Config:")
    pprint(config["ml_params"])
    print("Crypto Params:")
    pprint(config["crypto_params"])
    if config["crypto_params"]["run_bootstrap"]:
        print("Running with bootstrap")
        pprint(config["crypto_bootstrap_params"])
    ml_conf = config["ml_params"]
    batch_size = ml_conf["batch_size"]
    lr = ml_conf["lr"]
    epochs = ml_conf["epochs"]

    ################################################
    # Generate data
    ################################################

    X = np.random.rand(batch_size * 5, 5)
    y = (np.dot(X, np.random.rand(5, 1))) + np.random.rand(1)
    noise = np.random.randn(y.shape[0], y.shape[1])
    y = y + noise

    weights = np.random.rand(5, 1)
    print("#" * 10)
    print("Plaintext Performance")
    m_stat = train(X, y, weights, lr, epochs)

    print("#" * 10)
    print("Encrypted Performance")

    create_crypto(
        num_data_points=-1 if config["crypto_params"]["run_bootstrap"] else len(X),
        c_params=config["crypto_params"],
        bootstrap_params=config["crypto_bootstrap_params"]
    )

    enc_train(X, y, weight, lr, epochs, config["crypto_params"]["run_bootstrap"])



    inverse_scale = 1 / len(y)

    ####################################################################
    # We need to repeat the weights N-times bc we do the hadamard product then sum
    #   when we're doing the dot product
    weights = np.squeeze(weights, axis=1).tolist()
    repeated_weights = []
    for i in range(len(X)):
        repeated_weights.append(weights)
    weights = EMatrix.fromList(repeated_weights, packing="vertical", repeated=True)
    weights.encryptSelf()

    ####################################################################
    # We encrypt all at once. NOTE: this is not a true SGD - we're not shuffling
    #   between each epoch. Having said that, this is WAY faster
    e_X = EMatrix.fromList(X.tolist())
    e_y = EMatrix.fromList(y.tolist())
    e_X.encryptSelf()
    e_y.encryptSelf()
    # errors = []
    for epoch in range(n_epochs):
        y_pred, _ = predict("linear", e_X, weights, [])
        residuals, loss = calculate_loss(y_pred, label=e_y, inverse_num_samples_scale=inverse_scale)
        weights, grads = apply_gradient(e_X, weights, residuals, inverse_scale, alpha, len(X))

        ################################################
        # The following isn't realistic - we don't always know the loss. However,
        #   this is primarily for demo purposes
        ################################################
        loss.decryptSelf()
        print(f"epoch: {epoch} ----> MSE: {loss[0]}")

        # Things accumulate noise as we do computations. These following are options to handle the noise:
        #   - bootstrapping, which is expensive
        #   - decrypting and re-encrypting, which comes with its own tradeoffs
        if run_bootstrap_mode:
            weights.bootstrap_self()
        else:
            weights.recrypt_self()
    

