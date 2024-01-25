# OpenFHE-Tutorials

## Installing

1. Install [OpenFHE-development](https://github.com/openfheorg/openfhe-development)
2. Install [OpenFHE-python](https://github.com/openfheorg/openfhe-python)

## Running the code

1. Read through `runner.py` and make sure you understand what the code is doing
2. Run `runner.py`

## Caveats:

- the code shown below is highly unoptimized and is meant to be used for educational purposes.

## File Structure

`crypto_utils.py`:

- code to handle under-the-hood calculations for setting up the `CryptoContext`, the `PublicKey`, `PrivateKey`, and `RotationKeys`
- if you're curious about how it does all of this, the most relevant function would be `setup`

`ematrix.py`:

- code providing a matrix interface similar to numpy.
- Ematrix works on both encrypted and in-the-clear/plaintext data
- besides providing a familiar interface, it is meant to be hide away many low-level implementation requirements

`ematrix_tests.py`

- tests for the `ematrix.py` class

`reference.py`

- a numpy reference of the code. If everything goes to plan, you will see the exact same results

`regression_code.py`

- code providing a higher-level interface for regression
- the function signatures mimic those in `reference.py`

`runner.py`

- the top-level code for this library

`utils.py`

- misc. utils
