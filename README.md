# OpenFHE-Tutorials

## Installing

1. Install [OpenFHE-development](https://github.com/openfheorg/openfhe-development)
2. Install [OpenFHE-python](https://github.com/openfheorg/openfhe-python)

## Caveats:

- the code shown below is highly unoptimized and is meant to be used for educational purposes.

## Exercises

There are a total of three exercises:

1) implementing a naive linear regression using the code in the [naive_regression](./naive_regression) folder. Work off of
`exe_lin_reg.py` and see `sol_lin_reg.py` for one possible solution.

2) Implementing an optimized logistic regression in the [efficient_regression](./efficient_regression) folder. Work off of
the `exe_log_reg.py` and see `sol_log_reg.py` for a possible implementation.

3) Implementing an optimized Nesterov-accelerated gradient logistic regression in the [efficient_regression](./efficient_regression) folder. Work off of
   the `exe_nag_log_reg.py` and see `sol_nag_log_reg.py` for a possible implementation. You may find it useful to reference
   the plaintext implementation in `nag_logreg_reference.ipynb` which shows how it is implemented in raw numpy.

## Tips

Some tips for working with FHE problems:

1) start with a small-ish ring dimension
2) turn off the security setting (via `HEStd_NotSet`)
3) create a reference numpy implementation
4) Try to do as much as possible in plaintext-space before finally encrypting things
5) ciphertext refreshing speeds up iteration, so start with that for prototyping