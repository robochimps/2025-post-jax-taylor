# JAX Taylor-mode automatic differentiation examples

This repository contains code examples used in the Medium article:

**“High-order derivatives using JAX Taylor-mode automatic differentiation”**

🔗 [Read the article on Medium »](https://medium.com/@andrey.yachmenev_75311/high-order-derivatives-using-jax-taylor-mode-automatic-differentiation-27c63be6ace9)

## Contents
- [example_nested_vs_taylor.ipynb](example_nested_vs_taylor.ipynb)
  Compares performance of Taylor-mode automatic differentiation using JAX's `jet` module against nested applications of `jax.grad`.
- [example_sin.ipynb](example_sin.ipynb)
  Demonstrates how to compute the Taylor series expansion of simple functions using the `jet` module.
- [example_multivariate.ipynb](example_multivariate.ipynb)
  Shows how to use `jet` for multivariate differentiation.

## Related
- [robochimps/vibrojet](https://github.com/robochimps/vibrojet)