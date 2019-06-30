An Ising-like undirected graphical model
================

This is an educational exercise, to understand how undirected graphical models work.

The model that I have chosen to implement can be described as follows:
There is a set `I` of vertices, and a set of edges between the vertices; we use `E` to denote the set of pairs of vertices that are connected by an edge.
For each vertex `i` in `I`, we have a visible variable `x_i in {-1, +1}` and a target variable `y_i in {-1, +1}`.
The probability distribution for the `y_i`'s given the `x_i`'s is

```
P(y_I | x_I) = exp(a sum_(i in I) y_i x_i + b sum_(i, j in E) y_i y_j)
```

(For example, the vertices could represent spin-half atoms on a lattice; two atoms are connected by an edge if they are neighbours on the lattice.
In this example, the `y_i`'s could represent the spins of the atoms, and the `x_i`'s could represent local magnetic fields.
The analogy would be better if the `x_i`'s were continuous variables, but never mind...)

To make things more interesting, we assume that some but not all of the `y_i`'s are visible.
Thus the set `I` partitions into a disjoint union of two subsets `U` and `L`, which are the "unlabelled" and "labelled" vertices respectively.

The code solves two problems:

**1) Training**. 

Given the values of `x_i` for `i in I` and `y_i` for `i in L`, we wish to find the values of the parameters `a` and `b` that maximise
log-likelihood `L = log P(y_L | x_I )`.

This is solved by gradient descent: the gradients are

```
dL / da = E_obs [ sum_i x_i y_i ] - E_pred [ sum_i x_i y_i ]
```
```
dL / db = E_obs [ sum_(i,j) y_i y_j ] - E_pred [ sum_(i,j) y_i y_j ]
```

Here, the expectation `E_obs` is evaluated with respect to the conditional distribution for the `y_U`'s given the observed values of the `y_L`'s,
while the expectation `E_pred` is evaluated with respect to the joint distribution for all the `y_I`'s.
These expectations can be estimated by Gibbs sampling.

**2) Predicting**.

Given values for `x_i` for `i in I` and `y_i` for `i in L`, and given our postulated values for the parameters `a` and `b`,
we wish to find the marginal probabilities `P(y_i | y_L, x_I)` for individual `i in U`.

This quantity is again computable by Gibbs sampling.

In my implementation, I perform the Gibbs sampling *in parallel*. This means that, when sampling a given variable, the view we have of the neighbouring variables
can potentially be slightly out of date. I have never seen any theoretical analysis of this approximation, but given how many vertices we have relative to the number
of cores on a computer, this seems like a sensible performance optimisation to make.
