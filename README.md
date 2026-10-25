# py_lattice_basis_reduction

Reduces an integer lattice basis using the LLL (Lenstra-Lenstra-Lovász) algorithm.  Pure Python, no dependencies.

```python
from py_lattice_basis_reduction import lll_reduce

# A poorly-conditioned 2-D basis: one vector is a million times the other.
reduced = lll_reduce([[1, 0], [1000000, 1]])
# -> [[0, 1], [1, 0]]
```

## Why this exists

LLL is a standard first step before running exact algorithms on a lattice: shortening the basis makes everything downstream (integer-relation detection, knapsack cryptanalysis, CVP/SVP heuristics) faster and more stable.  The common need is a *dependency-free*, *deterministic* implementation you can paste into a notebook or a CTF script without `pip install`.

The trade-off: this library uses exact rational arithmetic (`fractions.Fraction`) for all Gram-Schmidt quantities, not floating point.  That guarantees the same input always yields the same reduced basis on every machine — no rounding-dependent branches.  The cost is speed: this is fine for bases up to ~40 vectors of ~40 dimensions, but it will not compete with fpLLL on large instances.

## API

`lll_reduce(basis, delta=3/4) -> list[list[int]]`

`basis` is a list of `n` integer vectors, each of length `n` (a square matrix, row-major).  `delta` must be in the open interval `(1/4, 1]`.  Returns a new reduced basis; the input is not mutated.

`LLLReducer(basis, delta)` — the class form.  Call `.reduce()` to run; the result is also stored in `.basis`.

`dot(u, v) -> int` — exact integer dot product, exposed as a convenience.

## Edge cases

* **Non-square or empty input** raises `ValueError`.
* **`delta` outside `(1/4, 1]`** raises `ValueError`.  We reject rather than clamp, because passing `delta=0` almost always means a caller mistake.
* **Linearly dependent basis vectors** are accepted; the reduced output will contain a zero vector in the dependent position.
* All returned entries are Python `int` (never `float`), even though the computation internally uses `Fraction`.
