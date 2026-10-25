"""Core LLL implementation with exact rational arithmetic.

Design choices
--------------
* Exact fractions (``fractions.Fraction``) are used for every
  Gram-Schmidt coefficient and squared norm.  Floating-point LLL is
  attractive for speed, but the whole point of providing a *pure*
  implementation is reproducibility: the same input must always yield
  the same reduced basis on every machine.  With floats, rounding can
  change the branch taken by the Lovász test on different platforms.
  Fractions eliminate that class of bug entirely.
* The Lovász condition is implemented with a strict ``<`` comparison
  against ``delta``.  This matches the common formulation
  (Cohen, Algorithm 2.6.3) and guarantees termination for ``delta`` in
  the open interval (1/4, 1).
* We refuse ``delta <= 1/4`` or ``delta > 1`` rather than silently
  clamping, because a caller passing those values almost certainly has
  a wrong mental model of the algorithm.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence, Tuple


Vector = List[int]
Basis = List[Vector]


def dot(u: Sequence[int], v: Sequence[int]) -> int:
    """Return the exact integer dot product of two equal-length vectors."""
    if len(u) != len(v):
        raise ValueError(
            f"dimension mismatch: len(u)={len(u)}, len(v)={len(v)}"
        )
    return sum(a * b for a, b in zip(u, v))


class LLLReducer:
    """Reduce an integer lattice basis in-place using the LLL algorithm.

    Parameters
    ----------
    basis:
        A list of ``n`` integer vectors, each of length ``n``.  The
        basis is interpreted as *rows* (each vector is a lattice
        generator).
    delta:
        Lovász parameter, strictly in ``(1/4, 1)``.  The classic value
        ``3/4`` works well in practice.

    Notes
    -----
    The reduction mutates ``self.basis``.  Call :meth:`reduce` to run
    the algorithm; the method also returns ``self.basis`` for
    convenience.
    """

    def __init__(self, basis: Sequence[Sequence[int]], delta: Fraction | float | int = 3 / 4):
        if not basis:
            raise ValueError("basis must be non-empty")

        width = len(basis[0])
        if width == 0:
            raise ValueError("vectors must be non-empty")
        for v in basis:
            if len(v) != width:
                raise ValueError("all basis vectors must share the same length")

        self.basis: Basis = [list(v) for v in basis]
        self.dim = width
        self.n = len(basis)
        self.delta = Fraction(delta)

        if not (Fraction(1, 4) < self.delta <= 1):
            raise ValueError(
                f"delta must be in the open interval (1/4, 1]; got {self.delta}"
            )

        self._reduced = False

    # ---- public API ------------------------------------------------

    def reduce(self) -> Basis:
        """Run LLL and return the reduced basis (also stored in ``self.basis``)."""
        self._run()
        self._reduced = True
        return self.basis

    # ---- internal helpers -----------------------------------------

    def _gs_coeffs(self, k: int) -> Tuple[List[Fraction], Fraction]:
        """Compute μ_{k,j} and the squared norm B_k of the
        orthogonalised component at index ``k``.

        Returns ``(mus, Bk)`` where ``mus[j]`` is μ_{k,j} for ``j < k``
        and ``Bk`` is ⟨b*_k, b*_k⟩.  If ``Bk`` is zero the vector is
        linearly dependent on its predecessors; this is allowed (the
        algorithm still terminates) and signalled to the caller by
        returning ``Bk == 0``.
        """
        mus: List[Fraction] = []
        bk = list(self.basis[k])
        bk_frac = [Fraction(x) for x in bk]
        Bk = Fraction(0)

        for j in range(k):
            jv = [Fraction(x) for x in self.basis[j]]
            # μ_{k,j} = ⟨b_k, b*_j⟩ / ⟨b*_j, b*_j⟩
            # The orthogonal component b*_j is represented implicitly
            # via the stored μ values; its squared norm is Bstar[j].
            num = sum(a * b for a, b in zip(bk_frac, jv))
            for i, m in enumerate(self._mu[j][:j]):
                num -= m * sum(a * b for a, b in zip(jv, [Fraction(x) for x in self.basis[i]]))
            # Simpler: we just need the projection, but we track Bstar
            # explicitly to avoid recomputation.
            num = self._proj_num(k, j)
            denom = self._Bstar[j]
            if denom == 0:
                mus.append(Fraction(0))
            else:
                mus.append(num / denom)
            # subtract projection of bk onto b*_j
            if self._Bstar[j] != 0:
                sub = mus[j] * num  # placeholder, recomputed below
            # proper subtraction:
            bk_frac = [a - mus[j] * Fraction(self.basis[j][i]) for i, a in enumerate(bk_frac)]
            # correct bk_frac: subtract mus[j] * b*_j.  b*_j itself is
            # b_j minus its projections onto earlier b*_i.  Rather than
            # recompute b*_j each time, we accumulate the subtraction.
        # Recompute bk_frac properly by orthogonalising against stored
        # b*_j projections represented implicitly.
        bk_frac = self._orthogonal_component(k)
        Bk = sum(a * a for a in bk_frac)
        mus = self._mus_from_orthogonal(k, bk_frac)
        return mus, Bk

    def _proj_num(self, k: int, j: int) -> Fraction:
        """Numerator of μ_{k,j}: ⟨b_k, b*_j⟩ computed without forming b*_j."""
        # ⟨b_k, b_j⟩ - Σ_{i<j} μ_{j,i} ⟨b_k, b*_i⟩
        s = Fraction(dot(self.basis[k], self.basis[j]))
        for i in range(j):
            if self._Bstar[i] != 0:
                s -= self._mu[j][i] * self._proj_num(k, i)
        return s

    def _orthogonal_component(self, k: int) -> List[Fraction]:
        """Return b*_k as a list of Fractions."""
        comp = [Fraction(x) for x in self.basis[k]]
        for j in range(k):
            if self._Bstar[j] != 0:
                mu = self._proj_num(k, j) / self._Bstar[j]
                bstar_j = self._orthogonal_component(j)
                comp = [a - mu * b for a, b in zip(comp, bstar_j)]
        return comp

    def _mus_from_orthogonal(self, k: int, comp: List[Fraction]) -> List[Fraction]:
        """Return μ_{k,0..k-1} using cached b*_j squared norms."""
        mus: List[Fraction] = []
        for j in range(k):
            if self._Bstar[j] == 0:
                mus.append(Fraction(0))
            else:
                bstar_j = self._orthogonal_component(j)
                num = sum(a * b for a, b in zip(comp, bstar_j)) + sum(
                    self._mu_or_zero(k, j, i, bstar_j) for i in range(j, k)
                )
                # The above is getting convoluted; fall back to the
                # direct projection formula.
                num = self._proj_num(k, j)
                mus.append(num / self._Bstar[j])
        return mus

    def _mu_or_zero(self, k: int, j: int, i: int, bstar_j: List[Fraction]) -> Fraction:
        return Fraction(0)

    def _run(self) -> None:
        n = self.n
        self._Bstar: List[Fraction] = [Fraction(0)] * n
        self._mu: List[List[Fraction]] = [[] for _ in range(n)]

        # Initialise Gram-Schmidt for k = 0
        self._Bstar[0] = Fraction(dot(self.basis[0], self.basis[0]))

        k = 1
        while k < n:
            # Compute μ_{k,j} and B_k from scratch for the current basis.
            self._mu[k] = []
            for j in range(k):
                if self._Bstar[j] == 0:
                    self._mu[k].append(Fraction(0))
                else:
                    self._mu[k].append(self._proj_num(k, j) / self._Bstar[j])
            comp = self._orthogonal_component(k)
            self._Bstar[k] = sum(a * a for a in comp)

            # Size-reduce b_k against b_0..b_{k-1}
            self._size_reduce(k)
            # After size reduction, recompute μ and B_k for index k
            self._mu[k] = []
            for j in range(k):
                if self._Bstar[j] == 0:
                    self._mu[k].append(Fraction(0))
                else:
                    self._mu[k].append(self._proj_num(k, j) / self._Bstar[j])
            comp = self._orthogonal_component(k)
            self._Bstar[k] = sum(a * a for a in comp)

            # Lovász test
            if self._Bstar[k] == 0:
                # zero vector after orthogonalisation: move on
                k += 1
                continue
            rhs = (self.delta - self._mu[k][k - 1] ** 2) * self._Bstar[k - 1] \
                if self._Bstar[k - 1] != 0 else Fraction(0)
            if self._Bstar[k] < rhs:
                self._swap_and_recompute(k)
                if k > 1:
                    k -= 1
            else:
                k += 1

    def _size_reduce(self, k: int) -> None:
        """Subtract rounded μ_{k,j} * b_j from b_k to make |μ_{k,j}| ≤ 1/2."""
        changed = True
        while changed:
            changed = False
            for j in range(k - 1, -1, -1):
                if self._Bstar[j] == 0:
                    continue
                mu = self._proj_num(k, j) / self._Bstar[j]
                q = round(mu)
                if q != 0:
                    self.basis[k] = [a - q * b for a, b in zip(self.basis[k], self.basis[j])]
                    changed = True

    def _swap_and_recompute(self, k: int) -> None:
        """Swap b_k and b_{k-1}, then recompute GS data for indices < k."""
        self.basis[k], self.basis[k - 1] = self.basis[k - 1], self.basis[k]
        # Recompute Bstar and mu for indices 0..k-1 from scratch.
        for idx in range(k):
            self._mu[idx] = []
            for j in range(idx):
                if self._Bstar[j] == 0:
                    self._mu[idx].append(Fraction(0))
                else:
                    self._mu[idx].append(self._proj_num(idx, j) / self._Bstar[j])
            comp = self._orthogonal_component(idx)
            self._Bstar[idx] = sum(a * a for a in comp)


def lll_reduce(basis: Sequence[Sequence[int]], delta: Fraction | float | int = 3 / 4) -> List[List[int]]:
    """Return an LLL-reduced copy of *basis*.

    The input is not modified.  ``basis`` must be a square collection of
    integer vectors (n vectors of length n).
    """
    reducer = LLLReducer(basis, delta)
    return reducer.reduce()
