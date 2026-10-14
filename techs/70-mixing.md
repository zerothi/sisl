# 70 · Mixing

Owns: `src/sisl/mixing/*`

Paths are relative to `src/sisl/mixing/`.

`sisl.mixing` holds mixers that accelerate a self-consistent loop.
The user runs the loop. The mixers know nothing about Hamiltonians or density matrices.
No other part of sisl imports `sisl.mixing`. `sisl.__getattr__` imports it lazily.

## Classes

| Class | File | Content |
|---|---|---|
| `BaseMixer` | `base.py` | Abstract `__call__(f, df, *args, **kwargs)`. Arithmetic operators. |
| `CompositeMixer` | `base.py` | Two operands and one operator. |
| `BaseWeightMixer` | `base.py` | `weight`. `set_weight` asserts `weight > 0`. |
| `BaseHistoryWeightMixer` | `base.py` | `weight` and a `History`. |
| `StepMixer` | `base.py` | Changes between mixers with generator functions. |
| `History` | `base.py` | A `deque(maxlen=history)` of tuples. |
| `LinearMixer` | `linear.py` | `f + w df`. |
| `AndersonMixer` | `linear.py` | Anderson mixing. |
| `DIISMixer` (`PulayMixer`) | `diis.py` | DIIS / Pulay extrapolation. |
| `AdaptiveDIISMixer` (`AdaptivePulayMixer`) | `diis.py` | DIIS with a weight that changes with the Lagrange multiplier. |

## Call contract

`mixer(f, df, [delta], append=True)` returns the next input.

- `f` is the input of the current step.
- `df = f_out - f_in` is the residual.
- The mixers add `df`. `LinearMixer` returns `f + w * df`. Thus `w` is the fraction of the output.
- `delta` is optional. DIIS uses it in the metric instead of `df`.
- `BaseHistoryWeightMixer.__call__` only appends to the history.
  It stores `(f, df, *args)` without `None` arguments, unless `append=False`.

### Input types

- `LinearMixer` needs `+`, `-` and multiplication by a scalar.
- `DIISMixer` and `AndersonMixer` also need an inner product.
  The default is `a.ravel().conj().dot(b.ravel()).real`. It works on numpy arrays.
- `SparseCSR` has no `ravel`. For sparse matrices, give `DIISMixer` a `metric=`.
  `AndersonMixer` has no `metric` argument.
- `History` stores references, not copies.
  Do not change `f` in place after a call. That changes the history too.

## History

- When the deque is full, the oldest entry drops.
- Entries can have different lengths.
- An `int` as `history` makes a new `History`. A `History` object is used as it is. Thus mixers can share one.
- `clear(index)` deletes the entries in reverse order.

## Composite mixers

`+`, `-`, `*`, `/`, `**` and unary `-` make a `CompositeMixer(op, A, B)`.

- It calls each mixer operand with the same `(f, df, ...)`. Then it applies `op`.
- Each mixer appends to its own history. A mixer on both sides appends two times.
- The user makes the weights sum to 1. Example: `0.1 * AndersonMixer() + 0.9 * LinearMixer()`.

## StepMixer

`StepMixer(*gen_funcs)` chains generator functions that yield mixers.

- `__init__` gets the first mixer at once.
- `next()` starts the generators again when they are empty. Thus the cycle repeats.
- `__getattr__` sends unknown attributes to the current mixer.
- A generator can do work between yields, for example `history.clear()` to restart.

## DIIS

`DIISMixer.solve_lagrange`:

1. Fill the symmetric matrix `B[i, j] = metric(e_i, e_j)`. `e_i` is the last element of history entry `i` (`delta` or `df`).
2. Set the constraint row and column to `scale = (max + min) / 2` of the `B` block, not to 1.
   Reason: better conditioning (0.16.4).
3. Solve with `solve_destroy(..., assume_a="sym")` (LAPACK `sysv`, symmetric indefinite).
4. Return `c * scale` (the sum is 1) and the Lagrange multiplier `-c[-1] * scale**2`.

- With no history: `c = [1]`, Lagrange multiplier `100`.
- With one entry: `c = [1]`, Lagrange multiplier `metric(e, e)`.
- `mix` returns `Σ c_i (f_i + w df_i)`.

`AdaptiveDIISMixer` has `weight=(min, max)`. A scalar `w` gives `(max(0.001, 0.1 w), min(1, 2 w))`.
Each step: `w = min + (max - min) / (exp((ln λ + 13) / 7) + 1)`.
A large Lagrange multiplier `λ` gives `min`. `λ → 0` gives `max`.

## Anderson

- The history holds `(f, f + df, delta)`.
- `β = <d2 | d2 - d1> / |d2 - d1|²`.
- It returns `(1 - w) n_in + w n_out`. The first step is linear.
- The default is `history=1`. Only the last entry is used.
- When `|d2 - d1|² = 0`, `β = 0`. Thus the step is linear and there is no division by zero.

## Add a mixer

1. Subclass `BaseHistoryWeightMixer`.
2. Set a default `history > 0` in `__init__`.
3. In `__call__`, call `super().__call__(f, df, *extras, append=append)`. Then use `self.history` and `self.weight`.
4. Add it to `__all__`. Use `@set_module("sisl.mixing")`. List it in `docs/api/mixing.rst`.
5. Add a test in `tests/` with `pytestmark = pytest.mark.mixing`.

The tests iterate `f = cos(f)` on `linspace(0, 7, 1000)` until `max|df|` is small.
Most tests check only that the loop converges. `test_anderson_mixer_second_step` checks one Anderson step against the equation. There are no tests for sparse input, `metric=`, `delta` or a singular `B`.

## Known defects

Confirmed by running or reading the code (2026-10-14).

- When the DIIS solve fails, `c = [1]` mixes `history[0]`, the oldest entry. The comment says "the last entry".
- A call with an empty history (`append=False` on a new mixer, or `history=0`) raises
  `TypeError: reduce() of empty iterable`.
