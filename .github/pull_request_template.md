## Description

<!-- Briefly describe the purpose of this PR and what problem it solves. -->

Fixes / Closes #(issue_number)


## Type of Change

- [ ] **New Estimator / Algorithm** (`feat`)
- [ ] **Performance Optimization** (`perf`)
- [ ] **Bug Fix** (`fix`)
- [ ] **Core / Linear Algebra / Kernels** (`core`)
- [ ] **Documentation / Tutorials** (`docs`)
- [ ] **CI / Tooling / Recipe** (`chore`)

---

## API & Parity Checklist (if applicable)

- [ ] **Scikit-Learn Parity**: Standard lifecycle (`fit`, `predict`, `score`), trailing `_` on fitted attributes, input validation.
- [ ] **Traits Implemented**: `Classifier` / `Regressor` / `Clusterer` / `Transformer`, `Copyable`, `Movable`, `Serializable`.
- [ ] **Type & Precision**: Parameterized on `compute_dtype`, generic dtype casting, sparse `CSRMatrix` (where applicable).

---

## Quality Checklist

- [ ] **Formatting**: `pixi run format-check` passes with zero diffs (`pixi run format`).
- [ ] **Compilation**: `pixi run build` compiles with 0 errors and 0 warnings.
- [ ] **Testing**: Unit tests added in `tests/test_*.mojo` & all test suites pass (`pixi run test-runner`).
- [ ] **Documentation**: Diátaxis docstrings with runnable examples & docs build cleanly (`pixi run build-docs`).
- [ ] **Benchmarks**: Tested vs. Scikit-Learn baseline via `benchmarks/runner.py` (if performance-sensitive).

---

## Benchmark Results (if applicable)

<!-- If this PR optimizes kernels or adds an estimator, include key speedup and parity numbers below. -->

| Estimator / Kernel | Phase | Strata Median | Scikit-Learn Median | Speedup | Parity / Quality Metric |
| :--- | :---: | :---: | :---: | :---: | :--- |
| | | | | | |

---

## Additional Context

<!-- Any additional architectural notes, references, or context for reviewers. -->
