import numpy as np

from nlos_cs.inverse.base import InverseProblem
from nlos_cs.inverse.tikhonov import (
    LCurveTikhonovSolver,
    TikhonovSolver,
    compute_lcurve_sweep,
    make_lambda_grid,
    tikhonov_direct_solve,
    tikhonov_svd_solve,
    compute_svd,
)


def test_tikhonov_svd_matches_direct_real():
    rng = np.random.default_rng(42)

    A = rng.normal(size=(30, 8))
    x_true = rng.normal(size=8)
    y = A @ x_true + 0.01 * rng.normal(size=30)

    lam = 1.25

    U, s, Vh = compute_svd(A)
    x_svd = tikhonov_svd_solve(U, s, Vh, y, lam)
    x_direct = tikhonov_direct_solve(A, y, lam)

    assert np.allclose(x_svd, x_direct, rtol=1e-10, atol=1e-10)


def test_tikhonov_svd_matches_direct_complex():
    rng = np.random.default_rng(123)

    A = rng.normal(size=(20, 6)) + 1j * rng.normal(size=(20, 6))
    x_true = rng.normal(size=6) + 1j * rng.normal(size=6)
    y = A @ x_true + 0.01 * (rng.normal(size=20) + 1j * rng.normal(size=20))

    lam = 0.75

    U, s, Vh = compute_svd(A)
    x_svd = tikhonov_svd_solve(U, s, Vh, y, lam)
    x_direct = tikhonov_direct_solve(A, y, lam)

    assert np.allclose(x_svd, x_direct, rtol=1e-10, atol=1e-10)


def test_fixed_lambda_solver_matches_direct_solution():
    rng = np.random.default_rng(7)

    A = rng.normal(size=(25, 7))
    y = rng.normal(size=25)
    lam = 2.0

    problem = InverseProblem(A=A, y=y)
    result = TikhonovSolver(lambda_value=lam, use_svd=True).solve(problem)
    x_direct = tikhonov_direct_solve(A, y, lam)

    assert np.allclose(result.x_hat, x_direct, rtol=1e-10, atol=1e-10)
    assert result.lambda_value == lam
    assert result.solver_name == "tikhonov"


def test_lcurve_sweep_shapes_and_valid_index():
    rng = np.random.default_rng(99)

    A = rng.normal(size=(40, 10))
    y = rng.normal(size=40)
    lambdas = make_lambda_grid(lambda_min_exp=-2.0, lambda_max_exp=3.0, n_lambda=25)

    sweep = compute_lcurve_sweep(A, y, lambdas)

    assert sweep.lambdas.shape == (25,)
    assert sweep.residual_norms.shape == (25,)
    assert sweep.solution_norms.shape == (25,)
    assert sweep.curvature.shape == (25,)
    assert sweep.x_hats.shape == (25, 10)
    assert 0 <= sweep.idx_opt < 25
    assert sweep.lambda_opt == sweep.lambdas[sweep.idx_opt]


def test_lcurve_solver_returns_lambda_from_grid():
    rng = np.random.default_rng(314)

    A = rng.normal(size=(35, 9))
    y = rng.normal(size=35)
    lambdas = make_lambda_grid(lambda_min_exp=-1.0, lambda_max_exp=2.0, n_lambda=19)

    problem = InverseProblem(A=A, y=y)
    result = LCurveTikhonovSolver(lambdas=lambdas).solve(problem)

    assert result.solver_name == "tikhonov_lcurve"
    assert result.lambda_value in set(lambdas.tolist())
    assert result.metadata["lambdas"].shape == (19,)
    assert result.metadata["residual_norms"].shape == (19,)
    assert result.metadata["solution_norms"].shape == (19,)
    assert result.metadata["curvature"].shape == (19,)


def test_zero_lambda_reduces_to_least_squares_when_full_rank():
    rng = np.random.default_rng(2718)

    A = rng.normal(size=(18, 5))
    x_true = rng.normal(size=5)
    y = A @ x_true

    lam = 0.0

    U, s, Vh = compute_svd(A)
    x_svd = tikhonov_svd_solve(U, s, Vh, y, lam)
    x_direct = tikhonov_direct_solve(A, y, lam)

    assert np.allclose(x_svd, x_true, rtol=1e-10, atol=1e-10)
    assert np.allclose(x_direct, x_true, rtol=1e-10, atol=1e-10)
    assert np.allclose(x_svd, x_direct, rtol=1e-10, atol=1e-10)