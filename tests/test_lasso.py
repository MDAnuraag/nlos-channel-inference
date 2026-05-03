import numpy as np

from nlos_cs.inverse.base import InverseProblem
from nlos_cs.inverse.lasso import (
    LassoSolver,
    LassoSolverConfig,
    lasso_objective,
    soft_threshold,
    solve_lasso_ista,
)


def test_soft_threshold_basic():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=float)
    out = soft_threshold(x, 1.0)

    expected = np.array([-1.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
    assert np.allclose(out, expected)


def test_soft_threshold_zero_threshold():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=float)
    out = soft_threshold(x, 0.0)

    assert np.allclose(out, x)


def test_soft_threshold_rejects_negative_threshold():
    try:
        soft_threshold(np.array([1.0, -1.0]), -0.1)
        assert False, "Expected ValueError for negative threshold"
    except ValueError as exc:
        assert "threshold must be non-negative" in str(exc)


def test_lasso_objective_basic():
    A = np.eye(3)
    y = np.array([1.0, 0.0, 2.0], dtype=float)
    x = np.array([0.5, 0.0, 1.5], dtype=float)
    alpha = 0.2

    obj = lasso_objective(A, y, x, alpha)

    residual = A @ x - y
    expected = 0.5 * np.dot(residual, residual) + alpha * np.sum(np.abs(x))
    assert np.isclose(obj, expected)


def test_solve_lasso_ista_identity_small_alpha_recovers_sparse_vector():
    A = np.eye(3)
    y = np.array([0.0, 2.0, 0.0], dtype=float)

    x_hat, meta = solve_lasso_ista(
        A,
        y,
        alpha=0.1,
        maxiter=1000,
        tol=1e-10,
        use_fista=True,
    )

    # For identity and LASSO, solution is soft-threshold(y, alpha)
    expected = np.array([0.0, 1.9, 0.0], dtype=float)

    assert x_hat.shape == (3,)
    assert np.allclose(x_hat, expected, atol=1e-5)
    assert meta["use_fista"] is True
    assert meta["iterations"] > 0


def test_solve_lasso_ista_large_alpha_shrinks_to_zero():
    A = np.eye(3)
    y = np.array([0.2, 0.5, 0.8], dtype=float)

    x_hat, meta = solve_lasso_ista(
        A,
        y,
        alpha=1.0,
        maxiter=500,
        tol=1e-10,
    )

    assert np.allclose(x_hat, np.zeros(3), atol=1e-8)
    assert meta["iterations"] > 0


def test_solve_lasso_ista_complex_system_returns_real_solution():
    A = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 1.0j, 1.0 + 0.0j],
        ],
        dtype=complex,
    )
    y = np.array([1.0 + 0.0j, 0.0 + 1.0j], dtype=complex)

    x_hat, meta = solve_lasso_ista(
        A,
        y,
        alpha=0.1,
        maxiter=500,
        tol=1e-9,
    )

    assert x_hat.dtype == float
    assert x_hat.shape == (2,)
    assert meta["converged"] in (True, False)


def test_solve_lasso_ista_rejects_bad_alpha():
    A = np.eye(2)
    y = np.array([1.0, 0.0], dtype=float)

    try:
        solve_lasso_ista(A, y, alpha=-0.1)
        assert False, "Expected ValueError for negative alpha"
    except ValueError as exc:
        assert "alpha must be non-negative" in str(exc)


def test_solve_lasso_ista_rejects_bad_x0_shape():
    A = np.eye(3)
    y = np.array([1.0, 0.0, 0.0], dtype=float)

    try:
        solve_lasso_ista(A, y, alpha=0.1, x0=np.array([1.0, 0.0]))
        assert False, "Expected ValueError for bad x0 shape"
    except ValueError as exc:
        assert "x0 must have shape" in str(exc)


def test_lasso_solver_basic():
    A = np.eye(3)
    y = np.array([0.0, 2.0, 0.0], dtype=float)

    solver = LassoSolver(LassoSolverConfig(alpha=0.1, maxiter=1000, tol=1e-10))
    problem = InverseProblem(A=A, y=y)

    result = solver.solve(problem)

    expected = np.array([0.0, 1.9, 0.0], dtype=float)

    assert result.solver_name == "lasso"
    assert result.x_hat.shape == (3,)
    assert np.allclose(result.x_hat, expected, atol=1e-5)
    assert np.isclose(result.lambda_value, 0.1)
    assert result.metadata["real_x_assumption"] is True


def test_lasso_solver_normalise_by_sum():
    A = np.eye(3)
    y = np.array([0.0, 2.0, 0.0], dtype=float)

    solver = LassoSolver(
        LassoSolverConfig(
            alpha=0.1,
            maxiter=1000,
            tol=1e-10,
            normalise_by_sum=True,
        )
    )
    result = solver.solve(InverseProblem(A=A, y=y))

    assert np.isclose(np.sum(np.abs(result.x_hat)), 1.0)
    assert np.allclose(result.x_hat, np.array([0.0, 1.0, 0.0]), atol=1e-6)


def test_lasso_solver_normalise_by_peak():
    A = np.eye(3)
    y = np.array([1.0, 0.0, 4.0], dtype=float)

    solver = LassoSolver(
        LassoSolverConfig(
            alpha=0.5,
            maxiter=1000,
            tol=1e-10,
            normalise_by_peak=True,
        )
    )
    result = solver.solve(InverseProblem(A=A, y=y))

    assert np.isclose(np.max(np.abs(result.x_hat)), 1.0)
    assert result.x_hat[2] == 1.0


def test_lasso_solver_rejects_conflicting_normalisations():
    try:
        LassoSolver(
            LassoSolverConfig(
                alpha=0.1,
                normalise_by_sum=True,
                normalise_by_peak=True,
            )
        )
        assert False, "Expected ValueError for conflicting normalisation flags"
    except ValueError as exc:
        assert "cannot both be True" in str(exc)


def test_lasso_solver_ista_and_fista_both_run():
    A = np.eye(3)
    y = np.array([0.0, 2.0, 0.0], dtype=float)

    solver_ista = LassoSolver(
        LassoSolverConfig(
            alpha=0.1,
            maxiter=1000,
            tol=1e-10,
            use_fista=False,
        )
    )
    solver_fista = LassoSolver(
        LassoSolverConfig(
            alpha=0.1,
            maxiter=1000,
            tol=1e-10,
            use_fista=True,
        )
    )

    result_ista = solver_ista.solve(InverseProblem(A=A, y=y))
    result_fista = solver_fista.solve(InverseProblem(A=A, y=y))

    expected = np.array([0.0, 1.9, 0.0], dtype=float)

    assert np.allclose(result_ista.x_hat, expected, atol=1e-5)
    assert np.allclose(result_fista.x_hat, expected, atol=1e-5)
    assert result_ista.metadata["use_fista"] is False
    assert result_fista.metadata["use_fista"] is True