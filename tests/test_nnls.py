import numpy as np

from nlos_cs.inverse.base import InverseProblem
from nlos_cs.inverse.nnls import (
    NNLSSolver,
    NNLSSolverConfig,
    solve_nnls,
)


def test_solve_nnls_identity_system():
    A = np.eye(3)
    y = np.array([0.0, 2.0, 0.0], dtype=float)

    x_hat = solve_nnls(A, y)

    assert x_hat.shape == (3,)
    assert np.all(x_hat >= 0.0)
    assert np.allclose(x_hat, np.array([0.0, 2.0, 0.0]))


def test_solve_nnls_clips_negative_solution_by_constraint():
    A = np.eye(3)
    y = np.array([-1.0, 2.0, -3.0], dtype=float)

    x_hat = solve_nnls(A, y)

    assert x_hat.shape == (3,)
    assert np.all(x_hat >= 0.0)
    assert np.allclose(x_hat, np.array([0.0, 2.0, 0.0]))


def test_solve_nnls_complex_system_returns_real_nonnegative_solution():
    A = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 1.0j, 1.0 + 0.0j],
        ],
        dtype=complex,
    )
    y = np.array([1.0 + 0.0j, 0.0 + 1.0j], dtype=complex)

    x_hat = solve_nnls(A, y)

    assert x_hat.dtype == float
    assert x_hat.shape == (2,)
    assert np.all(x_hat >= 0.0)


def test_nnls_solver_basic():
    A = np.eye(3)
    y = np.array([1.0, 0.0, 2.0], dtype=float)

    solver = NNLSSolver()
    problem = InverseProblem(A=A, y=y)

    result = solver.solve(problem)

    assert result.solver_name == "nnls"
    assert result.x_hat.shape == (3,)
    assert np.all(result.x_hat >= 0.0)
    assert np.allclose(result.x_hat, np.array([1.0, 0.0, 2.0]))
    assert result.lambda_value is None
    assert result.metadata["nonnegative_constraint"] is True
    assert result.metadata["sum_constraint"] is False


def test_nnls_solver_normalise_by_sum():
    A = np.eye(3)
    y = np.array([1.0, 0.0, 3.0], dtype=float)

    solver = NNLSSolver(NNLSSolverConfig(normalise_by_sum=True))
    problem = InverseProblem(A=A, y=y)

    result = solver.solve(problem)

    assert np.all(result.x_hat >= 0.0)
    assert np.isclose(np.sum(result.x_hat), 1.0)
    assert np.allclose(result.x_hat, np.array([0.25, 0.0, 0.75]))


def test_nnls_solver_normalise_by_peak():
    A = np.eye(3)
    y = np.array([1.0, 0.0, 4.0], dtype=float)

    solver = NNLSSolver(NNLSSolverConfig(normalise_by_peak=True))
    problem = InverseProblem(A=A, y=y)

    result = solver.solve(problem)

    assert np.all(result.x_hat >= 0.0)
    assert np.isclose(np.max(np.abs(result.x_hat)), 1.0)
    assert np.allclose(result.x_hat, np.array([0.25, 0.0, 1.0]))


def test_nnls_solver_rejects_conflicting_normalisations():
    try:
        NNLSSolver(
            NNLSSolverConfig(
                normalise_by_sum=True,
                normalise_by_peak=True,
            )
        )
        assert False, "Expected ValueError for conflicting normalisation flags"
    except ValueError as exc:
        assert "cannot both be True" in str(exc)


def test_nnls_solver_zero_vector_stays_zero_under_normalisation():
    A = np.eye(3)
    y = np.zeros(3, dtype=float)

    solver_sum = NNLSSolver(NNLSSolverConfig(normalise_by_sum=True))
    solver_peak = NNLSSolver(NNLSSolverConfig(normalise_by_peak=True))

    result_sum = solver_sum.solve(InverseProblem(A=A, y=y))
    result_peak = solver_peak.solve(InverseProblem(A=A, y=y))

    assert np.allclose(result_sum.x_hat, np.zeros(3))
    assert np.allclose(result_peak.x_hat, np.zeros(3))


def test_nnls_solver_on_rectangular_system():
    A = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    y = np.array([1.0, 2.0, 3.0], dtype=float)

    solver = NNLSSolver()
    result = solver.solve(InverseProblem(A=A, y=y))

    assert result.x_hat.shape == (2,)
    assert np.all(result.x_hat >= 0.0)
    assert result.residual_norm >= 0.0
    assert result.objective_value >= 0.0