import numpy as np

from nlos_cs.inverse.base import InverseProblem
from nlos_cs.inverse.huber_simplex import (
    HuberSimplexConfig,
    HuberSimplexSolver,
    huber_gradient_wrt_residual,
    huber_penalty,
    simplex_project,
    solve_huber_simplex,
)


def test_simplex_project_basic_properties():
    x = np.array([-1.0, 2.0, 3.0], dtype=float)
    xp = simplex_project(x)

    assert xp.shape == x.shape
    assert np.all(xp >= 0.0)
    assert np.isclose(np.sum(xp), 1.0)
    assert np.allclose(xp, np.array([0.0, 2.0 / 5.0, 3.0 / 5.0]))


def test_simplex_project_zero_vector_returns_uniform():
    x = np.zeros(4, dtype=float)
    xp = simplex_project(x)

    assert np.all(xp >= 0.0)
    assert np.isclose(np.sum(xp), 1.0)
    assert np.allclose(xp, np.ones(4) / 4.0)


def test_huber_penalty_matches_definition():
    r = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=float)
    delta = 1.0

    out = huber_penalty(r, delta)

    expected = np.array([
        1.0 * (2.0 - 0.5 * 1.0),  # linear regime
        0.5 * (0.5**2),           # quadratic regime
        0.0,
        0.5 * (0.5**2),
        1.0 * (2.0 - 0.5 * 1.0),
    ])
    assert np.allclose(out, expected)


def test_huber_gradient_matches_definition():
    r = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=float)
    delta = 1.0

    out = huber_gradient_wrt_residual(r, delta)
    expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    assert np.allclose(out, expected)


def test_solve_huber_simplex_identity_system_recovers_true_vertex():
    A = np.eye(3)
    y = np.array([0.0, 1.0, 0.0], dtype=float)

    x_hat = solve_huber_simplex(A, y, delta=0.1)

    assert x_hat.shape == (3,)
    assert np.all(x_hat >= -1e-12)
    assert np.isclose(np.sum(x_hat), 1.0)
    assert int(np.argmax(x_hat)) == 1
    assert np.allclose(x_hat, np.array([0.0, 1.0, 0.0]), atol=1e-6)


def test_solve_huber_simplex_respects_warm_start_shape():
    A = np.eye(3)
    y = np.array([1.0, 0.0, 0.0], dtype=float)

    try:
        solve_huber_simplex(A, y, delta=0.1, warm_start=np.array([1.0, 0.0]))
        assert False, "Expected ValueError for bad warm_start shape"
    except ValueError as exc:
        assert "warm_start must have shape" in str(exc)


def test_solve_huber_simplex_rejects_bad_delta():
    A = np.eye(2)
    y = np.array([1.0, 0.0], dtype=float)

    try:
        solve_huber_simplex(A, y, delta=0.0)
        assert False, "Expected ValueError for non-positive delta"
    except ValueError as exc:
        assert "delta must be positive" in str(exc)


def test_solve_huber_simplex_complex_system_returns_real_simplex_solution():
    A = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 1.0j, 1.0 + 0.0j],
        ],
        dtype=complex,
    )
    y = np.array([1.0 + 0.0j, 0.0 + 1.0j], dtype=complex)

    x_hat = solve_huber_simplex(A, y, delta=0.2)

    assert x_hat.dtype == float
    assert x_hat.shape == (2,)
    assert np.all(x_hat >= -1e-12)
    assert np.isclose(np.sum(x_hat), 1.0)


def test_huber_simplex_solver_basic():
    A = np.eye(3)
    y = np.array([1.0, 0.0, 0.0], dtype=float)

    solver = HuberSimplexSolver(HuberSimplexConfig(delta=0.1))
    problem = InverseProblem(A=A, y=y)

    result = solver.solve(problem)

    assert result.solver_name == "huber_simplex"
    assert result.x_hat.shape == (3,)
    assert np.all(result.x_hat >= -1e-12)
    assert np.isclose(np.sum(result.x_hat), 1.0)
    assert int(np.argmax(result.x_hat)) == 0
    assert result.lambda_value is None
    assert result.metadata["simplex_constraint"] is True
    assert result.metadata["l1_term_redundant_under_simplex"] is True


def test_huber_simplex_solver_without_tikhonov_warm_start():
    A = np.eye(3)
    y = np.array([0.0, 0.0, 1.0], dtype=float)

    cfg = HuberSimplexConfig(
        delta=0.1,
        use_tikhonov_warm_start=False,
    )
    solver = HuberSimplexSolver(cfg)
    problem = InverseProblem(A=A, y=y)

    result = solver.solve(problem)

    assert result.solver_name == "huber_simplex"
    assert np.isclose(np.sum(result.x_hat), 1.0)
    assert int(np.argmax(result.x_hat)) == 2
    assert result.metadata["use_tikhonov_warm_start"] is False


def test_huber_simplex_solver_config_rejects_bad_delta():
    try:
        HuberSimplexSolver(HuberSimplexConfig(delta=0.0))
        assert False, "Expected ValueError for non-positive delta"
    except ValueError as exc:
        assert "delta must be positive" in str(exc)