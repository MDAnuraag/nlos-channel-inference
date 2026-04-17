import numpy as np

from nlos_cs.inverse.base import InverseProblem, ReconstructionResult, InverseSolver
from nlos_cs.metrics.discrimination import (
    compute_discrimination_from_measurements,
    compute_discrimination_from_operator,
    compute_discrimination_from_xhats,
    group_leakage_summary,
    hardest_pairs,
)


class IdentitySolver(InverseSolver):
    name = "identity"

    def solve(self, problem: InverseProblem) -> ReconstructionResult:
        problem.validate()
        x_hat = problem.y.copy()
        return ReconstructionResult(
            x_hat=x_hat,
            residual_norm=0.0,
            solution_norm=float(np.linalg.norm(x_hat)),
            solver_name=self.name,
        )


def test_compute_discrimination_from_xhats_basic():
    x_hats = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.3, 2.0, 0.4],
            [0.1, 0.5, 4.0],
        ],
        dtype=float,
    )

    result = compute_discrimination_from_xhats(x_hats)

    expected_leakage = np.array(
        [
            [0.0, 0.2 / 1.0, 0.1 / 1.0],
            [0.3 / 2.0, 0.0, 0.4 / 2.0],
            [0.1 / 4.0, 0.5 / 4.0, 0.0],
        ]
    )
    expected_discrimination = 1.0 - expected_leakage
    np.fill_diagonal(expected_discrimination, 1.0)

    assert np.allclose(result.leakage, expected_leakage)
    assert np.allclose(result.discrimination, expected_discrimination)
    assert np.array_equal(result.peak_indices, np.array([0, 1, 2]))


def test_compute_discrimination_from_xhats_uses_absolute_values():
    x_hats = np.array(
        [
            [-2.0, 0.5, -0.25],
            [0.2, -4.0, 0.8],
            [0.1, -0.3, -5.0],
        ],
        dtype=float,
    )

    result = compute_discrimination_from_xhats(x_hats, use_abs=True)

    assert np.isclose(result.leakage[0, 1], 0.5 / 2.0)
    assert np.isclose(result.leakage[1, 2], 0.8 / 4.0)
    assert np.isclose(result.leakage[2, 1], 0.3 / 5.0)
    assert np.allclose(np.diag(result.discrimination), 1.0)
    assert np.allclose(np.diag(result.leakage), 0.0)


def test_compute_discrimination_from_xhats_rejects_zero_reference():
    x_hats = np.array(
        [
            [0.0, 1.0],
            [0.2, 1.0],
        ],
        dtype=float,
    )

    try:
        compute_discrimination_from_xhats(x_hats)
        assert False, "Expected ValueError for zero reference value"
    except ValueError as exc:
        assert "Reference value is zero or near-zero" in str(exc)


def test_compute_discrimination_from_operator_with_identity_solver():
    A = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.3, 2.0, 0.4],
            [0.1, 0.5, 4.0],
        ],
        dtype=float,
    )
    positions = np.array([65.0, 70.0, 75.0], dtype=float)

    result = compute_discrimination_from_operator(
        A=A,
        solver=IdentitySolver(),
        positions_mm=positions,
    )

    expected = compute_discrimination_from_xhats(A.T, positions_mm=positions)

    assert np.allclose(result.leakage, expected.leakage)
    assert np.allclose(result.discrimination, expected.discrimination)
    assert np.allclose(result.positions_mm, positions)


def test_compute_discrimination_from_measurements_square_case():
    A = np.eye(3)
    Y = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.3, 2.0, 0.4],
            [0.1, 0.5, 4.0],
        ],
        dtype=float,
    )

    result = compute_discrimination_from_measurements(
        A=A,
        Y=Y,
        solver=IdentitySolver(),
    )

    expected = compute_discrimination_from_xhats(Y)

    assert np.allclose(result.leakage, expected.leakage)
    assert np.allclose(result.discrimination, expected.discrimination)


def test_hardest_pairs_returns_sorted_pairs():
    x_hats = np.array(
        [
            [2.0, 1.0, 0.1],
            [0.2, 2.0, 0.6],
            [0.4, 0.5, 2.0],
        ],
        dtype=float,
    )
    result = compute_discrimination_from_xhats(x_hats)

    pairs = hardest_pairs(result, top_k=3)

    assert len(pairs) == 3
    assert pairs[0][2] >= pairs[1][2] >= pairs[2][2]
    assert pairs[0][:2] == (0, 1)  # leakage = 1.0 / 2.0 = 0.5


def test_group_leakage_summary():
    x_hats = np.array(
        [
            [4.0, 1.0, 0.4, 0.2],
            [0.8, 5.0, 0.5, 0.1],
            [0.2, 0.3, 6.0, 1.2],
            [0.1, 0.2, 0.9, 3.0],
        ],
        dtype=float,
    )
    result = compute_discrimination_from_xhats(x_hats)

    groups = {
        "A": {0, 1},
        "B": {2, 3},
    }
    summary = group_leakage_summary(result, groups)

    assert set(summary.keys()) == {"A->A", "A->B", "B->A", "B->B"}

    a_to_a_expected = np.mean([
        result.leakage[0, 1],
        result.leakage[1, 0],
    ])
    b_to_b_expected = np.mean([
        result.leakage[2, 3],
        result.leakage[3, 2],
    ])

    assert np.isclose(summary["A->A"], a_to_a_expected)
    assert np.isclose(summary["B->B"], b_to_b_expected)


def test_discrimination_result_summary_helpers():
    x_hats = np.array(
        [
            [3.0, 0.6, 0.3],
            [0.5, 4.0, 0.2],
            [0.1, 0.4, 5.0],
        ],
        dtype=float,
    )
    result = compute_discrimination_from_xhats(x_hats)

    i, j, val = result.worst_pair()
    assert (i, j) == (0, 1)
    assert np.isclose(val, 0.6 / 3.0)

    mean_leak = result.mean_off_diagonal_leakage()
    mean_discrim = result.mean_off_diagonal_discrimination()

    assert mean_leak > 0.0
    assert mean_discrim < 1.0
    assert np.isclose(mean_discrim, 1.0 - mean_leak)