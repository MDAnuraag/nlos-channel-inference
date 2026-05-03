import numpy as np

from nlos_cs.perturb.multipath import (
    add_multipath_leakage,
    add_multipath_leakage_rows,
    build_leakage_contamination,
    choose_leakage_indices,
)


def _make_test_operator() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.2, 0.1],
            [0.3, 2.0, 0.4],
            [0.1, 0.5, 4.0],
            [0.2, 0.1, 0.3],
        ],
        dtype=float,
    )


def test_choose_leakage_indices_basic():
    idx = choose_leakage_indices(
        5,
        exclude_index=2,
        n_leak=2,
        random_seed=42,
    )

    assert idx.shape == (2,)
    assert 2 not in idx
    assert len(set(idx.tolist())) == 2


def test_choose_leakage_indices_reproducible():
    i1 = choose_leakage_indices(6, exclude_index=1, n_leak=3, random_seed=123)
    i2 = choose_leakage_indices(6, exclude_index=1, n_leak=3, random_seed=123)

    assert np.array_equal(i1, i2)


def test_choose_leakage_indices_rejects_bad_inputs():
    try:
        choose_leakage_indices(0, n_leak=1)
        assert False, "Expected ValueError for non-positive n_columns"
    except ValueError as exc:
        assert "n_columns must be positive" in str(exc)

    try:
        choose_leakage_indices(5, exclude_index=10, n_leak=1)
        assert False, "Expected ValueError for bad exclude_index"
    except ValueError as exc:
        assert "out of range" in str(exc)

    try:
        choose_leakage_indices(1, exclude_index=0, n_leak=1)
        assert False, "Expected ValueError when no candidates remain"
    except ValueError as exc:
        assert "No candidate columns remain" in str(exc)


def test_build_leakage_contamination_basic():
    A = _make_test_operator()
    chosen = np.array([1, 2], dtype=np.int64)

    contamination = build_leakage_contamination(A, chosen_indices=chosen)

    expected = A[:, 1] + A[:, 2]
    assert np.allclose(contamination, expected)


def test_build_leakage_contamination_with_weights():
    A = _make_test_operator()
    chosen = np.array([0, 2], dtype=np.int64)
    weights = np.array([2.0, 0.5], dtype=float)

    contamination = build_leakage_contamination(
        A,
        chosen_indices=chosen,
        weights=weights,
    )

    expected = 2.0 * A[:, 0] + 0.5 * A[:, 2]
    assert np.allclose(contamination, expected)


def test_add_multipath_leakage_zero_fraction_returns_identity():
    A = _make_test_operator()
    y = A[:, 0].copy()

    result = add_multipath_leakage(
        y,
        A,
        contamination_fraction_of_rms=0.0,
        exclude_index=0,
        n_leak=2,
        random_seed=42,
    )

    assert np.array_equal(result.y_clean, y)
    assert np.array_equal(result.y_perturbed, y)
    assert np.allclose(result.contamination, np.zeros_like(y))
    assert result.rms_contamination == 0.0


def test_add_multipath_leakage_with_explicit_indices():
    A = _make_test_operator()
    y = A[:, 0].copy()

    result = add_multipath_leakage(
        y,
        A,
        contamination_fraction_of_rms=0.1,
        exclude_index=0,
        chosen_indices=np.array([1, 2], dtype=np.int64),
    )

    assert np.array_equal(result.chosen_indices, np.array([1, 2]))
    assert result.rms_contamination > 0.0
    assert not np.allclose(result.y_perturbed, y)


def test_add_multipath_leakage_hits_target_rms():
    A = _make_test_operator()
    y = A[:, 1].copy()
    frac = 0.15

    result = add_multipath_leakage(
        y,
        A,
        contamination_fraction_of_rms=frac,
        exclude_index=1,
        chosen_indices=np.array([0, 2], dtype=np.int64),
    )

    expected = frac * result.rms_signal
    assert np.isclose(result.rms_contamination, expected, rtol=1e-6, atol=1e-10)


def test_add_multipath_leakage_reproducible():
    A = _make_test_operator()
    y = A[:, 2].copy()

    r1 = add_multipath_leakage(
        y,
        A,
        contamination_fraction_of_rms=0.1,
        exclude_index=2,
        n_leak=2,
        random_seed=99,
    )
    r2 = add_multipath_leakage(
        y,
        A,
        contamination_fraction_of_rms=0.1,
        exclude_index=2,
        n_leak=2,
        random_seed=99,
    )

    assert np.array_equal(r1.chosen_indices, r2.chosen_indices)
    assert np.allclose(r1.contamination, r2.contamination)
    assert np.allclose(r1.y_perturbed, r2.y_perturbed)


def test_add_multipath_leakage_rejects_bad_inputs():
    A = _make_test_operator()
    y = A[:, 0].copy()

    try:
        add_multipath_leakage(
            y,
            A,
            contamination_fraction_of_rms=-0.1,
            exclude_index=0,
        )
        assert False, "Expected ValueError for negative contamination fraction"
    except ValueError as exc:
        assert "must be non-negative" in str(exc)

    try:
        add_multipath_leakage(
            y,
            A,
            contamination_fraction_of_rms=0.1,
            exclude_index=0,
            chosen_indices=np.array([0], dtype=np.int64),
        )
        assert False, "Expected ValueError when chosen_indices includes exclude_index"
    except ValueError as exc:
        assert "includes exclude_index" in str(exc)


def test_add_multipath_leakage_rows_basic():
    A = _make_test_operator()
    Y = A.T.copy()  # rows correspond to measurements generated by each column

    Y_out, results = add_multipath_leakage_rows(
        Y,
        A,
        contamination_fraction_of_rms=0.1,
        exclude_indices=np.array([0, 1, 2], dtype=np.int64),
        n_leak=2,
        random_seed=10,
    )

    assert Y_out.shape == Y.shape
    assert len(results) == Y.shape[0]
    assert results[0].random_seed == 10
    assert results[1].random_seed == 11
    assert results[2].random_seed == 12


def test_add_multipath_leakage_rows_rejects_bad_exclude_shape():
    A = _make_test_operator()
    Y = A.T.copy()

    try:
        add_multipath_leakage_rows(
            Y,
            A,
            contamination_fraction_of_rms=0.1,
            exclude_indices=np.array([0, 1], dtype=np.int64),
        )
        assert False, "Expected ValueError for bad exclude_indices shape"
    except ValueError as exc:
        assert "exclude_indices must have shape" in str(exc)