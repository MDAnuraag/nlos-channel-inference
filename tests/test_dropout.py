import numpy as np

from nlos_cs.perturb.dropout import (
    add_dropout,
    add_dropout_rows,
    apply_dropout_mask,
    apply_dropout_mask_rows,
    make_dropout_mask,
)


def test_make_dropout_mask_basic():
    mask = make_dropout_mask(10, dropout_fraction=0.2, random_seed=42)

    assert mask.shape == (10,)
    assert mask.dtype == bool


def test_make_dropout_mask_zero_fraction():
    mask = make_dropout_mask(5, dropout_fraction=0.0, random_seed=1)
    assert np.array_equal(mask, np.ones(5, dtype=bool))


def test_make_dropout_mask_full_fraction():
    mask = make_dropout_mask(5, dropout_fraction=1.0, random_seed=1)
    assert np.array_equal(mask, np.zeros(5, dtype=bool))


def test_make_dropout_mask_reproducible():
    m1 = make_dropout_mask(20, dropout_fraction=0.3, random_seed=123)
    m2 = make_dropout_mask(20, dropout_fraction=0.3, random_seed=123)
    assert np.array_equal(m1, m2)


def test_make_dropout_mask_rejects_bad_inputs():
    try:
        make_dropout_mask(0, dropout_fraction=0.2)
        assert False, "Expected ValueError for non-positive n"
    except ValueError as exc:
        assert "n must be positive" in str(exc)

    try:
        make_dropout_mask(10, dropout_fraction=-0.1)
        assert False, "Expected ValueError for negative fraction"
    except ValueError as exc:
        assert "must be in [0, 1]" in str(exc)

    try:
        make_dropout_mask(10, dropout_fraction=1.1)
        assert False, "Expected ValueError for fraction > 1"
    except ValueError as exc:
        assert "must be in [0, 1]" in str(exc)


def test_apply_dropout_mask_basic():
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    mask = np.array([True, False, True, False], dtype=bool)

    result = apply_dropout_mask(y, mask)

    assert np.array_equal(result.y_clean, y)
    assert np.array_equal(result.mask, mask)
    assert np.array_equal(result.y_dropped, np.array([1.0, 0.0, 3.0, 0.0]))
    assert np.isclose(result.dropout_fraction, 0.5)
    assert np.isclose(result.kept_fraction, 0.5)


def test_apply_dropout_mask_accepts_complex():
    y = np.array([1.0 + 1.0j, 2.0 - 1.0j], dtype=complex)
    mask = np.array([False, True], dtype=bool)

    result = apply_dropout_mask(y, mask)

    assert np.array_equal(result.y_dropped, np.array([0.0 + 0.0j, 2.0 - 1.0j]))
    assert np.iscomplexobj(result.y_dropped)


def test_apply_dropout_mask_rejects_shape_mismatch():
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    mask = np.array([True, False], dtype=bool)

    try:
        apply_dropout_mask(y, mask)
        assert False, "Expected ValueError for shape mismatch"
    except ValueError as exc:
        assert "Length mismatch" in str(exc)


def test_add_dropout_basic_and_reproducible():
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    r1 = add_dropout(y, dropout_fraction=0.25, random_seed=99)
    r2 = add_dropout(y, dropout_fraction=0.25, random_seed=99)

    assert np.array_equal(r1.mask, r2.mask)
    assert np.array_equal(r1.y_dropped, r2.y_dropped)
    assert np.array_equal(r1.y_clean, y)
    assert r1.random_seed == 99


def test_apply_dropout_mask_rows_shared_mask():
    Y = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=float,
    )
    mask = np.array([True, False, True], dtype=bool)

    Y_out, results = apply_dropout_mask_rows(Y, mask)

    expected = np.array(
        [
            [1.0, 0.0, 3.0],
            [4.0, 0.0, 6.0],
        ],
        dtype=float,
    )

    assert np.array_equal(Y_out, expected)
    assert len(results) == 2
    for res in results:
        assert np.array_equal(res.mask, mask)


def test_add_dropout_rows_shared_mask():
    Y = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=float,
    )

    Y_out, results = add_dropout_rows(
        Y,
        dropout_fraction=0.5,
        random_seed=10,
        shared_mask=True,
    )

    assert len(results) == 2
    assert np.array_equal(results[0].mask, results[1].mask)
    assert np.array_equal(Y_out[0] == 0.0, Y_out[1] == 0.0)


def test_add_dropout_rows_independent_masks():
    Y = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=float,
    )

    Y_out, results = add_dropout_rows(
        Y,
        dropout_fraction=0.5,
        random_seed=10,
        shared_mask=False,
    )

    assert len(results) == 2
    assert results[0].random_seed == 10
    assert results[1].random_seed == 11


def test_add_dropout_rows_rejects_non_2d_input():
    y = np.array([1.0, 2.0, 3.0], dtype=float)

    try:
        add_dropout_rows(y, dropout_fraction=0.5)
        assert False, "Expected ValueError for non-2D input"
    except ValueError as exc:
        assert "Y must be 2D" in str(exc)