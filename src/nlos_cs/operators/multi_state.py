"""Multi-state sensing operator construction.

A multi-state operator combines several single-state operators that share the
same latent position space. The standard composition is vertical stacking:

    y_tilde = [y^(1); y^(2); ...; y^(K)]
    A_tilde = [A^(1); A^(2); ...; A^(K)]

so that

    y_tilde = A_tilde x + eps_tilde

Each state contributes additional measurements of the same unknown x.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from nlos_cs.operators.single_state import SingleStateOperator

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
OperatorArray = npt.NDArray[np.float64] | npt.NDArray[np.complex128]


@dataclass(frozen=True)
class MultiStateOperator:
    """Composite operator built from multiple single-state operators.

    Attributes
    ----------
    state_ids:
        Ordered list of state identifiers contributing row blocks to A.
    positions_mm:
        Shared latent position grid across all states.
    coords_by_state:
        Mapping from state_id to the ordered 2D probe-plane coordinates used in that block.
    in_plane_axes_by_state:
        Mapping from state_id to the in-plane axes for that block.
    measurement_kind:
        Shared measurement kind across all states, e.g. "e_mag".
    row_slices:
        Mapping from state_id to the row slice occupied by that state's block in A.
    A:
        Composite matrix of shape (sum_k M_k, N).
    """

    state_ids: tuple[str, ...]
    positions_mm: FloatArray
    coords_by_state: dict[str, FloatArray]
    in_plane_axes_by_state: dict[str, tuple[str, str]]
    measurement_kind: str
    row_slices: dict[str, slice]
    A: OperatorArray

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix shape (M_total, N)."""
        return self.A.shape

    @property
    def n_measurements(self) -> int:
        """Total number of measurement rows."""
        return int(self.A.shape[0])

    @property
    def n_positions(self) -> int:
        """Number of latent positions."""
        return int(self.A.shape[1])

    @property
    def n_states(self) -> int:
        """Number of sensing states."""
        return len(self.state_ids)

    @property
    def is_complex(self) -> bool:
        """True if A is complex-valued."""
        return np.iscomplexobj(self.A)

    def block(self, state_id: str) -> OperatorArray:
        """Return the row block for one state."""
        if state_id not in self.row_slices:
            raise KeyError(f"Unknown state_id: {state_id}")
        return self.A[self.row_slices[state_id], :]

    def summary(self) -> dict[str, float | int | str]:
        """Compact summary suitable for logs or manifests."""
        return {
            "n_states": self.n_states,
            "state_ids": ",".join(self.state_ids),
            "measurement_kind": self.measurement_kind,
            "n_measurements": self.n_measurements,
            "n_positions": self.n_positions,
            "is_complex": int(self.is_complex),
        }


def _validate_state_ids(operators: tuple[SingleStateOperator, ...]) -> None:
    state_ids = [op.state_id for op in operators]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError(f"Duplicate state_ids detected: {state_ids}")


def _validate_shared_position_space(operators: tuple[SingleStateOperator, ...]) -> None:
    ref = operators[0].positions_mm
    for op in operators[1:]:
        if ref.shape != op.positions_mm.shape:
            raise ValueError(
                f"Position shape mismatch: {ref.shape} vs {op.positions_mm.shape}"
            )
        if not np.allclose(ref, op.positions_mm, atol=0.0, rtol=0.0):
            raise ValueError(
                f"Position grid mismatch between states '{operators[0].state_id}' "
                f"and '{op.state_id}'"
            )


def _validate_measurement_kind(operators: tuple[SingleStateOperator, ...]) -> None:
    ref = operators[0].measurement_kind
    for op in operators[1:]:
        if op.measurement_kind != ref:
            raise ValueError(
                f"Measurement kind mismatch: '{ref}' vs '{op.measurement_kind}'"
            )


def _validate_dtype_family(operators: tuple[SingleStateOperator, ...]) -> None:
    ref_is_complex = np.iscomplexobj(operators[0].A)
    for op in operators[1:]:
        if np.iscomplexobj(op.A) != ref_is_complex:
            raise ValueError("Cannot combine real and complex operators in one composite A")


def build_multi_state_operator(
    *operators: SingleStateOperator,
) -> MultiStateOperator:
    """Build a multi-state operator by vertically stacking state blocks.

    Parameters
    ----------
    operators:
        One or more compatible single-state operators.

    Returns
    -------
    MultiStateOperator
        Composite operator with shared latent position space.

    Raises
    ------
    ValueError
        If the operators are incompatible.
    """
    if len(operators) == 0:
        raise ValueError("At least one SingleStateOperator is required")

    ops = tuple(operators)

    _validate_state_ids(ops)
    _validate_shared_position_space(ops)
    _validate_measurement_kind(ops)
    _validate_dtype_family(ops)

    is_complex = np.iscomplexobj(ops[0].A)
    A = np.vstack([op.A for op in ops]).astype(np.complex128 if is_complex else np.float64)

    row_slices: dict[str, slice] = {}
    coords_by_state: dict[str, FloatArray] = {}
    in_plane_axes_by_state: dict[str, tuple[str, str]] = {}

    start = 0
    for op in ops:
        stop = start + op.n_measurements
        row_slices[op.state_id] = slice(start, stop)
        coords_by_state[op.state_id] = op.coords_in_plane_mm.copy()
        in_plane_axes_by_state[op.state_id] = tuple(op.in_plane_axes)
        start = stop

    return MultiStateOperator(
        state_ids=tuple(op.state_id for op in ops),
        positions_mm=ops[0].positions_mm.copy(),
        coords_by_state=coords_by_state,
        in_plane_axes_by_state=in_plane_axes_by_state,
        measurement_kind=ops[0].measurement_kind,
        row_slices=row_slices,
        A=A,
    )


def build_weighted_multi_state_operator(
    operators: list[SingleStateOperator],
    weights: list[float],
) -> MultiStateOperator:
    """Build a weighted multi-state operator.

    Each state block is scaled by a scalar weight before stacking:

        A_tilde = [w1 A^(1); w2 A^(2); ...; wK A^(K)]

    This is useful when testing:
    - state-confidence weighting,
    - energy normalisation,
    - synthetic design emphasis.
    """
    if len(operators) == 0:
        raise ValueError("At least one operator is required")
    if len(operators) != len(weights):
        raise ValueError(
            f"Number of operators ({len(operators)}) must match number of weights ({len(weights)})"
        )

    ops = tuple(operators)

    _validate_state_ids(ops)
    _validate_shared_position_space(ops)
    _validate_measurement_kind(ops)
    _validate_dtype_family(ops)

    scaled_ops: list[SingleStateOperator] = []
    for op, w in zip(ops, weights):
        if w < 0:
            raise ValueError(f"State weights must be non-negative, got {w}")

        A_scaled = op.A * w
        scaled_ops.append(
            SingleStateOperator(
                state_id=op.state_id,
                positions_mm=op.positions_mm.copy(),
                coords_in_plane_mm=op.coords_in_plane_mm.copy(),
                in_plane_axes=tuple(op.in_plane_axes),
                measurement_kind=op.measurement_kind,
                A=A_scaled,
            )
        )

    return build_multi_state_operator(*scaled_ops)


def validate_multi_state_operator(operator: MultiStateOperator) -> None:
    """Raise if a MultiStateOperator is internally inconsistent."""
    if operator.A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {operator.A.shape}")

    if operator.positions_mm.ndim != 1:
        raise ValueError("positions_mm must be 1D")

    if len(operator.state_ids) == 0:
        raise ValueError("state_ids must not be empty")

    if set(operator.state_ids) != set(operator.row_slices.keys()):
        raise ValueError("state_ids and row_slices keys do not match")

    if set(operator.state_ids) != set(operator.coords_by_state.keys()):
        raise ValueError("state_ids and coords_by_state keys do not match")

    if set(operator.state_ids) != set(operator.in_plane_axes_by_state.keys()):
        raise ValueError("state_ids and in_plane_axes_by_state keys do not match")

    for state_id in operator.state_ids:
        sl = operator.row_slices[state_id]
        block = operator.A[sl, :]
        coords = operator.coords_by_state[state_id]

        if block.shape[0] != coords.shape[0]:
            raise ValueError(
                f"Row count mismatch for state '{state_id}': "
                f"block has {block.shape[0]} rows, coords have {coords.shape[0]}"
            )
        if block.shape[1] != operator.positions_mm.shape[0]:
            raise ValueError(
                f"Column count mismatch for state '{state_id}': "
                f"block has {block.shape[1]} cols, expected {operator.positions_mm.shape[0]}"
            )


def split_multi_state_operator(operator: MultiStateOperator) -> dict[str, OperatorArray]:
    """Return row blocks keyed by state_id."""
    return {state_id: operator.block(state_id) for state_id in operator.state_ids}


def per_state_row_counts(operator: MultiStateOperator) -> dict[str, int]:
    """Return number of measurement rows contributed by each state."""
    return {
        state_id: operator.block(state_id).shape[0]
        for state_id in operator.state_ids
    }