# type: ignore
"""Warp BSR assembly and cuDSS solves for the direct_pgo factor equations.

Each keyframe has an interleaved [phi(3), v(3)] block: left SO(3)
perturbations, body-frame velocity. H[k,k+1] = H[k+1,k]^T couples adjacent
keyframes, yielding a block-tridiagonal Hessian with O(N) factorization.
B windows share one BSR assembly matrix, then pack into B scalar CSR systems.

Assembly uses fp32. GN factorization uses fp64: anchor/prior weights near 1e6
can put cond(H) near 1e7 and erase fp32 step accuracy. Sparse ridge regularization differs from dense eigenvalue flooring.

Patterns and solver plans are cached per (B, N, device). Outputs are detached
pseudo-labels. CUDA, nvmath-python and nvidia-cudss are required; inputs/state
use fp32 because the Warp block type is fixed.
"""

from __future__ import annotations

import glob
import logging
import os
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import pypose as pp
import jaxtyping as jt
import numpy as np

import warp as wp
import warp.sparse as ws

from Interface.graph import I_IMUFactorGraph
from .outlier_gate import mahalanobis_d2, rejection_mask


wp.init()

# A dedicated quiet logger avoids per-iteration cuDSS messages on the root logger.
_cudss_logger = logging.getLogger("sparse_pgo.cudss")
_cudss_logger.setLevel(logging.WARNING)


def _find_cudss_mtlayer() -> str | None:
    """Find cuDSS's host-planning multithreading library in wheel or system paths."""
    try:
        import nvidia  # namespace package of the nvidia-* wheels
        roots = list(getattr(nvidia, "__path__", []))
    except ImportError:
        roots = []
    for root in roots:
        hits = sorted(glob.glob(os.path.join(root, "**", "libcudss_mtlayer_gomp.so*"), recursive=True))
        if hits:
            return hits[0]
    for candidate in (
        "/etc/alternatives/libcudss_mtlayer_gomp.so.0",
        "/usr/lib/x86_64-linux-gnu/libcudss_mtlayer_gomp.so.0",
        "/usr/lib/aarch64-linux-gnu/libcudss_mtlayer_gomp.so.0",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


_CUDSS_MTLAYER = _find_cudss_mtlayer()


def _csr_tensor_quiet(*args, **kwargs) -> torch.Tensor:
    """Build CSR from the precomputed pattern without beta/invariant notices."""
    torch.sparse.check_sparse_tensor_invariants(False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
        warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")
        return torch.sparse_csr_tensor(*args, **kwargs)


_EPS_COV = 1e-10  # diagonal ridge for covariance inversion

# Cap on chi-square rejection/re-solve rounds (matches direct_pgo).
_MAX_REJECT_PASSES = 5


mat66_t = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
mat33_t = wp.types.matrix(shape=(3, 3), dtype=wp.float32)
vec6_t  = wp.types.vector(length=6, dtype=wp.float32)


def _safe_cov_inv(cov: torch.Tensor) -> torch.Tensor:
    """Symmetrize and invert with an _EPS_COV diagonal ridge."""
    eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
    cov = 0.5 * (cov + cov.transpose(-1, -2)) + _EPS_COV * eye
    return torch.linalg.inv(cov)


# Cache per (B, N, device): eager BSR pattern and lazy fp64 step solver. Shared
# value/RHS buffers are mutated in place; callers must use the bundle
# synchronously from one Python thread.

# Diagonal ridge for numerically indefinite fp32 Hessians. Unlike dense
# eigenvalue flooring, this shifts every eigenvalue by the same amount.
_CUDSS_RIDGE = 1.0e-6


@dataclass
class _PatternResources:
    """Immutable BSR pattern shared by graphs with the same (B, N, device)."""
    offsets_wp:     "wp.array"    # int32, length B*N+1   — block-row pointers
    columns_wp:     "wp.array"    # int32, length nnz     — block-col indices
    nnz_idx:        torch.Tensor  # (B, N, 3) int32       — -1 where slot absent
    nnz:            int           # total non-zero blocks  — = B * (3*N - 2)
    M_blocks:       int           # block-row count        — = B * N


@dataclass
class _CudssStepResources:
    """Lazy GN solver for all B windows; refresh bound values/RHS without reallocating."""
    csr_indptr_t:  torch.Tensor   # int32 — CSR indptr for one window
    csr_indices_t: torch.Tensor   # int32 — CSR col-indices for one window
    values_buf:    torch.Tensor   # (B, total_nnz) fp64 — MUTATED per step
    wp_values:     "wp.array"     # Warp view of values_buf
    rhs_storage:   torch.Tensor   # (B, 1, 6N) fp64 row-major — MUTATED per step
    rhs:           torch.Tensor   # (B, 6N, 1) col-major view of rhs_storage (what cuDSS sees)
    A_batched:     torch.Tensor   # 3D sparse CSR wrapping values_buf
    solver:        Any            # nvmath DirectSolver, already planned


@dataclass
class _GraphResources:
    pattern: _PatternResources
    step:    _CudssStepResources | None = None  # built lazily on first step()


_GRAPH_RESOURCES: dict[tuple[int, int, str], _GraphResources] = {}


def _build_block_tridiag_pattern(B: int, N: int, device: torch.device):
    """Emit block-tridiagonal COO entries in lower/diag/upper column order.

    Sorted emission makes post-sort nnz positions predictable without BSR readback.
    Return (rows, cols, nnz_idx); missing boundary slots in nnz_idx are -1.
    """
    rows: list[int] = []
    cols: list[int] = []
    nnz_idx = np.full((B, N, 3), -1, dtype=np.int32)
    e = 0
    for b in range(B):
        for k in range(N):
            r = b * N + k
            if k >= 1:
                rows.append(r); cols.append(r - 1); nnz_idx[b, k, 1] = e; e += 1
            rows.append(r); cols.append(r); nnz_idx[b, k, 0] = e; e += 1
            if k <= N - 2:
                rows.append(r); cols.append(r + 1); nnz_idx[b, k, 2] = e; e += 1
    rows_t = torch.tensor(rows, dtype=torch.int32, device=device)
    cols_t = torch.tensor(cols, dtype=torch.int32, device=device)
    nnz_idx_t = torch.from_numpy(nnz_idx).to(device)
    return rows_t, cols_t, nnz_idx_t


def _build_pattern(B: int, N: int, device: torch.device,
                   wp_device: str) -> _PatternResources:
    rows_t, cols_t, nnz_idx = _build_block_tridiag_pattern(B, N, device)
    nnz = rows_t.numel()
    M_blocks = B * N

    # Use Warp to build row pointers; only offsets/columns are shared.
    # Each graph allocates its own values.
    rows_wp_tmp = wp.from_torch(rows_t, dtype=wp.int32)
    cols_wp_tmp = wp.from_torch(cols_t, dtype=wp.int32)
    vals_tmp    = wp.zeros(nnz, dtype=mat66_t, device=wp_device)
    A_tmp = ws.bsr_from_triplets(
        M_blocks, M_blocks, rows_wp_tmp, cols_wp_tmp, vals_tmp,
        prune_numerical_zeros=False,
    )
    # Copy pattern arrays so the temporary BSR can be released.
    offsets_wp = wp.array(A_tmp.offsets.numpy(), dtype=wp.int32, device=wp_device)
    columns_wp = wp.array(A_tmp.columns.numpy(), dtype=wp.int32, device=wp_device)
    assert A_tmp.nnz == nnz, f"nnz mismatch: {A_tmp.nnz} vs {nnz}"

    return _PatternResources(
        offsets_wp=offsets_wp,
        columns_wp=columns_wp,
        nnz_idx=nnz_idx,
        nnz=nnz,
        M_blocks=M_blocks,
    )


def _make_csr_structure(N: int, device: torch.device, blk: int = 6):
    """int32 CSR (indptr, indices, nnz) of a (blk*N)^2 block-tridiagonal matrix: row blk*k+i spans cols from blk*max(k-1,0)."""
    k = np.arange(N)
    per_row = blk * (1 + (k >= 1) + (k <= N - 2))                          # (N,)
    row_len = np.repeat(per_row, blk)                                      # (blk*N,)
    indptr = np.zeros(blk * N + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(row_len)
    total_nnz = int(indptr[-1])
    row_start_col = np.repeat(blk * np.maximum(k - 1, 0), blk)             # (blk*N,)
    offs = np.arange(total_nnz) - np.repeat(indptr[:-1], row_len)          # position within row
    indices = (np.repeat(row_start_col, row_len) + offs).astype(np.int32)
    return (
        torch.from_numpy(indptr.astype(np.int32)).to(device),
        torch.from_numpy(indices).to(device),
        total_nnz,
    )


def _import_nvmath():
    try:
        from nvmath.sparse.advanced import (
            DirectSolver, DirectSolverOptions, DirectSolverMatrixType,
        )
    except ImportError as e:
        raise ImportError(
            "Sparse_IMUPGOGraph requires nvmath-python + nvidia-cudss for its "
            "direct solver. Install with: pip install nvmath-python[cu13] "
            "nvidia-cudss-cu13  (swap to cu12 for CUDA 12.x). See "
            "requirements.txt. Dense_IMUPGOGraph needs neither."
        ) from e
    return DirectSolver, DirectSolverOptions, DirectSolverMatrixType


def _build_cudss_step(B: int, N: int, device: torch.device,
                      wp_device: str) -> _CudssStepResources:
    """Build the fp64 GN solver with O(B*N) storage.

    Transpose row-major RHS storage so cuDSS's operand has unit stride on dim -2.
    """
    DirectSolver, DirectSolverOptions, DirectSolverMatrixType = _import_nvmath()

    csr_indptr_t, csr_indices_t, total_nnz = _make_csr_structure(N, device)
    values_buf  = torch.zeros((B, total_nnz), dtype=torch.float64, device=device)
    wp_values   = wp.from_torch(values_buf, dtype=wp.float64)
    rhs_storage = torch.zeros((B, 1, 6 * N), dtype=torch.float64, device=device)
    rhs         = rhs_storage.transpose(-1, -2)
    assert rhs.stride(-2) == 1

    A_batched = _csr_tensor_quiet(
        csr_indptr_t.unsqueeze(0).expand(B, -1).contiguous(),
        csr_indices_t.unsqueeze(0).expand(B, -1).contiguous(),
        values_buf,
        size=(B, 6 * N, 6 * N),
    )
    # SYMMETRIC → LDL^T: H + ridge*I is SPD in exact arithmetic but can be
    # numerically indefinite on stationary segments; LDL^T tolerates that where
    # a strict Cholesky would fail the whole batch.
    options = DirectSolverOptions(
        sparse_system_type=DirectSolverMatrixType.SYMMETRIC,
        multithreading_lib=_CUDSS_MTLAYER,
        logger=_cudss_logger,
    )
    solver = DirectSolver(A_batched, rhs, options=options)
    solver.plan()
    return _CudssStepResources(
        csr_indptr_t=csr_indptr_t,
        csr_indices_t=csr_indices_t,
        values_buf=values_buf,
        wp_values=wp_values,
        rhs_storage=rhs_storage,
        rhs=rhs,
        A_batched=A_batched,
        solver=solver,
    )


# Conventions — match Dense_IMUPGOGraph:
#   state layout per KF: [phi (3), v (3)]  with rot stored as quaternion
#   r_imu[e] = [Log(dR^-1 R_e^T R_{e+1}); R_e^T R_{e+1} v_{e+1} - v_e - dv]
#   r_obs[k] = v_k - net_vel[k]
#   r_anc    = Log(R_anchor^-1 R_0)
#   r_vp     = v_0 - vel_prior         (optional initial velocity prior on KF 0)
#   left perturbation: R_k <- Exp(delta_phi) @ R_k
#
# Residual layout in the flat (B, D) array: same order as Dense_IMUPGOGraph
#   D = 6*(N-1) + 3*N + 3 [+ 3 when vel_prior given]
#   IMU rows   [0, 6(N-1))   per-edge stride 6: [r_R(3), r_v(3)]
#   Obs rows   [6(N-1), 6(N-1)+3N)
#   Anchor     [6(N-1)+3N, 6(N-1)+3N+3)
#   Vel-prior  [6(N-1)+3N+3, D)   — present only when vel_prior is supplied
#
# Warp and PyPose quaternions both use [x, y, z, w] (real-last).


@wp.func
def _quat_log(q: wp.quat) -> wp.vec3:
    """Unit quaternion to so(3) axis-angle vector, matching PyPose's Log convention."""
    axis = wp.vec3(0.0, 0.0, 0.0)
    angle = float(0.0)
    wp.quat_to_axis_angle(q, axis, angle)
    return axis * angle


@wp.func
def _so3_jr_inv(phi: wp.vec3) -> mat33_t:
    """Inverse of the right Jacobian of SO(3) at tangent ``phi``.

    J_r^-1(phi) = I + 0.5 [phi]_x + c2 * [phi]_x^2,
        where c2 = 1/theta^2 - 0.5*cot(theta/2)/theta  (= 1/12 + theta^2/720 + ...).

    wp.select evaluates both branches to avoid divergent small-angle guards.
    """
    px = phi[0]
    py = phi[1]
    pz = phi[2]
    theta2 = px * px + py * py + pz * pz

    Phi = mat33_t(
        0.0, -pz,   py,
         pz,  0.0, -px,
        -py,  px,   0.0,
    )
    Phi2 = Phi @ Phi

    # c2_far  = 1/theta^2 - cot(theta/2) / (2*theta).
    # c2_near = 1/12 + theta^2/720 (next-order Taylor). theta^4/30240 dropped.
    # The Taylor branch is exact to machine precision below theta2 ~ 1e-10.
    theta = wp.sqrt(wp.max(theta2, 1.0e-30))     # guard sqrt(0) without branching
    half_cot = 0.5 / wp.tan(0.5 * theta)
    c2_far  = 1.0 / wp.max(theta2, 1.0e-30) - half_cot / theta
    c2_near = (1.0 / 12.0) + theta2 * (1.0 / 720.0)
    c2 = wp.where(theta2 < 1.0e-8, c2_near, c2_far)

    return mat33_t(
        1.0 + 0.5 * Phi[0,0] + c2 * Phi2[0,0],
              0.5 * Phi[0,1] + c2 * Phi2[0,1],
              0.5 * Phi[0,2] + c2 * Phi2[0,2],
              0.5 * Phi[1,0] + c2 * Phi2[1,0],
        1.0 + 0.5 * Phi[1,1] + c2 * Phi2[1,1],
              0.5 * Phi[1,2] + c2 * Phi2[1,2],
              0.5 * Phi[2,0] + c2 * Phi2[2,0],
              0.5 * Phi[2,1] + c2 * Phi2[2,1],
        1.0 + 0.5 * Phi[2,2] + c2 * Phi2[2,2],
    )


@wp.func
def _exp_so3(phi: wp.vec3) -> wp.quat:
    """SO(3) Exp: tangent so(3) vector to unit quaternion via axis-angle."""
    theta = wp.length(phi)
    if theta < 1.0e-7:
        return wp.quat_identity()
    return wp.quat_from_axis_angle(phi / theta, theta)


@wp.kernel
def _residual_imu_kernel(
    rot:        wp.array2d(dtype=wp.quat),     # (B, N)
    vel:        wp.array2d(dtype=wp.vec3),     # (B, N)
    imu_dr:     wp.array2d(dtype=wp.vec3),     # (B, N)  slot N-1 unused
    imu_dv:     wp.array2d(dtype=wp.vec3),     # (B, N)
    r_out:      wp.array3d(dtype=wp.float32),  # (B, N-1, 6)
):
    b, e = wp.tid()                  # IMU edge e: KF e -> KF e+1
    Ri = rot[b, e]
    Rj = rot[b, e + 1]
    vi = vel[b, e]
    vj = vel[b, e + 1]
    dr = imu_dr[b, e]
    dv = imu_dv[b, e]

    dR = _exp_so3(dr)
    # r_R = Log(dR^-1 * R_i^-1 * R_j)
    q = wp.quat_inverse(dR) * wp.quat_inverse(Ri) * Rj
    r_R = _quat_log(q)

    # r_v = R_i^T R_j v_j - v_i - dv  (in B_i frame)
    Ri_T = wp.transpose(wp.quat_to_matrix(Ri))
    Rj_mat = wp.quat_to_matrix(Rj)
    RiT_Rj = Ri_T @ Rj_mat
    r_v = RiT_Rj @ vj - vi - dv

    r_out[b, e, 0] = r_R[0]
    r_out[b, e, 1] = r_R[1]
    r_out[b, e, 2] = r_R[2]
    r_out[b, e, 3] = r_v[0]
    r_out[b, e, 4] = r_v[1]
    r_out[b, e, 5] = r_v[2]


@wp.kernel
def _residual_obs_kernel(
    vel:        wp.array2d(dtype=wp.vec3),     # (B, N)
    net_vel:    wp.array2d(dtype=wp.vec3),     # (B, N)
    r_out:      wp.array3d(dtype=wp.float32),  # (B, N, 3)
):
    b, k = wp.tid()
    diff = vel[b, k] - net_vel[b, k]
    r_out[b, k, 0] = diff[0]
    r_out[b, k, 1] = diff[1]
    r_out[b, k, 2] = diff[2]


@wp.kernel
def _residual_anc_kernel(
    rot:        wp.array2d(dtype=wp.quat),     # (B, N)
    rot_anc:    wp.array(dtype=wp.quat),       # (B,)
    r_out:      wp.array2d(dtype=wp.float32),  # (B, 3)
):
    b = wp.tid()
    R0 = rot[b, 0]
    Ranc = rot_anc[b]
    q = wp.quat_inverse(Ranc) * R0
    r = _quat_log(q)
    r_out[b, 0] = r[0]
    r_out[b, 1] = r[1]
    r_out[b, 2] = r[2]


@wp.kernel
def _residual_vp_kernel(
    vel:        wp.array2d(dtype=wp.vec3),     # (B, N) — only [b, 0] read
    vel_prior:  wp.array(dtype=wp.vec3),       # (B,)
    r_out:      wp.array2d(dtype=wp.float32),  # (B, 3)
):
    """Body-frame initial-velocity prior residual at KF 0: r_vp = v_0 - vel_prior."""
    b = wp.tid()
    d = vel[b, 0] - vel_prior[b]
    r_out[b, 0] = d[0]
    r_out[b, 1] = d[1]
    r_out[b, 2] = d[2]


@wp.kernel
def _zero_bsr_values(values: wp.array(dtype=mat66_t)):
    i = wp.tid()
    values[i] = mat66_t(0.0)


@wp.kernel
def _apply_step_kernel(
    dx:    wp.array3d(dtype=wp.float32),     # (B, N, 6) — interleaved [phi(3), v(3)]
    rot:   wp.array2d(dtype=wp.quat),        # (B, N) — updated in place: R <- Exp(phi) * R
    vel:   wp.array2d(dtype=wp.vec3),        # (B, N) — updated in place: v += delta_v
):
    """Update rotation by left perturbation and body-frame velocity additively, in place."""
    b, k = wp.tid()
    phi = wp.vec3(dx[b, k, 0], dx[b, k, 1], dx[b, k, 2])
    dv  = wp.vec3(dx[b, k, 3], dx[b, k, 4], dx[b, k, 5])
    rot[b, k] = _exp_so3(phi) * rot[b, k]
    vel[b, k] = vel[b, k] + dv


@wp.kernel
def _rot_chain_kernel(
    anchor_quat: wp.array(dtype=wp.quat),     # (B,) anchor R_wb at KF 0
    imu_dr:      wp.array2d(dtype=wp.vec3),   # (B, N) per-segment so(3) tangent
    rot_out:     wp.array2d(dtype=wp.quat),   # (B, N) chained rotations
    N:           wp.int32,
):
    """Chain anchor_quat through imu_dr[:, :N-1] sequentially per batch.
        R_0 = anchor;  R_{k+1} = R_k * Exp(imu_dr[k])

    One thread per batch element — N-1 sequential composes inside the thread.
    """
    b = wp.tid()
    R = anchor_quat[b]
    rot_out[b, 0] = R
    for k in range(N - 1):
        dq = _exp_so3(imu_dr[b, k])
        R = R * dq
        rot_out[b, k + 1] = R


@wp.func
def _accum_block(
    H: wp.array(dtype=mat66_t),
    nnz_idx: wp.int32,
    block: mat66_t,
):
    if nnz_idx >= 0:
        wp.atomic_add(H, nnz_idx, block)


@wp.kernel
def _assemble_obs_kernel(
    W_obs:        wp.array4d(dtype=wp.float32),   # (B, N, 3, 3) inverse cov
    nnz_diag:     wp.array2d(dtype=wp.int32),     # (B, N)
    H_values:     wp.array(dtype=mat66_t),
    g:            wp.array2d(dtype=wp.float32),   # (B, 6N)
    r_obs:        wp.array3d(dtype=wp.float32),   # (B, N, 3)
):
    """Per (b, k): velocity-observation contribution to H[k,k] and g.

    J_obs = [0_3x3, I_3]  (3x6 row stack);  Σ_obs^{-1} = W_obs (3x3)
        H block (6x6): lower-right 3x3 = W_obs, top-left 3x3 = 0
        g vel rows (3): W_obs @ r_obs[k]
    """
    b, k = wp.tid()

    Wv = mat33_t(
        W_obs[b, k, 0, 0], W_obs[b, k, 0, 1], W_obs[b, k, 0, 2],
        W_obs[b, k, 1, 0], W_obs[b, k, 1, 1], W_obs[b, k, 1, 2],
        W_obs[b, k, 2, 0], W_obs[b, k, 2, 1], W_obs[b, k, 2, 2],
    )
    zero3 = mat33_t(0.0)
    obs_block = _pack_mat66_from_blocks(zero3, zero3, zero3, Wv)
    _accum_block(H_values, nnz_diag[b, k], obs_block)

    rv = wp.vec3(r_obs[b, k, 0], r_obs[b, k, 1], r_obs[b, k, 2])
    Wv_r = Wv @ rv
    base = 6 * k
    wp.atomic_add(g, b, base + 3, Wv_r[0])
    wp.atomic_add(g, b, base + 4, Wv_r[1])
    wp.atomic_add(g, b, base + 5, Wv_r[2])


@wp.kernel
def _assemble_anc_kernel(
    rot:          wp.array2d(dtype=wp.quat),      # (B, N) — only [b, 0] is read
    anchor_w:     wp.float32,
    nnz_diag:     wp.array2d(dtype=wp.int32),     # (B, N) — only [b, 0] is read
    H_values:     wp.array(dtype=mat66_t),
    g:            wp.array2d(dtype=wp.float32),   # (B, 6N)
    r_anc:        wp.array2d(dtype=wp.float32),   # (B, 3)
):
    """Per b: rotation anchor contribution to H[0,0] and g (KF 0 of each batch).

    J_anc = Jr_inv(r_anc) @ R_0^T   (3x3, on phi columns of KF 0)
        H phi-block += anchor_w * J^T J
        g phi-rows  += anchor_w * J^T r_anc

    Separate from the obs kernel so this launches B threads with no branch,
    rather than B*N threads with a sparse `if k == 0`.
    """
    b = wp.tid()
    ra = wp.vec3(r_anc[b, 0], r_anc[b, 1], r_anc[b, 2])
    Jr_inv = _so3_jr_inv(ra)
    R0_T = wp.transpose(wp.quat_to_matrix(rot[b, 0]))
    J = Jr_inv @ R0_T
    JT = wp.transpose(J)

    JTJ = JT @ J
    # H phi-block (6x6): top-left 3x3 = w * J^T J, others zero.
    zero3 = mat33_t(0.0)
    anc_block = _pack_mat66_from_blocks(anchor_w * JTJ, zero3, zero3, zero3)
    _accum_block(H_values, nnz_diag[b, 0], anc_block)

    Jr_anc = JT @ ra
    wp.atomic_add(g, b, 0, anchor_w * Jr_anc[0])
    wp.atomic_add(g, b, 1, anchor_w * Jr_anc[1])
    wp.atomic_add(g, b, 2, anchor_w * Jr_anc[2])


@wp.kernel
def _assemble_vp_kernel(
    vp_w:         wp.float32,
    nnz_diag:     wp.array2d(dtype=wp.int32),     # (B, N) — only [b, 0] is read
    H_values:     wp.array(dtype=mat66_t),
    g:            wp.array2d(dtype=wp.float32),   # (B, 6N)
    r_vp:         wp.array2d(dtype=wp.float32),   # (B, 3)
):
    """Per b: velocity-prior contribution to H[0,0] and g (KF 0 of each batch).

    J_vp = [0_3x3, I_3]  (3x6 row stack);  Σ_vp^{-1} = vp_w * I_3
        H block (6x6): bottom-right 3x3 (v cols) = vp_w * I3, others zero
        g vel rows (3): vp_w * r_vp

    """
    b = wp.tid()
    zero3 = mat33_t(0.0)
    Wv = mat33_t(
        vp_w, 0.0,  0.0,
        0.0,  vp_w, 0.0,
        0.0,  0.0,  vp_w,
    )
    vp_block = _pack_mat66_from_blocks(zero3, zero3, zero3, Wv)
    _accum_block(H_values, nnz_diag[b, 0], vp_block)

    # g vel rows of KF 0 (rows 3, 4, 5): vp_w * r_vp.
    wp.atomic_add(g, b, 3, vp_w * r_vp[b, 0])
    wp.atomic_add(g, b, 4, vp_w * r_vp[b, 1])
    wp.atomic_add(g, b, 5, vp_w * r_vp[b, 2])


@wp.func
def _pack_mat66_from_blocks(
    A: mat33_t,    # top-left
    B: mat33_t,    # top-right
    C: mat33_t,    # bottom-left
    D: mat33_t,    # bottom-right
) -> mat66_t:
    """Pack four 3x3 blocks into one 6x6 constructor to avoid per-element writes."""
    return mat66_t(
        A[0,0], A[0,1], A[0,2], B[0,0], B[0,1], B[0,2],
        A[1,0], A[1,1], A[1,2], B[1,0], B[1,1], B[1,2],
        A[2,0], A[2,1], A[2,2], B[2,0], B[2,1], B[2,2],
        C[0,0], C[0,1], C[0,2], D[0,0], D[0,1], D[0,2],
        C[1,0], C[1,1], C[1,2], D[1,0], D[1,1], D[1,2],
        C[2,0], C[2,1], C[2,2], D[2,0], D[2,1], D[2,2],
    )


@wp.kernel
def _assemble_imu_kernel(
    rot:        wp.array2d(dtype=wp.quat),         # (B, N)
    imu_dr:     wp.array2d(dtype=wp.vec3),         # (B, N)
    vel:        wp.array2d(dtype=wp.vec3),         # (B, N)
    r_imu:      wp.array3d(dtype=wp.float32),      # (B, N-1, 6) — pre-computed
    W_imu:      wp.array4d(dtype=wp.float32),      # (B, N, 6, 6) inverse cov  (slot N-1 unused)
    nnz_diag:   wp.array2d(dtype=wp.int32),        # (B, N)
    nnz_lower:  wp.array2d(dtype=wp.int32),        # (B, N) — lower of KF k means H[k, k-1]
    nnz_upper:  wp.array2d(dtype=wp.int32),        # (B, N) — upper of KF k means H[k, k+1]
    H_values:   wp.array(dtype=mat66_t),
    g:          wp.array2d(dtype=wp.float32),      # (B, 6N)
):
    """Assemble each IMU edge's J^T W J blocks and J^T W r contributions.

    JR_phii = -JR_phij and Jv_phii = -Jv_phij determine the endpoint signs.
    The gradient reads r_imu; Jr_inv uses a recomputed rotation residual.
    """
    b, e = wp.tid()
    Ri = rot[b, e]
    Rj = rot[b, e + 1]
    vj = vel[b, e + 1]
    dr = imu_dr[b, e]

    # --- Jacobian sub-blocks ---
    # Recompute r_R for Jr_inv (cheaper than another array load + assembly).
    dR = _exp_so3(dr)
    q_rR = wp.quat_inverse(dR) * wp.quat_inverse(Ri) * Rj
    r_R = _quat_log(q_rR)
    Jr_inv = _so3_jr_inv(r_R)

    Ri_mat = wp.quat_to_matrix(Ri)
    Rj_mat = wp.quat_to_matrix(Rj)
    Ri_T   = wp.transpose(Ri_mat)
    Rj_T   = wp.transpose(Rj_mat)
    RiT_Rj = Ri_T @ Rj_mat                      # = Jv_vj

    # JR_phij  = +Jr_inv R_j^T,   JR_phii = -JR_phij
    JR_phij = Jr_inv @ Rj_T
    # Jv_phii = +[RiT_Rj vj]_x Ri^T,   Jv_phij = -Jv_phii
    skew_arg = RiT_Rj @ vj
    skew_S = mat33_t(
         0.0,         -skew_arg[2],  skew_arg[1],
         skew_arg[2],  0.0,         -skew_arg[0],
        -skew_arg[1],  skew_arg[0],  0.0,
    )
    Jv_phii = skew_S @ Ri_T

    # --- Load W as four 3x3 sub-blocks directly from W_imu (no mat66 intermediate) ---
    Waa = mat33_t(
        W_imu[b,e,0,0], W_imu[b,e,0,1], W_imu[b,e,0,2],
        W_imu[b,e,1,0], W_imu[b,e,1,1], W_imu[b,e,1,2],
        W_imu[b,e,2,0], W_imu[b,e,2,1], W_imu[b,e,2,2],
    )
    Wab = mat33_t(
        W_imu[b,e,0,3], W_imu[b,e,0,4], W_imu[b,e,0,5],
        W_imu[b,e,1,3], W_imu[b,e,1,4], W_imu[b,e,1,5],
        W_imu[b,e,2,3], W_imu[b,e,2,4], W_imu[b,e,2,5],
    )
    Wba = mat33_t(
        W_imu[b,e,3,0], W_imu[b,e,3,1], W_imu[b,e,3,2],
        W_imu[b,e,4,0], W_imu[b,e,4,1], W_imu[b,e,4,2],
        W_imu[b,e,5,0], W_imu[b,e,5,1], W_imu[b,e,5,2],
    )
    Wbb = mat33_t(
        W_imu[b,e,3,3], W_imu[b,e,3,4], W_imu[b,e,3,5],
        W_imu[b,e,4,3], W_imu[b,e,4,4], W_imu[b,e,4,5],
        W_imu[b,e,5,3], W_imu[b,e,5,4], W_imu[b,e,5,5],
    )

    # --- W @ J columns. Identities used:
    # JR_phii = -JR_phij, Jv_phij = -Jv_phii, Jv_vi = -I, Jv_vj = RiT_Rj.
    # column phi_j: top = JR_phij, bot = Jv_phij = -Jv_phii
    Wcphij_top = Waa @ JR_phij - Wab @ Jv_phii
    Wcphij_bot = Wba @ JR_phij - Wbb @ Jv_phii
    # column phi_i = -column phi_j, so Wcphii = -Wcphij.
    Wcphii_top = wp.neg(Waa @ JR_phij) + Wab @ Jv_phii
    Wcphii_bot = wp.neg(Wba @ JR_phij) + Wbb @ Jv_phii
    # column v_i: top = 0, bot = -I3   →  Wc = -W column-bot
    Wcvi_top   = wp.neg(Wab)            # Wab @ (-I) = -Wab
    Wcvi_bot   = wp.neg(Wbb)
    # column v_j: top = 0, bot = RiT_Rj
    Wcvj_top   = Wab @ RiT_Rj
    Wcvj_bot   = Wbb @ RiT_Rj

    JR_phij_T = wp.transpose(JR_phij)
    Jv_phii_T = wp.transpose(Jv_phii)
    JR_phii_T = wp.neg(JR_phij_T)       # = -JR_phij_T
    Jv_phij_T = wp.neg(Jv_phii_T)       # = -Jv_phii_T
    # Jv_vi_T = -I3, Jv_vj_T = RiT_Rj^T
    Jv_vj_T   = wp.transpose(RiT_Rj)

    # --- H sub-blocks (each 3x3) = col_a^T W col_b ---
    # H[i,i]
    H_phii_phii = JR_phii_T @ Wcphii_top + Jv_phii_T @ Wcphii_bot
    H_phii_vi   = JR_phii_T @ Wcvi_top   + Jv_phii_T @ Wcvi_bot
    H_vi_phii   = wp.neg(Wcphii_bot)    # Jv_vi_T = -I → = -Wcphii_bot
    H_vi_vi     = wp.neg(Wcvi_bot)      # = -Wcvi_bot

    Hii = _pack_mat66_from_blocks(H_phii_phii, H_phii_vi,
                                  H_vi_phii,   H_vi_vi)
    _accum_block(H_values, nnz_diag[b, e], Hii)

    # H[j,j]
    H_phij_phij = JR_phij_T @ Wcphij_top + Jv_phij_T @ Wcphij_bot
    H_phij_vj   = JR_phij_T @ Wcvj_top   + Jv_phij_T @ Wcvj_bot
    H_vj_phij   = Jv_vj_T   @ Wcphij_bot
    H_vj_vj     = Jv_vj_T   @ Wcvj_bot

    Hjj = _pack_mat66_from_blocks(H_phij_phij, H_phij_vj,
                                  H_vj_phij,   H_vj_vj)
    _accum_block(H_values, nnz_diag[b, e + 1], Hjj)

    # H[i,j]
    H_phii_phij = JR_phii_T @ Wcphij_top + Jv_phii_T @ Wcphij_bot
    H_phii_vj   = JR_phii_T @ Wcvj_top   + Jv_phii_T @ Wcvj_bot
    H_vi_phij   = wp.neg(Wcphij_bot)
    H_vi_vj     = wp.neg(Wcvj_bot)

    Hij = _pack_mat66_from_blocks(H_phii_phij, H_phii_vj,
                                  H_vi_phij,   H_vi_vj)
    _accum_block(H_values, nnz_upper[b, e], Hij)

    # H[j,i] = H[i,j]^T — pack from the same 3x3 sub-blocks transposed.
    Hji = _pack_mat66_from_blocks(
        wp.transpose(H_phii_phij), wp.transpose(H_vi_phij),
        wp.transpose(H_phii_vj),   wp.transpose(H_vi_vj),
    )
    _accum_block(H_values, nnz_lower[b, e + 1], Hji)

    # --- gradient contribution g += J_e^T W r_imu_e ---
    r_top = wp.vec3(r_imu[b, e, 0], r_imu[b, e, 1], r_imu[b, e, 2])
    r_bot = wp.vec3(r_imu[b, e, 3], r_imu[b, e, 4], r_imu[b, e, 5])

    # W @ r via the 4 mat33 partition (cleaner SIMD than a 6x6 inner loop).
    Wr_top = Waa @ r_top + Wab @ r_bot
    Wr_bot = Wba @ r_top + Wbb @ r_bot

    # phi_i rows: JR_phii^T @ Wr_top + Jv_phii^T @ Wr_bot
    g_phii = JR_phii_T @ Wr_top + Jv_phii_T @ Wr_bot
    # v_i rows: Jv_vi^T @ Wr_bot = -Wr_bot
    g_vi   = wp.neg(Wr_bot)
    g_phij = JR_phij_T @ Wr_top + Jv_phij_T @ Wr_bot
    g_vj   = Jv_vj_T @ Wr_bot

    base_i = 6 * e
    base_j = 6 * (e + 1)
    wp.atomic_add(g, b, base_i + 0, g_phii[0])
    wp.atomic_add(g, b, base_i + 1, g_phii[1])
    wp.atomic_add(g, b, base_i + 2, g_phii[2])
    wp.atomic_add(g, b, base_i + 3, g_vi[0])
    wp.atomic_add(g, b, base_i + 4, g_vi[1])
    wp.atomic_add(g, b, base_i + 5, g_vi[2])
    wp.atomic_add(g, b, base_j + 0, g_phij[0])
    wp.atomic_add(g, b, base_j + 1, g_phij[1])
    wp.atomic_add(g, b, base_j + 2, g_phij[2])
    wp.atomic_add(g, b, base_j + 3, g_vj[0])
    wp.atomic_add(g, b, base_j + 4, g_vj[1])
    wp.atomic_add(g, b, base_j + 5, g_vj[2])


@wp.kernel
def _bsr_to_csr_values_f64_kernel(
    H_values:  wp.array(dtype=mat66_t),      # (nnz_total,)
    nnz_diag:  wp.array2d(dtype=wp.int32),   # (B, N)
    nnz_lower: wp.array2d(dtype=wp.int32),   # (B, N) — -1 at k=0
    nnz_upper: wp.array2d(dtype=wp.int32),   # (B, N) — -1 at k=N-1
    N:         wp.int32,
    ridge:     wp.float32,
    values_out: wp.array2d(dtype=wp.float64),  # (B, nnz_per_window)
):
    """Pack per-window 6x6 block-tridiagonal H values into row-major scalar CSR.

    One thread per (b_in_chunk, k, i). Each thread writes one scalar row's
    nnz: 12 entries for k in {0, N-1}, 18 entries for k in [1, N-2]. The
    cursor layout is hand-computed to avoid an explicit indptr scan inside
    the kernel.

    Values are up-cast to fp64 on the way out for the GN-step solve.
    """
    b, k, i = wp.tid()
    cursor = wp.int32(0)
    has_lower = k >= 1
    has_upper = k <= N - 2
    if k == 0:
        cursor = 12 * i
    elif k <= N - 2:
        cursor = 72 + (k - 1) * 108 + 18 * i
    else:
        cursor = 72 + (N - 2) * 108 + 12 * i

    Hkk = H_values[nnz_diag[b, k]]
    if has_lower:
        Hkl = H_values[nnz_lower[b, k]]
        for j in range(6):
            values_out[b, cursor + j] = wp.float64(Hkl[i, j])
        cursor = cursor + 6
    for j in range(6):
        v = Hkk[i, j]
        if i == j:
            v = v + ridge
        values_out[b, cursor + j] = wp.float64(v)
    cursor = cursor + 6
    if has_upper:
        Hku = H_values[nnz_upper[b, k]]
        for j in range(6):
            values_out[b, cursor + j] = wp.float64(Hku[i, j])


class Sparse_IMUPGOGraph(I_IMUFactorGraph):
    def __init__(
        self,
        net_vel_b: jt.Float[torch.Tensor,  "B N 3"  ],
        net_vel_cov_b: jt.Float[torch.Tensor,  "B N 3 3"],
        imu_dv_b     : jt.Float[torch.Tensor,  "B N 3"  ],
        imu_dr_b     : jt.Float[torch.Tensor,  "B N 3"  ],
        imu_dt_ms       : jt.Float[torch.Tensor,  "B N 1"  ],  # ms (unused; kept for API parity)
        imu_dvr_cov  : jt.Float[torch.Tensor,  "B N 6 6"],
        rot_wb       : jt.Float[pp.LieTensor,  "B 1 3"  ],
        vel_prior        : jt.Float[torch.Tensor, "B 1 3"] | None = None,
        anchor_weight    : float = 1.0e6,
        vel_prior_weight : float = 1.0e6,
        max_iter     : int   = 10,
        tol          : float = 1.0e-5,
        obs_reject_chi2  : float | None = None,
        obs_reject_all   : bool = False,
    ):
        torch.nn.Module.__init__(self)

        assert net_vel_b.dim() == 3 and net_vel_b.shape[-1] == 3
        B, N, _ = net_vel_b.shape
        assert N >= 2
        assert (
            net_vel_cov_b.shape == (B, N, 3, 3)
            and imu_dv_b.shape == (B, N, 3)
            and imu_dr_b.shape == (B, N, 3)
            and imu_dt_ms.shape == (B, N, 1)
            and imu_dvr_cov.shape == (B, N, 6, 6)
            and rot_wb.shape == (B, 1, 3)
        )

        anchor_weight = float(anchor_weight)
        vel_prior_weight = float(vel_prior_weight)
        max_iter = int(max_iter)
        tol = float(tol)
        if anchor_weight <= 0.0:
            raise ValueError(f"anchor_weight must be > 0, got {anchor_weight}")
        if vel_prior_weight <= 0.0:
            raise ValueError(f"vel_prior_weight must be > 0, got {vel_prior_weight}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if tol <= 0.0:
            raise ValueError(f"tol must be > 0, got {tol}")

        device = net_vel_b.device
        if device.type != "cuda":
            raise RuntimeError(
                f"Sparse_IMUPGOGraph requires CUDA inputs, got device={device}. "
                "Use Dense_IMUPGOGraph for CPU."
            )

        self._B = B
        self._N = N
        self._anchor_weight    = anchor_weight
        self._vel_prior_weight = vel_prior_weight
        self._max_iter = max_iter
        self._tol      = tol
        # Chi-square(3) gate shared with the dense backend; None disables it.
        if obs_reject_chi2 is not None and float(obs_reject_chi2) <= 0.0:
            raise ValueError(
                f"obs_reject_chi2 must be > 0 or null, got {obs_reject_chi2}"
            )
        self._obs_reject_chi2 = None if obs_reject_chi2 is None else float(obs_reject_chi2)
        # True: reject every violator per pass (trajectory-level graphs); False:
        # <=1 per pass. See Optimizer/outlier_gate.py.
        self._obs_reject_all = bool(obs_reject_all)
        self._device   = device
        self._wp_device = f"cuda:{device.index}" if device.index is not None else "cuda:0"

        dtype = torch.float32  # Fixed Warp BSR block dtype.

        self.imu_dv_b      = torch.nn.Buffer(imu_dv_b.to(dtype).contiguous())            # (B, N, 3)
        self.imu_dr_b      = torch.nn.Buffer(imu_dr_b.to(dtype).contiguous())            # (B, N, 3)
        self.imu_dvr_cov   = torch.nn.Buffer(imu_dvr_cov.to(dtype).contiguous())         # (B, N, 6, 6)
        self.net_vel_b = torch.nn.Buffer(net_vel_b.to(dtype).contiguous())       # (B, N, 3)
        self.net_vel_cov_b = torch.nn.Buffer(net_vel_cov_b.to(dtype).contiguous())       # (B, N, 3, 3)
        W_imu_full = _safe_cov_inv(self.imu_dvr_cov)            # (B, N, 6, 6)
        W_obs_full = _safe_cov_inv(self.net_vel_cov_b)           # (B, N, 3, 3)
        self.W_imu = torch.nn.Buffer(W_imu_full.contiguous())
        self.W_obs = torch.nn.Buffer(W_obs_full.contiguous())
        # _wp_W_obs is a warp view onto self.W_obs's memory, so rejection edits it
        # in place; keep the unmasked copy and the mask to rebuild from.
        self._W_obs_unmasked = W_obs_full.contiguous().clone()
        self._obs_mask = torch.ones((B, N), dtype=dtype, device=device)
        self._n_rejected = torch.zeros(B, dtype=torch.long, device=device)

        anchor_so3 = pp.so3(rot_wb.to(dtype)).Exp()              # (B, 1) SO3
        self.anchor_quat = torch.nn.Buffer(
            anchor_so3.tensor().squeeze(1).contiguous()           # (B, 4)
        )

        self._has_vel_prior = vel_prior is not None
        if self._has_vel_prior:
            vp = vel_prior.to(dtype)
            assert vp.shape == (B, 1, 3)
            self.vel_prior = torch.nn.Buffer(vp.squeeze(1).contiguous())   # (B, 3)

        # Chain body-frame increments by right multiplication in one Warp launch.
        R_init = torch.empty((B, N, 4), dtype=dtype, device=device)
        anchor_wp = wp.from_torch(self.anchor_quat.contiguous(), dtype=wp.quat)   # (B,)
        imu_dr_wp = wp.from_torch(self.imu_dr_b,                  dtype=wp.vec3)  # (B, N)
        rot_out_wp = wp.from_torch(R_init,                        dtype=wp.quat)  # (B, N)
        wp.launch(
            _rot_chain_kernel, dim=B,
            inputs=[anchor_wp, imu_dr_wp, rot_out_wp, int(N)],
            device=self._wp_device,
        )
        wp.synchronize()
        self._rot = torch.nn.Parameter(R_init.contiguous())                 # (B, N, 4) raw quat
        self._vel = torch.nn.Parameter(self.net_vel_b.clone())          # (B, N, 3)
        if self._has_vel_prior:
            with torch.no_grad():
                self._vel[:, 0] = self.vel_prior

        # Reuse the shape-dependent pattern and lazily built solver bundles.
        key = (B, N, str(self._wp_device))
        if key not in _GRAPH_RESOURCES:
            _GRAPH_RESOURCES[key] = _GraphResources(
                pattern=_build_pattern(B, N, device, self._wp_device)
            )
        self._resources = _GRAPH_RESOURCES[key]
        pattern = self._resources.pattern
        self._nnz_idx = torch.nn.Buffer(pattern.nnz_idx.contiguous())  # (B, N, 3) int32

        # Per-instance Hessian values share immutable cached offsets/columns.
        values_wp = wp.zeros(pattern.nnz, dtype=mat66_t, device=self._wp_device)
        self._H = ws.bsr_zeros(pattern.M_blocks, pattern.M_blocks, mat66_t, device=self._wp_device)
        self._H.offsets = pattern.offsets_wp
        self._H.columns = pattern.columns_wp
        self._H.values  = values_wp
        self._H.nnz     = pattern.nnz

        # Stable scratch/output storage lets Warp views survive every step.
        # Update buffers in place; do not reallocate them.
        D = 6 * (N - 1) + 3 * N + 3 + (3 if self._has_vel_prior else 0)
        self._D = D
        self._r_imu_buf = torch.nn.Buffer(torch.zeros(B, N - 1, 6, dtype=dtype, device=device))
        self._r_obs_buf = torch.nn.Buffer(torch.zeros(B, N, 3,    dtype=dtype, device=device))
        self._r_anc_buf = torch.nn.Buffer(torch.zeros(B, 3,       dtype=dtype, device=device))
        if self._has_vel_prior:
            self._r_vp_buf = torch.nn.Buffer(torch.zeros(B, 3,    dtype=dtype, device=device))
        # Gradient g and step dx are flat (B*6N,) for kernel addressing.
        self._g_torch     = torch.nn.Buffer(torch.zeros(B * N * 6, dtype=dtype, device=device))
        self._dx_torch    = torch.nn.Buffer(torch.zeros(B * N * 6, dtype=dtype, device=device))
        # Pre-sliced nnz lookup tensors (contiguous, so wp.from_torch is happy).
        self._nnz_diag_torch  = torch.nn.Buffer(self._nnz_idx[..., 0].contiguous())
        self._nnz_lower_torch = torch.nn.Buffer(self._nnz_idx[..., 1].contiguous())
        self._nnz_upper_torch = torch.nn.Buffer(self._nnz_idx[..., 2].contiguous())

        # Bind zero-copy Warp views once; all referenced storage must keep its identity.
        self._wp_imu_dr     = wp.from_torch(self.imu_dr_b,                dtype=wp.vec3)
        self._wp_imu_dv     = wp.from_torch(self.imu_dv_b,                dtype=wp.vec3)
        self._wp_W_imu      = wp.from_torch(self.W_imu,                   dtype=wp.float32)
        self._wp_W_obs      = wp.from_torch(self.W_obs,                   dtype=wp.float32)
        self._wp_net_vel    = wp.from_torch(self.net_vel_b,           dtype=wp.vec3)
        self._wp_anchor     = wp.from_torch(self.anchor_quat,             dtype=wp.quat)
        self._wp_nnz_diag   = wp.from_torch(self._nnz_diag_torch,         dtype=wp.int32)
        self._wp_nnz_lower  = wp.from_torch(self._nnz_lower_torch,        dtype=wp.int32)
        self._wp_nnz_upper  = wp.from_torch(self._nnz_upper_torch,        dtype=wp.int32)
        self._wp_rot        = wp.from_torch(self._rot.detach(),           dtype=wp.quat)
        self._wp_vel        = wp.from_torch(self._vel.detach(),           dtype=wp.vec3)
        self._wp_r_imu      = wp.from_torch(self._r_imu_buf,              dtype=wp.float32)
        self._wp_r_obs      = wp.from_torch(self._r_obs_buf,              dtype=wp.float32)
        self._wp_r_anc      = wp.from_torch(self._r_anc_buf,              dtype=wp.float32)
        if self._has_vel_prior:
            self._wp_vel_prior = wp.from_torch(self.vel_prior,           dtype=wp.vec3)
            self._wp_r_vp      = wp.from_torch(self._r_vp_buf,           dtype=wp.float32)
        self._wp_g          = wp.from_torch(self._g_torch.view(B, 6 * N), dtype=wp.float32)
        # The update kernel reads the same step buffer through an interleaved view.
        self._wp_dx_3d      = wp.from_torch(self._dx_torch.view(B, N, 6),  dtype=wp.float32)

    @torch.no_grad()
    def forward(self) -> jt.Float[torch.Tensor, "B D"]:
        """Residual order: IMU edges, velocity observations, rotation anchor, velocity prior.

        D = 9N - 3 without the optional velocity prior, 9N with it.
        """
        self._fill_residuals()
        parts = [
            self._r_imu_buf.reshape(self._B, -1),
            self._r_obs_buf.reshape(self._B, -1),
            self._r_anc_buf,
        ]
        if self._has_vel_prior:
            parts.append(self._r_vp_buf)
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def _fill_residuals(self) -> None:
        B, N = self._B, self._N
        wp.launch(
            _residual_imu_kernel, dim=(B, N - 1),
            inputs=[self._wp_rot, self._wp_vel, self._wp_imu_dr,
                    self._wp_imu_dv, self._wp_r_imu],
            device=self._wp_device,
        )
        wp.launch(
            _residual_obs_kernel, dim=(B, N),
            inputs=[self._wp_vel, self._wp_net_vel, self._wp_r_obs],
            device=self._wp_device,
        )
        wp.launch(
            _residual_anc_kernel, dim=B,
            inputs=[self._wp_rot, self._wp_anchor, self._wp_r_anc],
            device=self._wp_device,
        )
        if self._has_vel_prior:
            wp.launch(
                _residual_vp_kernel, dim=B,
                inputs=[self._wp_vel, self._wp_vel_prior, self._wp_r_vp],
                device=self._wp_device,
            )

    @torch.no_grad()
    def _assemble_hessian_and_gradient(self) -> None:
        """Assemble H and g in place without changing rotation or velocity state."""
        B, N = self._B, self._N
        self._fill_residuals()

        # Zero H values and g in-place (preserves storage / pre-bound Warp views).
        wp.launch(_zero_bsr_values, dim=self._H.values.shape[0],
                  inputs=[self._H.values], device=self._wp_device)
        self._g_torch.zero_()

        wp.launch(
            _assemble_imu_kernel, dim=(B, N - 1),
            inputs=[self._wp_rot, self._wp_imu_dr, self._wp_vel, self._wp_r_imu,
                    self._wp_W_imu,
                    self._wp_nnz_diag, self._wp_nnz_lower, self._wp_nnz_upper,
                    self._H.values, self._wp_g],
            device=self._wp_device,
        )
        # Separate B-thread anchor kernels avoid a k==0 branch across B*N observations.
        wp.launch(
            _assemble_obs_kernel, dim=(B, N),
            inputs=[self._wp_W_obs, self._wp_nnz_diag,
                    self._H.values, self._wp_g, self._wp_r_obs],
            device=self._wp_device,
        )
        wp.launch(
            _assemble_anc_kernel, dim=B,
            inputs=[self._wp_rot, float(self._anchor_weight),
                    self._wp_nnz_diag, self._H.values, self._wp_g, self._wp_r_anc],
            device=self._wp_device,
        )
        if self._has_vel_prior:
            wp.launch(
                _assemble_vp_kernel, dim=B,
                inputs=[float(self._vel_prior_weight), self._wp_nnz_diag,
                        self._H.values, self._wp_g, self._wp_r_vp],
                device=self._wp_device,
            )

    @torch.no_grad()
    def step(self) -> dict:
        """One Gauss-Newton iteration: assemble H, g → pack into batched scalar
        CSR (fp64, + ridge) → cuDSS factorize + solve H dx = -g → apply.

        Returns ``loss`` (r^T W r, monitoring), ``step_norm`` (max |dx|) and
        ``bad`` (B,) — windows whose solution came back non-finite. Those
        windows get a zero step (their state is left where it was); the caller
        decides what to do with them via ``solve()["failed"]``.
        """
        B, N = self._B, self._N

        self._assemble_hessian_and_gradient()

        # Cache the symbolic plan per (B, N, device).
        if self._resources.step is None:
            self._resources.step = _build_cudss_step(B, N, self._device, self._wp_device)
        st = self._resources.step
        wp.launch(
            _bsr_to_csr_values_f64_kernel, dim=(B, N, 6),
            inputs=[self._H.values, self._wp_nnz_diag, self._wp_nnz_lower, self._wp_nnz_upper,
                    int(N), float(_CUDSS_RIDGE), st.wp_values],
            device=self._wp_device,
        )
        # RHS b = -g, up-cast into the live operand the solver is bound to.
        st.rhs_storage[:, 0, :].copy_(self._g_torch.view(B, 6 * N)).neg_()

        # Reuse the symbolic plan; only numeric factorization and solves repeat.
        st.solver.factorize()
        x = st.solver.solve()                                   # (B, 6N, 1) fp64

        # Zero non-finite per-window steps before updating state.
        dx_b = x.reshape(B, 6 * N)
        bad  = ~torch.isfinite(dx_b).all(dim=-1)                # (B,)
        if bool(bad.any()):
            dx_b = torch.where(bad[:, None], torch.zeros_like(dx_b), dx_b)
        self._dx_torch.copy_(dx_b.reshape(-1))                  # fp64 → fp32

        wp.launch(
            _apply_step_kernel, dim=(B, N),
            inputs=[self._wp_dx_3d, self._wp_rot, self._wp_vel],
            device=self._wp_device,
        )

        loss = self._mahalanobis_loss()
        return {
            "loss":      loss,
            "step_norm": self._dx_torch.abs().max().item(),
            "bad":       bad,
        }

    @torch.no_grad()
    def _mahalanobis_loss(self) -> float:
        """Compute r^T W r over the current residuals (for monitoring)."""
        # r_imu (B, N-1, 6) with W_imu[:, :N-1]
        r_i  = self._r_imu_buf
        Wr_i = (self.W_imu[:, :self._N - 1] @ r_i.unsqueeze(-1)).squeeze(-1)
        l_imu = (r_i * Wr_i).sum().item()
        r_o  = self._r_obs_buf
        Wr_o = (self.W_obs @ r_o.unsqueeze(-1)).squeeze(-1)
        l_obs = (r_o * Wr_o).sum().item()
        l_anc = self._anchor_weight * (self._r_anc_buf ** 2).sum().item()
        l_vp  = (
            self._vel_prior_weight * (self._r_vp_buf ** 2).sum().item()
            if self._has_vel_prior else 0.0
        )
        return (l_imu + l_obs + l_anc + l_vp) / max(self._B, 1)

    @torch.no_grad()
    def _reject_obs_outliers(self) -> int:
        """Re-score observations through the shared gate, updating Warp-bound weights in place."""
        if self._obs_reject_chi2 is None:
            return 0
        d2 = mahalanobis_d2(self._vel - self.net_vel_b, self._W_obs_unmasked)
        keep, n_rejected = rejection_mask(
            d2, self._obs_reject_chi2, n_rejected_prev=self._n_rejected,
            reject_all=self._obs_reject_all,
        )
        if torch.equal(keep, self._obs_mask):
            return 0
        self._obs_mask, self._n_rejected = keep, n_rejected
        self.W_obs.copy_(self._W_obs_unmasked * self._obs_mask[..., None, None])
        return int(n_rejected.sum())

    @torch.no_grad()
    def _gn_loop(self) -> tuple[list[float], bool, torch.Tensor]:
        """Run GN until convergence or max_iter. Returns (history, converged, failed)
        with ``failed`` a (B,) mask of windows whose solve ever came back non-finite
        — identical contract to ``Dense_IMUPGOGraph._gn_loop``."""
        history = []
        prev_loss = None
        converged = False
        failed = torch.zeros(self._B, dtype=torch.bool, device=self._device)
        for _ in range(self._max_iter):
            info = self.step()
            history.append(info["loss"])
            failed |= info["bad"]
            if bool(failed.all()):
                break
            if info["step_norm"] < self._tol:
                converged = True
                break
            if prev_loss is not None:
                rel = abs(prev_loss - info["loss"]) / max(prev_loss, 1.0e-12)
                if rel < 5.0e-4:
                    converged = True
                    break
            prev_loss = info["loss"]
        return history, converged, failed

    @torch.no_grad()
    def solve(self) -> dict:
        """Run GN with bounded rejection/re-solve rounds; return the dense solve contract."""
        history, converged, failed = self._gn_loop()
        for _ in range(_MAX_REJECT_PASSES):
            n = self._reject_obs_outliers()
            if n == 0:
                break
            h, converged, f = self._gn_loop()
            history += h
            failed |= f
        return {
            "iters":      len(history),
            "final_loss": history[-1] if history else float("nan"),
            "converged":  converged and not bool(failed.any()),
            "failed":     failed,                        # (B,) windows whose solve came back non-finite
            "rejected":   int((self._obs_mask == 0).sum().item()),   # velocity observations gated out (total)
        }

    @property
    def get_state(self) -> jt.Float[torch.Tensor, "B N 3"]:
        return self._vel.detach()

    @property
    def get_rotation(self) -> jt.Float[pp.LieTensor, "B N 4"]:
        return pp.SO3(self._rot.detach())
