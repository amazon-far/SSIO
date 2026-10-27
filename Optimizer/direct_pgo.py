"""Batched per-window IMU + velocity pose-graph optimizer.

Optimizes (rotation, body-frame velocity) jointly per window via batched
Gauss-Newton with dense Cholesky. Position is intentionally not part of the
state — body-frame velocity labels are translation-invariant and the
optimizer only needs to recover (R, v).

State per batch element: x = [phi_0, ..., phi_{N-1}, v_0, ..., v_{N-1}] in R^{6N}.
- Rotations stored as pp.SO3 quaternions, updated via left perturbation:
    R_k <- Exp(delta_phi_k) @ R_k
- Velocities stored as plain (N, 3) tensors, additive update.

Factor graph:
1. IMU preintegration (binary, edges k -> k+1, 6-vector residual each):
    r_R[k] = Log( Exp(imu_dr_b[k])^{-1} R_k^T R_{k+1} )
    r_v[k] = R_k^T R_{k+1} v_{k+1} - v_k - imu_dv_b[k]
2. Velocity observation (unary, per KF, 3-vector):
    r_obs[k] = v_k - net_vel_b[k]
3. Rotation anchor (unary at KF 0, 3-vector):
    r_anc = Log( Exp(rot_wb)^{-1} R_0 )
4. Initial velocity prior (unary at KF 0, 3-vector) — optional, active only
   when ``vel_prior`` is supplied (body frame, same convention as the velocity
   state and net_vel_b):
    r_vp = v_0 - vel_prior

Conventions:
- IMU slot k describes segment KF_k -> KF_{k+1}; slot N-1 is unused.
- All Jacobians use left-perturbation SO(3).
- Information matrix is block-diagonal across factors: inv(imu_dvr_cov[k])
  for each IMU edge, inv(net_vel_cov_b[k]) for each obs, anchor_weight * I_3
  for the rotation anchor, and (when present) vel_prior_weight * I_3 for the
  initial velocity prior.
"""

from __future__ import annotations

import torch
import pypose as pp
import jaxtyping as jt

from Interface.graph import I_IMUFactorGraph
from .outlier_gate import mahalanobis_d2, rejection_mask


_MAX_REJECT_PASSES = 5
_EPS_COV = 1e-10


def _safe_cov_inv(cov: torch.Tensor) -> torch.Tensor:
    """Symmetrize and invert with an _EPS_COV diagonal ridge.

    The ridge regularizes singular IMU covariances, including zero rotation noise.
    """
    eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
    cov = 0.5 * (cov + cov.transpose(-1, -2)) + _EPS_COV * eye
    return torch.linalg.inv(cov)


class Dense_IMUPGOGraph(I_IMUFactorGraph):
    def __init__(
        self,
        net_vel_b: jt.Float[torch.Tensor,  "B N 3"  ],
        net_vel_cov_b: jt.Float[torch.Tensor,  "B N 3 3"],
        imu_dv_b     : jt.Float[torch.Tensor,  "B N 3"  ],
        imu_dr_b     : jt.Float[torch.Tensor,  "B N 3"  ],
        imu_dt_ms       : jt.Float[torch.Tensor,  "B N 1"  ],  # ms
        imu_dvr_cov  : jt.Float[torch.Tensor,  "B N 6 6"],
        rot_wb       : jt.Float[pp.LieTensor,  "B 1 3"  ],
        vel_prior        : jt.Float[torch.Tensor, "B 1 3"] | None = None,
        anchor_weight    : float = 1.0e6,
        vel_prior_weight : float = 1.0e6,
        max_iter         : int   = 10,
        tol              : float = 1.0e-5,
        obs_reject_chi2  : float | None = None,
        obs_reject_all   : bool = False,
    ):
        # I_IMUFactorGraph declares no __init__, so initialize torch.nn.Module
        # directly for buffer/parameter registration.
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

        dtype  = net_vel_b.dtype

        self._B = B
        self._N = N
        self._anchor_weight     = anchor_weight
        self._vel_prior_weight  = vel_prior_weight
        self._max_iter = max_iter
        self._tol      = tol
        # Chi-square(3) gating limits the influence of outlying velocity observations.
        if obs_reject_chi2 is not None and float(obs_reject_chi2) <= 0.0:
            raise ValueError(
                f"obs_reject_chi2 must be > 0 or null, got {obs_reject_chi2}"
            )
        self._obs_reject_chi2 = None if obs_reject_chi2 is None else float(obs_reject_chi2)
        # True: reject every violator per pass (trajectory-level graphs); False:
        # <=1 per pass. See Optimizer/outlier_gate.py.
        self._obs_reject_all = bool(obs_reject_all)
        self._obs_mask = torch.ones(
            (B, N), dtype=net_vel_b.dtype, device=net_vel_b.device,
        )
        self._n_rejected = torch.zeros(B, dtype=torch.long, device=net_vel_b.device)

        dt_seconds = imu_dt_ms.to(dtype).squeeze(-1) * 1.0e-3      # (B, N)
        self.imu_dv_b      = torch.nn.Buffer(imu_dv_b.to(dtype))       # (B, N, 3)
        self.imu_dr_b      = torch.nn.Buffer(imu_dr_b.to(dtype))       # (B, N, 3)
        self.imu_dt_s      = torch.nn.Buffer(dt_seconds)               # (B, N)
        self.imu_dvr_cov   = torch.nn.Buffer(imu_dvr_cov.to(dtype))    # (B, N, 6, 6)
        self.net_vel_b = torch.nn.Buffer(net_vel_b.to(dtype))  # (B, N, 3)
        self.net_vel_cov_b = torch.nn.Buffer(net_vel_cov_b.to(dtype))  # (B, N, 3, 3)

        # rot_wb is (B, 1, 3) so(3) tangent — anchor for KF 0.
        anchor_so3 = pp.so3(rot_wb.to(dtype)).Exp()                    # (B, 1, 4) SO3
        self.anchor_rot_quat = torch.nn.Buffer(anchor_so3.tensor())    # store raw tensor

        self._has_vel_prior = vel_prior is not None
        if self._has_vel_prior:
            vp = vel_prior.to(dtype)
            assert vp.shape == (B, 1, 3)
            self.vel_prior = torch.nn.Buffer(vp)                       # (B, 1, 3)

        # Rotation init: R_0 = anchor, R_{k+1} = R_k · Exp(dr_b[k]) (dr is in the body frame at KF k).
        with torch.no_grad():
            chain = torch.cat([anchor_so3.tensor(), pp.so3(imu_dr_b[:, :-1].to(dtype)).Exp().tensor()], dim=1)
            R_init = pp.cumprod(pp.SO3(chain), dim=1, left=False).tensor()   # (B, N, 4)

        self._rot = pp.Parameter(pp.SO3(R_init))                          # (B, N, 4)
        self._vel = torch.nn.Parameter(net_vel_b.to(dtype).clone())   # (B, N, 3)
        if self._has_vel_prior:
            with torch.no_grad():
                self._vel[:, 0] = self.vel_prior[:, 0]

        # Column slice helpers in 6N layout: phi block [0, 3N); v block [3N, 6N).
        self._dim = 6 * N
        # IMU residual rows:  [0, 6(N-1)); per-edge layout [r_R(3), r_v(3)].
        # Vel-obs rows:       [6(N-1), 6(N-1)+3N) = [6N-6, 9N-6).
        # Rot-anchor rows:    [9N-6, 9N-3).
        # Vel-prior rows:     [9N-3, 9N) — present only when vel_prior is given.
        self._D_imu = 6 * (N - 1)
        self._D_obs = 3 * N
        self._D_anc = 3
        self._D_vp  = 3 if self._has_vel_prior else 0
        self._D = self._D_imu + self._D_obs + self._D_anc + self._D_vp

    def _residual_imu(self) -> torch.Tensor:
        """Per-edge [r_R, r_v] residuals in body frame B_k; see module equations."""
        R   = self._rot                                # (B, N, 4) SO3
        v   = self._vel                                # (B, N, 3)
        Ri  = pp.SO3(R[:, :-1])                        # (B, N-1, 4)
        Rj  = pp.SO3(R[:, 1:])                         # (B, N-1, 4)
        vi  = v[:, :-1]                                # (B, N-1, 3)
        vj  = v[:, 1:]                                 # (B, N-1, 3)
        dv  = self.imu_dv_b[:, :-1]                    # (B, N-1, 3)

        DR  = pp.so3(self.imu_dr_b[:, :-1]).Exp()      # (B, N-1, 4)
        r_R = (DR.Inv() @ Ri.Inv() @ Rj).Log().tensor()  # (B, N-1, 3)
        r_v = (Ri.Inv() @ Rj) @ vj - vi - dv             # (B, N-1, 3)
        return torch.cat([r_R, r_v], dim=-1)             # (B, N-1, 6)

    def _residual_obs(self) -> torch.Tensor:
        return self._vel - self.net_vel_b

    def _residual_anchor(self) -> torch.Tensor:
        R0   = pp.SO3(self._rot[:, 0:1])               # (B, 1, 4)
        Ranc = pp.SO3(self.anchor_rot_quat)            # (B, 1, 4)
        r_R  = (Ranc.Inv() @ R0).Log().tensor()        # (B, 1, 3)
        return r_R.squeeze(1)                          # (B, 3)

    def _residual_vel_prior(self) -> torch.Tensor:
        """Body-frame initial-velocity residual at KF 0; called only with a prior."""
        return self._vel[:, 0] - self.vel_prior[:, 0]  # (B, 3)

    def forward(self) -> jt.Float[torch.Tensor, "B D"]:
        """Residual order: IMU edges, velocity observations, rotation anchor, velocity prior.

        D = 9N - 3 without the optional velocity prior, 9N with it.
        """
        r_imu = self._residual_imu().reshape(self._B, -1)      # (B, 6(N-1))
        r_obs = self._residual_obs().reshape(self._B, -1)      # (B, 3N)
        r_anc = self._residual_anchor()                        # (B, 3)
        parts = [r_imu, r_obs, r_anc]
        if self._has_vel_prior:
            parts.append(self._residual_vel_prior())           # (B, 3)
        return torch.cat(parts, dim=-1)                        # (B, D)

    @torch.no_grad()
    def jacobian(self) -> jt.Float[torch.Tensor, "B D dim6N"]:
        """Analytic left-perturbation Jacobian; rows follow forward().

        Columns: [phi_0..phi_{N-1}, v_0..v_{N-1}].
        """
        B, N = self._B, self._N
        device, dtype = self._vel.device, self._vel.dtype
        I3 = torch.eye(3, device=device, dtype=dtype)

        J = torch.zeros((B, self._D, self._dim), device=device, dtype=dtype)

        # --- 1. IMU rows: edges e = 0..N-2 ---
        # Row block per edge: rows [6e, 6e+6) = [r_R(3), r_v(3)].
        Ri = pp.SO3(self._rot[:, :-1])                  # (B, N-1, 4)
        Rj = pp.SO3(self._rot[:, 1:] )                  # (B, N-1, 4)
        vj = self._vel[:, 1:]                           # (B, N-1, 3)

        Ri_mat = Ri.matrix()                            # (B, N-1, 3, 3)
        Rj_mat = Rj.matrix()                            # (B, N-1, 3, 3)
        Ri_T   = Ri_mat.transpose(-1, -2)               # (B, N-1, 3, 3)
        Rj_T   = Rj_mat.transpose(-1, -2)               # (B, N-1, 3, 3)
        RiT_Rj = Ri_T @ Rj_mat                          # (B, N-1, 3, 3)

        DR     = pp.so3(self.imu_dr_b[:, :-1]).Exp()   # (B, N-1, 4)
        r_R    = (DR.Inv() @ Ri.Inv() @ Rj).Log().tensor()  # (B, N-1, 3)
        Jr_inv = torch.linalg.inv(pp.so3(r_R).Jr())     # (B, N-1, 3, 3)

        # [R_i^T R_j v_j]_x R_i^T  appears in dr_v/dphi_{i,j}
        skew_RijTvj = pp.vec2skew((RiT_Rj @ vj.unsqueeze(-1)).squeeze(-1))  # (B, N-1, 3, 3)
        v_phi_block = skew_RijTvj @ Ri_T                # (B, N-1, 3, 3)

        # Row index helpers (per-edge slot of size 6 within the IMU row block).
        edge = torch.arange(N - 1, device=device)
        rR_rows = (6 * edge).view(1, -1, 1, 1) + torch.arange(3, device=device).view(1, 1, -1, 1)  # (1, N-1, 3, 1)
        rv_rows = (6 * edge + 3).view(1, -1, 1, 1) + torch.arange(3, device=device).view(1, 1, -1, 1)
        # Column index helpers (per-KF block of size 3 in phi cols [0,3N) or v cols [3N,6N)).
        phi_i_cols = (3 * edge).view(1, -1, 1, 1) + torch.arange(3, device=device).view(1, 1, 1, -1)
        phi_j_cols = (3 * (edge + 1)).view(1, -1, 1, 1) + torch.arange(3, device=device).view(1, 1, 1, -1)
        v_i_cols   = (3 * N + 3 * edge).view(1, -1, 1, 1) + torch.arange(3, device=device).view(1, 1, 1, -1)
        v_j_cols   = (3 * N + 3 * (edge + 1)).view(1, -1, 1, 1) + torch.arange(3, device=device).view(1, 1, 1, -1)

        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1)

        # dr_R/dphi_i = -Jr_inv R_j^T,   dr_R/dphi_j = +Jr_inv R_j^T
        block_rR_phii = -Jr_inv @ Rj_T                   # (B, N-1, 3, 3)
        block_rR_phij =  Jr_inv @ Rj_T
        J[b_idx, rR_rows, phi_i_cols] = block_rR_phii
        J[b_idx, rR_rows, phi_j_cols] = block_rR_phij

        # dr_v/dv_i = -I3,   dr_v/dv_j = R_i^T R_j
        I3_b = I3.expand(B, N - 1, 3, 3)
        J[b_idx, rv_rows, v_i_cols] = -I3_b
        J[b_idx, rv_rows, v_j_cols] =  RiT_Rj

        # dr_v/dphi_i = +[RiTRj vj]_x Ri^T,   dr_v/dphi_j = -[RiTRj vj]_x Ri^T
        J[b_idx, rv_rows, phi_i_cols] =  v_phi_block
        J[b_idx, rv_rows, phi_j_cols] = -v_phi_block

        # --- 2. Velocity observation rows: 3N rows starting at 6(N-1) ---
        obs_row0 = self._D_imu                              # 6(N-1)
        kf  = torch.arange(N, device=device)
        obs_rows = (obs_row0 + 3 * kf).view(1, -1, 1, 1) + torch.arange(3, device=device).view(1, 1, -1, 1)
        v_cols   = (3 * N + 3 * kf).view(1, -1, 1, 1)       + torch.arange(3, device=device).view(1, 1, 1, -1)
        b_idx_v  = torch.arange(B, device=device).view(B, 1, 1, 1)
        J[b_idx_v, obs_rows, v_cols] = I3.expand(B, N, 3, 3)

        # --- 3. Anchor rows: 3 rows at the bottom ---
        # dr_anc/dphi_0 = Jr_inv(r_anc) @ R_0^T
        R0_T = pp.SO3(self._rot[:, 0]).matrix().transpose(-1, -2)           # (B, 3, 3)
        Ranc = pp.SO3(self.anchor_rot_quat)                                 # (B, 1, 4)
        r_anc = (Ranc.Inv() @ pp.SO3(self._rot[:, 0:1])).Log().tensor().squeeze(1)  # (B, 3)
        Jr_inv_anc = torch.linalg.inv(pp.so3(r_anc).Jr())                   # (B, 3, 3)
        anc_row0 = self._D_imu + self._D_obs                                # 9N-6
        J[:, anc_row0:anc_row0 + 3, 0:3] = Jr_inv_anc @ R0_T

        # --- 4. Velocity prior rows (optional): 3 rows below the rot anchor ---
        # r_vp = v_0 - vel_prior  =>  dr_vp/dv_0 = I3 (cols [3N, 3N+3)), all else 0.
        if self._has_vel_prior:
            vp_row0 = self._D_imu + self._D_obs + self._D_anc               # 9N-3
            v0_col0 = 3 * N
            J[:, vp_row0:vp_row0 + 3, v0_col0:v0_col0 + 3] = I3.expand(B, 3, 3)

        return J

    def weight_matrix(self) -> jt.Float[torch.Tensor, "B D D"]:
        """Block-diagonal factor information in forward() residual order."""
        B, N = self._B, self._N
        device, dtype = self._vel.device, self._vel.dtype

        W_imu = _safe_cov_inv(self.imu_dvr_cov[:, :-1])     # (B, N-1, 6, 6)
        # Rejected observations receive zero information.
        W_obs = _safe_cov_inv(self.net_vel_cov_b) * self._obs_mask[..., None, None]
        W_anc = self._anchor_weight * torch.eye(3, device=device, dtype=dtype)  # (3, 3)
        W_anc = W_anc.expand(B, 3, 3)                        # (B, 3, 3)

        W = torch.zeros((B, self._D, self._D), device=device, dtype=dtype)
        for e in range(N - 1):
            W[:, 6*e:6*e+6, 6*e:6*e+6] = W_imu[:, e]
        for k in range(N):
            r0 = self._D_imu + 3 * k
            W[:, r0:r0+3, r0:r0+3] = W_obs[:, k]
        anc_r0 = self._D_imu + self._D_obs
        W[:, anc_r0:anc_r0+3, anc_r0:anc_r0+3] = W_anc
        if self._has_vel_prior:
            vp_r0 = self._D_imu + self._D_obs + self._D_anc
            W_vp  = self._vel_prior_weight * torch.eye(3, device=device, dtype=dtype)
            W[:, vp_r0:vp_r0+3, vp_r0:vp_r0+3] = W_vp.expand(B, 3, 3)
        return W

    def _apply_step(self, dx: torch.Tensor) -> None:
        """Apply state update: rotation gets left perturbation, velocity additive."""
        N = self._N
        delta_phi = dx[:, :3 * N].reshape(self._B, N, 3)
        delta_v   = dx[:, 3 * N:].reshape(self._B, N, 3)
        with torch.no_grad():
            # Left perturbation: R_k <- Exp(delta_phi_k) @ R_k
            new_R = pp.so3(delta_phi).Exp() @ pp.SO3(self._rot)
            self._rot.data.copy_(new_R.tensor())
            self._vel.data.add_(delta_v)

    @torch.no_grad()
    def step(self) -> dict:
        """Solve H dx = -g in fp64, then cast the update to the working dtype."""
        r = self.forward()                  # (B, D)
        J = self.jacobian()                 # (B, D, 6N)
        W = self.weight_matrix()            # (B, D, D)

        work_dtype = r.dtype
        # Normal equations in fp64: cond(H) reaches ~1e13 (anchor 1e6, stiff bias walk); fp32 Cholesky passes but GN diverges.
        J64, W64, r64 = J.double(), W.double(), r.double()

        JT_W = J64.transpose(-1, -2) @ W64              # (B, 6N, D)
        H    = JT_W @ J64                                # (B, 6N, 6N)
        H    = 0.5 * (H + H.transpose(-1, -2))          # symmetrize
        g    = (JT_W @ r64.unsqueeze(-1)).squeeze(-1)   # (B, 6N)

        loss = (r64 * (W64 @ r64.unsqueeze(-1)).squeeze(-1)).sum(dim=-1)  # (B,)

        # Per-window: a batched cholesky() raises for the whole batch on one
        # non-PD or NaN H, zeroing dx for every window (target == prior).
        L, info = torch.linalg.cholesky_ex(H)
        good = info == 0                                 # (B,)
        dx   = torch.zeros_like(g)
        if good.any():
            dx[good] = torch.cholesky_solve(-g[good].unsqueeze(-1), L[good]).squeeze(-1)

        dx = dx.to(work_dtype)
        self._apply_step(dx)

        return {
            "loss":        loss.mean().item(),
            "step_norm":   dx.abs().max().item(),
            "bad":         ~good,                        # (B,) per-window failure
        }

    @torch.no_grad()
    def _reject_obs_outliers(self) -> int:
        """Re-score all observations; return total rejections if the mask changed, else 0."""
        if self._obs_reject_chi2 is None:
            return 0
        d2 = mahalanobis_d2(self._residual_obs(), _safe_cov_inv(self.net_vel_cov_b))
        keep, n_rejected = rejection_mask(
            d2, self._obs_reject_chi2, n_rejected_prev=self._n_rejected,
            reject_all=self._obs_reject_all,
        )
        changed = not torch.equal(keep, self._obs_mask)
        self._obs_mask, self._n_rejected = keep, n_rejected
        return int(n_rejected.sum()) if changed else 0

    @torch.no_grad()
    def _gn_loop(self) -> tuple[list[float], bool, torch.Tensor]:
        """Run GN until convergence or max_iter. Returns (history, converged, failed)."""
        history = []
        prev_loss = None
        converged = False
        failed = torch.zeros(self._B, dtype=torch.bool, device=self._vel.device)
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
        """Run GN, with bounded chi-square rejection/re-solve rounds when configured."""
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
            "failed":     failed,                        # (B,) windows with any factorization failure
            "rejected":   int((self._obs_mask == 0).sum().item()),
        }

    @property
    def get_state(self) -> jt.Float[torch.Tensor, "B N 3"]:
        """Current optimized body-frame velocity state, detached."""
        return self._vel.detach()

    @property
    def get_rotation(self) -> jt.Float[pp.LieTensor, "B N 4"]:
        """Current optimized SO(3) rotation state (world<-body), detached."""
        return pp.SO3(self._rot.detach())
