#!/usr/bin/env python3
"""Offline benchmark for simple excavator actuator models.

This script reads logs from ``data_collection/drive_logger.py`` and compares a
small set of predictive models for the three hydraulic arm axes
``[boom, arm, bucket]``.

Always included:

- hold_velocity: ``qdot[k+1] = qdot[k]``
- linear_mimo: ``qdot[k+1] = B u[k] + b``
- quadratic_self: per-joint ``a q^2 u + b q u + c u + d``
- lagged_quadratic_self: per-joint ``e qdot + a q^2 u + b q u + c u + d``

Conditionally included when the training data is rich enough:

- pressure_quadratic_self: adds ``f (pext - pret) u`` term
- hammerstein_linear_mimo: per-channel deadband transform + linear MIMO map

The benchmark reports one-step velocity prediction metrics and an open-loop
joint-angle rollout metric on held-out data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd


TIME_COL = "timestamp"
SEGMENT_COL = "segment_id"

CMD_COLS = ["combined_cmd_lift", "combined_cmd_tilt", "combined_cmd_scoop"]
POS_COLS = ["joint_pos_boom", "joint_pos_arm", "joint_pos_bucket"]
VEL_COLS = ["joint_vel_boom", "joint_vel_arm", "joint_vel_bucket"]
PEXT_COLS = ["pext_lift", "pext_tilt", "pext_scoop"]
PRET_COLS = ["pret_lift", "pret_tilt", "pret_scoop"]
JOINT_NAMES = ["boom", "arm", "bucket"]

LEGACY_CMD_COLS = ["valve_cmd_0", "valve_cmd_1", "valve_cmd_2"]
LEGACY_POS_COLS = ["joint_pos_0", "joint_pos_1", "joint_pos_2"]
LEGACY_VEL_COLS = ["joint_vel_0", "joint_vel_1", "joint_vel_2"]


def _resolve_inputs(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_file() and p.suffix.lower() == ".csv":
            files.append(p.resolve())
        elif p.is_dir():
            files.extend(sorted(x.resolve() for x in p.glob("*.csv")))
        else:
            files.extend(sorted(x.resolve() for x in Path().glob(item)))

    seen = set()
    unique: List[Path] = []
    for p in files:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _has_all(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(col in df.columns for col in cols)


def _normalize_drive_log_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported log schemas to the current drive_logger layout.

    The primary target is ``data_collection/drive_logger.py``. A compact legacy
    three-joint logger schema is also accepted so the benchmark can be exercised
    on old sample files already present in the repo.
    """

    df = df.copy()

    if _has_all(df, CMD_COLS + POS_COLS + VEL_COLS):
        return df

    if _has_all(df, LEGACY_CMD_COLS + LEGACY_POS_COLS + LEGACY_VEL_COLS):
        rename_map = {
            "valve_cmd_0": "combined_cmd_lift",
            "valve_cmd_1": "combined_cmd_tilt",
            "valve_cmd_2": "combined_cmd_scoop",
            "joint_pos_0": "joint_pos_boom",
            "joint_pos_1": "joint_pos_arm",
            "joint_pos_2": "joint_pos_bucket",
            "joint_vel_0": "joint_vel_boom",
            "joint_vel_1": "joint_vel_arm",
            "joint_vel_2": "joint_vel_bucket",
        }
        df = df.rename(columns=rename_map)

        # Legacy sample logs already store radians and rad/s, which is the unit
        # system the benchmark and the Isaac training pipeline both use.
        for col in POS_COLS + VEL_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if SEGMENT_COL not in df.columns:
            df[SEGMENT_COL] = 0
        if "cmd_stale" not in df.columns:
            df["cmd_stale"] = 0
        if "pressure_stale" not in df.columns:
            df["pressure_stale"] = 0
        return df

    required = CMD_COLS + POS_COLS + VEL_COLS
    raise ValueError(
        "Unsupported CSV schema. Expected drive_logger columns "
        f"{required} or legacy columns {LEGACY_CMD_COLS + LEGACY_POS_COLS + LEGACY_VEL_COLS}."
    )


def _split_into_segments(df: pd.DataFrame, *, expected_dt: float, gap_factor: float) -> list[pd.DataFrame]:
    if SEGMENT_COL in df.columns:
        segments = []
        for _, seg in df.groupby(SEGMENT_COL, sort=True):
            seg = seg.sort_values(TIME_COL)
            if len(seg) >= 2:
                segments.append(seg.copy())
        return segments

    df = df.sort_values(TIME_COL)
    t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(np.float64)
    gaps = np.diff(t)
    split_idx = np.where((gaps <= 0.0) | (gaps > gap_factor * expected_dt))[0] + 1
    chunks = np.split(df, split_idx)
    return [c.copy() for c in chunks if len(c) >= 2]


def _sanitize_segment(seg: pd.DataFrame, *, min_rows: int) -> Optional[pd.DataFrame]:
    seg = seg.copy()
    seg = seg.replace([np.inf, -np.inf], np.nan)
    seg = seg.dropna(subset=[TIME_COL])
    seg = seg.sort_values(TIME_COL).drop_duplicates(TIME_COL, keep="first")

    if "cmd_stale" in seg.columns:
        cmd_stale = pd.to_numeric(seg["cmd_stale"], errors="coerce").fillna(0.0)
        seg = seg.loc[cmd_stale <= 0.5].copy()

    seg = seg.dropna(subset=CMD_COLS + POS_COLS + VEL_COLS)
    if len(seg) < min_rows:
        return None

    seg["_dt_next"] = pd.to_numeric(seg[TIME_COL], errors="coerce").shift(-1) - pd.to_numeric(seg[TIME_COL], errors="coerce")
    seg["_source_file"] = seg.get("_source_file", "")
    return seg


def _prepare_segment(seg: pd.DataFrame) -> pd.DataFrame:
    seg = seg.copy()
    for col in PEXT_COLS + PRET_COLS:
        if col not in seg.columns:
            seg[col] = np.nan
    seg["dp_lift"] = pd.to_numeric(seg[PEXT_COLS[0]], errors="coerce") - pd.to_numeric(seg[PRET_COLS[0]], errors="coerce")
    seg["dp_tilt"] = pd.to_numeric(seg[PEXT_COLS[1]], errors="coerce") - pd.to_numeric(seg[PRET_COLS[1]], errors="coerce")
    seg["dp_scoop"] = pd.to_numeric(seg[PEXT_COLS[2]], errors="coerce") - pd.to_numeric(seg[PRET_COLS[2]], errors="coerce")
    seg["y_next_boom"] = pd.to_numeric(seg[VEL_COLS[0]], errors="coerce").shift(-1)
    seg["y_next_arm"] = pd.to_numeric(seg[VEL_COLS[1]], errors="coerce").shift(-1)
    seg["y_next_bucket"] = pd.to_numeric(seg[VEL_COLS[2]], errors="coerce").shift(-1)
    seg = seg.iloc[:-1].copy()
    seg = seg.dropna(subset=["y_next_boom", "y_next_arm", "y_next_bucket", "_dt_next"])
    return seg


def _concat_prepared(segments: list[pd.DataFrame]) -> pd.DataFrame:
    prepared = [_prepare_segment(seg) for seg in segments]
    prepared = [seg for seg in prepared if len(seg) > 0]
    if not prepared:
        return pd.DataFrame()
    return pd.concat(prepared, axis=0, ignore_index=True)


def _fit_lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), rcond=None)
    return coef


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred), axis=0)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    mean = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true - mean) ** 2, axis=0)
    out = np.zeros(y_true.shape[1], dtype=np.float64)
    valid = ss_tot > 1e-12
    out[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    out[~valid] = np.nan
    return out


def _active_counts(frame: pd.DataFrame, threshold: float = 0.1) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for cmd_col, joint in zip(CMD_COLS, JOINT_NAMES):
        u = pd.to_numeric(frame[cmd_col], errors="coerce").to_numpy(np.float64)
        out[joint] = {
            "positive": int((u > threshold).sum()),
            "negative": int((u < -threshold).sum()),
        }
    return out


def _pressure_valid_fraction(frame: pd.DataFrame) -> float:
    cols = PEXT_COLS + PRET_COLS
    if not _has_all(frame, cols):
        return 0.0
    valid = np.isfinite(frame[cols].to_numpy(np.float64)).all(axis=1)
    return float(valid.mean()) if len(valid) > 0 else 0.0


def _segment_train_test_split(segments: list[pd.DataFrame], test_ratio: float) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    if not segments:
        return [], []
    if len(segments) == 1:
        seg = segments[0]
        cut = max(2, min(len(seg) - 2, int(round(len(seg) * (1.0 - test_ratio)))))
        return [seg.iloc[:cut].copy()], [seg.iloc[cut:].copy()]

    n_test = max(1, int(round(len(segments) * test_ratio)))
    n_test = min(n_test, len(segments) - 1)
    return segments[:-n_test], segments[-n_test:]


@dataclass
class BenchmarkResult:
    name: str
    enabled: bool
    reason: str
    metrics: dict[str, Any]
    artifacts: dict[str, Any]


class BaseModel:
    name = "base"

    def can_fit(self, train_frame: pd.DataFrame) -> tuple[bool, str]:
        return True, "ok"

    def fit(self, train_frame: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict_batch(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_state(self, q: np.ndarray, qdot: np.ndarray, u: np.ndarray, dp: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def artifacts(self) -> dict[str, Any]:
        return {}


class HoldVelocityModel(BaseModel):
    name = "hold_velocity"

    def fit(self, train_frame: pd.DataFrame) -> None:
        return None

    def predict_batch(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[VEL_COLS].to_numpy(np.float64)

    def predict_state(self, q: np.ndarray, qdot: np.ndarray, u: np.ndarray, dp: np.ndarray) -> np.ndarray:
        return np.asarray(qdot, dtype=np.float64)


class LinearMIMOModel(BaseModel):
    name = "linear_mimo"

    def __init__(self) -> None:
        self.coef = np.zeros((4, 3), dtype=np.float64)

    def fit(self, train_frame: pd.DataFrame) -> None:
        u = train_frame[CMD_COLS].to_numpy(np.float64)
        X = np.column_stack([np.ones(len(u), dtype=np.float64), u])
        y = train_frame[["y_next_boom", "y_next_arm", "y_next_bucket"]].to_numpy(np.float64)
        self.coef = _fit_lstsq(X, y)

    def predict_batch(self, frame: pd.DataFrame) -> np.ndarray:
        u = frame[CMD_COLS].to_numpy(np.float64)
        X = np.column_stack([np.ones(len(u), dtype=np.float64), u])
        return X @ self.coef

    def predict_state(self, q: np.ndarray, qdot: np.ndarray, u: np.ndarray, dp: np.ndarray) -> np.ndarray:
        X = np.concatenate([[1.0], np.asarray(u, dtype=np.float64)])
        return X @ self.coef

    def artifacts(self) -> dict[str, Any]:
        return {"coef": self.coef.tolist(), "feature_order": ["bias"] + CMD_COLS}


class QuadraticSelfModel(BaseModel):
    name = "quadratic_self"

    def __init__(self) -> None:
        self.coef = np.zeros((3, 4), dtype=np.float64)

    def _features(self, q: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.column_stack([
            np.ones(len(q), dtype=np.float64),
            u,
            q * u,
            (q ** 2) * u,
        ])

    def fit(self, train_frame: pd.DataFrame) -> None:
        q = train_frame[POS_COLS].to_numpy(np.float64)
        u = train_frame[CMD_COLS].to_numpy(np.float64)
        y = train_frame[["y_next_boom", "y_next_arm", "y_next_bucket"]].to_numpy(np.float64)
        coef = []
        for j in range(3):
            Xj = self._features(q[:, j], u[:, j])
            coef.append(_fit_lstsq(Xj, y[:, j]))
        self.coef = np.vstack(coef)

    def predict_batch(self, frame: pd.DataFrame) -> np.ndarray:
        q = frame[POS_COLS].to_numpy(np.float64)
        u = frame[CMD_COLS].to_numpy(np.float64)
        out = np.zeros((len(frame), 3), dtype=np.float64)
        for j in range(3):
            out[:, j] = self._features(q[:, j], u[:, j]) @ self.coef[j]
        return out

    def predict_state(self, q: np.ndarray, qdot: np.ndarray, u: np.ndarray, dp: np.ndarray) -> np.ndarray:
        out = np.zeros(3, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        for j in range(3):
            feat = np.array([1.0, u[j], q[j] * u[j], (q[j] ** 2) * u[j]], dtype=np.float64)
            out[j] = feat @ self.coef[j]
        return out

    def artifacts(self) -> dict[str, Any]:
        rows = {}
        for joint, beta in zip(JOINT_NAMES, self.coef):
            rows[joint] = {
                "bias": float(beta[0]),
                "u": float(beta[1]),
                "q_u": float(beta[2]),
                "q2_u": float(beta[3]),
            }
        return {"per_joint_formula": rows}


class LaggedQuadraticSelfModel(BaseModel):
    name = "lagged_quadratic_self"

    def __init__(self) -> None:
        self.coef = np.zeros((3, 5), dtype=np.float64)

    def _features(self, q: np.ndarray, qdot: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.column_stack([
            np.ones(len(q), dtype=np.float64),
            qdot,
            u,
            q * u,
            (q ** 2) * u,
        ])

    def fit(self, train_frame: pd.DataFrame) -> None:
        q = train_frame[POS_COLS].to_numpy(np.float64)
        qdot = train_frame[VEL_COLS].to_numpy(np.float64)
        u = train_frame[CMD_COLS].to_numpy(np.float64)
        y = train_frame[["y_next_boom", "y_next_arm", "y_next_bucket"]].to_numpy(np.float64)
        coef = []
        for j in range(3):
            Xj = self._features(q[:, j], qdot[:, j], u[:, j])
            coef.append(_fit_lstsq(Xj, y[:, j]))
        self.coef = np.vstack(coef)

    def predict_batch(self, frame: pd.DataFrame) -> np.ndarray:
        q = frame[POS_COLS].to_numpy(np.float64)
        qdot = frame[VEL_COLS].to_numpy(np.float64)
        u = frame[CMD_COLS].to_numpy(np.float64)
        out = np.zeros((len(frame), 3), dtype=np.float64)
        for j in range(3):
            out[:, j] = self._features(q[:, j], qdot[:, j], u[:, j]) @ self.coef[j]
        return out

    def predict_state(self, q: np.ndarray, qdot: np.ndarray, u: np.ndarray, dp: np.ndarray) -> np.ndarray:
        out = np.zeros(3, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        qdot = np.asarray(qdot, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        for j in range(3):
            feat = np.array([1.0, qdot[j], u[j], q[j] * u[j], (q[j] ** 2) * u[j]], dtype=np.float64)
            out[j] = feat @ self.coef[j]
        return out

    def artifacts(self) -> dict[str, Any]:
        rows = {}
        for joint, beta in zip(JOINT_NAMES, self.coef):
            rows[joint] = {
                "bias": float(beta[0]),
                "qdot": float(beta[1]),
                "u": float(beta[2]),
                "q_u": float(beta[3]),
                "q2_u": float(beta[4]),
            }
        return {"per_joint_formula": rows}


class PressureQuadraticSelfModel(BaseModel):
    name = "pressure_quadratic_self"

    def __init__(self, min_valid_frac: float = 0.6) -> None:
        self.min_valid_frac = float(min_valid_frac)
        self.coef = np.zeros((3, 5), dtype=np.float64)

    def can_fit(self, train_frame: pd.DataFrame) -> tuple[bool, str]:
        valid_frac = _pressure_valid_fraction(train_frame)
        if valid_frac < self.min_valid_frac:
            return False, f"pressure valid fraction {valid_frac:.2f} < {self.min_valid_frac:.2f}"
        return True, "ok"

    def _features(self, q: np.ndarray, u: np.ndarray, dp: np.ndarray) -> np.ndarray:
        return np.column_stack([
            np.ones(len(q), dtype=np.float64),
            u,
            q * u,
            (q ** 2) * u,
            dp * u,
        ])

    def fit(self, train_frame: pd.DataFrame) -> None:
        filtered = train_frame.dropna(subset=["dp_lift", "dp_tilt", "dp_scoop"]).copy()
        q = filtered[POS_COLS].to_numpy(np.float64)
        u = filtered[CMD_COLS].to_numpy(np.float64)
        dp = filtered[["dp_lift", "dp_tilt", "dp_scoop"]].to_numpy(np.float64)
        y = filtered[["y_next_boom", "y_next_arm", "y_next_bucket"]].to_numpy(np.float64)
        coef = []
        for j in range(3):
            Xj = self._features(q[:, j], u[:, j], dp[:, j])
            coef.append(_fit_lstsq(Xj, y[:, j]))
        self.coef = np.vstack(coef)

    def predict_batch(self, frame: pd.DataFrame) -> np.ndarray:
        q = frame[POS_COLS].to_numpy(np.float64)
        u = frame[CMD_COLS].to_numpy(np.float64)
        dp = frame[["dp_lift", "dp_tilt", "dp_scoop"]].to_numpy(np.float64)
        out = np.zeros((len(frame), 3), dtype=np.float64)
        for j in range(3):
            out[:, j] = self._features(q[:, j], u[:, j], dp[:, j]) @ self.coef[j]
        return out

    def predict_state(self, q: np.ndarray, qdot: np.ndarray, u: np.ndarray, dp: np.ndarray) -> np.ndarray:
        out = np.zeros(3, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        dp = np.asarray(dp, dtype=np.float64)
        for j in range(3):
            feat = np.array([1.0, u[j], q[j] * u[j], (q[j] ** 2) * u[j], dp[j] * u[j]], dtype=np.float64)
            out[j] = feat @ self.coef[j]
        return out

    def artifacts(self) -> dict[str, Any]:
        rows = {}
        for joint, beta in zip(JOINT_NAMES, self.coef):
            rows[joint] = {
                "bias": float(beta[0]),
                "u": float(beta[1]),
                "q_u": float(beta[2]),
                "q2_u": float(beta[3]),
                "dp_u": float(beta[4]),
            }
        return {"per_joint_formula": rows}


class HammersteinLinearMIMOModel(BaseModel):
    name = "hammerstein_linear_mimo"

    def __init__(self, min_active_per_sign: int = 50, max_deadband: float = 0.35) -> None:
        self.min_active_per_sign = int(min_active_per_sign)
        self.max_deadband = float(max_deadband)
        self.deadbands = np.zeros(3, dtype=np.float64)
        self.coef = np.zeros((4, 3), dtype=np.float64)

    def can_fit(self, train_frame: pd.DataFrame) -> tuple[bool, str]:
        counts = _active_counts(train_frame)
        for joint in JOINT_NAMES:
            if counts[joint]["positive"] < self.min_active_per_sign or counts[joint]["negative"] < self.min_active_per_sign:
                return False, (
                    f"{joint} active samples too low for deadband estimation "
                    f"(+{counts[joint]['positive']}, -{counts[joint]['negative']})"
                )
        return True, "ok"

    def _transform(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        db = self.deadbands.reshape(1, 3) if u.ndim == 2 else self.deadbands
        return np.sign(u) * np.maximum(0.0, np.abs(u) - db)

    def _estimate_deadband(self, u: np.ndarray, y: np.ndarray) -> float:
        best_db = 0.0
        best_mse = float("inf")
        for db in np.linspace(0.0, self.max_deadband, 36):
            z = np.sign(u) * np.maximum(0.0, np.abs(u) - db)
            if np.count_nonzero(np.abs(z) > 1e-6) < self.min_active_per_sign:
                continue
            X = np.column_stack([np.ones(len(z), dtype=np.float64), z])
            beta = _fit_lstsq(X, y)
            mse = float(np.mean((X @ beta - y) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_db = float(db)
        return best_db

    def fit(self, train_frame: pd.DataFrame) -> None:
        u = train_frame[CMD_COLS].to_numpy(np.float64)
        y = train_frame[["y_next_boom", "y_next_arm", "y_next_bucket"]].to_numpy(np.float64)
        for j in range(3):
            self.deadbands[j] = self._estimate_deadband(u[:, j], y[:, j])
        z = self._transform(u)
        X = np.column_stack([np.ones(len(z), dtype=np.float64), z])
        self.coef = _fit_lstsq(X, y)

    def predict_batch(self, frame: pd.DataFrame) -> np.ndarray:
        z = self._transform(frame[CMD_COLS].to_numpy(np.float64))
        X = np.column_stack([np.ones(len(z), dtype=np.float64), z])
        return X @ self.coef

    def predict_state(self, q: np.ndarray, qdot: np.ndarray, u: np.ndarray, dp: np.ndarray) -> np.ndarray:
        z = np.sign(u) * np.maximum(0.0, np.abs(u) - self.deadbands)
        X = np.concatenate([[1.0], z.astype(np.float64)])
        return X @ self.coef

    def artifacts(self) -> dict[str, Any]:
        return {
            "deadbands": {joint: float(db) for joint, db in zip(JOINT_NAMES, self.deadbands)},
            "coef": self.coef.tolist(),
            "feature_order": ["bias", "z_lift", "z_tilt", "z_scoop"],
        }


def _evaluate_one_step(model: BaseModel, frame: pd.DataFrame) -> dict[str, Any]:
    y_true = frame[["y_next_boom", "y_next_arm", "y_next_bucket"]].to_numpy(np.float64)
    y_pred = model.predict_batch(frame)
    rmse = _rmse(y_true, y_pred)
    mae = _mae(y_true, y_pred)
    r2 = _r2(y_true, y_pred)
    return {
        "rmse_mean": float(np.nanmean(rmse)),
        "mae_mean": float(np.nanmean(mae)),
        "r2_mean": float(np.nanmean(r2)),
        "rmse_per_joint": {joint: float(val) for joint, val in zip(JOINT_NAMES, rmse)},
        "mae_per_joint": {joint: float(val) for joint, val in zip(JOINT_NAMES, mae)},
        "r2_per_joint": {joint: float(val) for joint, val in zip(JOINT_NAMES, r2)},
    }


def _evaluate_rollout(model: BaseModel, segments: list[pd.DataFrame]) -> dict[str, Any]:
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    used_segments = 0

    for raw_seg in segments:
        seg = _prepare_segment(raw_seg)
        if len(seg) < 2:
            continue

        q_actual = seg[POS_COLS].to_numpy(np.float64)
        qdot_actual = seg[VEL_COLS].to_numpy(np.float64)
        u = seg[CMD_COLS].to_numpy(np.float64)
        dp = seg[["dp_lift", "dp_tilt", "dp_scoop"]].to_numpy(np.float64)
        dt = seg["_dt_next"].to_numpy(np.float64)

        q_pred = np.zeros_like(q_actual)
        q_pred[0] = q_actual[0]
        qdot_state = qdot_actual[0].copy()

        for k in range(len(seg) - 1):
            pred_next_qdot = model.predict_state(q_pred[k], qdot_state, u[k], dp[k])
            q_pred[k + 1] = q_pred[k] + dt[k] * pred_next_qdot
            qdot_state = pred_next_qdot

        all_true.append(q_actual[1:])
        all_pred.append(q_pred[1:])
        used_segments += 1

    if not all_true:
        return {"rmse_mean": float("nan"), "rmse_per_joint": {}, "segments": 0}

    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)
    rmse = _rmse(y_true, y_pred)
    mae = _mae(y_true, y_pred)
    return {
        "rmse_mean": float(np.nanmean(rmse)),
        "mae_mean": float(np.nanmean(mae)),
        "rmse_per_joint": {joint: float(val) for joint, val in zip(JOINT_NAMES, rmse)},
        "mae_per_joint": {joint: float(val) for joint, val in zip(JOINT_NAMES, mae)},
        "segments": int(used_segments),
    }


def _run_model(model: BaseModel, train_frame: pd.DataFrame, test_frame: pd.DataFrame, test_segments: list[pd.DataFrame]) -> BenchmarkResult:
    ok, reason = model.can_fit(train_frame)
    if not ok:
        return BenchmarkResult(name=model.name, enabled=False, reason=reason, metrics={}, artifacts={})

    model.fit(train_frame)
    metrics = {
        "one_step": _evaluate_one_step(model, test_frame),
        "rollout": _evaluate_rollout(model, test_segments),
    }
    return BenchmarkResult(name=model.name, enabled=True, reason="ok", metrics=metrics, artifacts=model.artifacts())


def _results_to_score_rows(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        if not result.enabled:
            rows.append({
                "model": result.name,
                "enabled": False,
                "reason": result.reason,
                "one_step_rmse_mean": np.nan,
                "one_step_mae_mean": np.nan,
                "one_step_r2_mean": np.nan,
                "rollout_rmse_mean": np.nan,
                "rollout_mae_mean": np.nan,
            })
            continue
        rows.append({
            "model": result.name,
            "enabled": True,
            "reason": result.reason,
            "one_step_rmse_mean": result.metrics["one_step"]["rmse_mean"],
            "one_step_mae_mean": result.metrics["one_step"]["mae_mean"],
            "one_step_r2_mean": result.metrics["one_step"]["r2_mean"],
            "rollout_rmse_mean": result.metrics["rollout"]["rmse_mean"],
            "rollout_mae_mean": result.metrics["rollout"]["mae_mean"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark simple excavator actuator models on drive logs")
    parser.add_argument("--input", nargs="+", required=True, help="Training CSV file(s), directory, or glob pattern(s)")
    parser.add_argument("--test-input", nargs="*", default=None, help="Optional held-out benchmark CSVs; if omitted, uses an internal split")
    parser.add_argument("--out", default="data_collection/model_benchmarks", help="Output directory")
    parser.add_argument("--dt", type=float, default=0.01, help="Expected sample period for gap splitting")
    parser.add_argument("--gap-factor", type=float, default=3.0, help="Split when timestamp gap exceeds dt*gap_factor")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Internal hold-out ratio when --test-input is omitted")
    parser.add_argument("--min-segment-sec", type=float, default=3.0, help="Drop segments shorter than this")
    parser.add_argument("--pressure-min-valid-frac", type=float, default=0.6, help="Minimum valid pressure fraction for pressure model")
    parser.add_argument("--hammerstein-min-active", type=int, default=50, help="Minimum positive and negative active samples per joint for Hammerstein deadband fit")
    args = parser.parse_args()

    train_files = _resolve_inputs(args.input)
    if not train_files:
        raise FileNotFoundError("No input CSV files found")

    test_files = _resolve_inputs(args.test_input or [])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    min_rows = max(3, int(round(float(args.min_segment_sec) / max(float(args.dt), 1e-6))))

    def load_segments(files: list[Path]) -> list[pd.DataFrame]:
        segments: list[pd.DataFrame] = []
        for file_path in files:
            df = pd.read_csv(file_path)
            df = _normalize_drive_log_schema(df)
            df["_source_file"] = str(file_path)
            raw_segments = _split_into_segments(df, expected_dt=float(args.dt), gap_factor=float(args.gap_factor))
            for seg in raw_segments:
                seg["_source_file"] = str(file_path)
                clean = _sanitize_segment(seg, min_rows=min_rows)
                if clean is not None and len(clean) >= min_rows:
                    segments.append(clean)
        return segments

    train_segments = load_segments(train_files)
    if not train_segments:
        raise RuntimeError("No usable training segments found after filtering")

    if test_files:
        test_segments = load_segments(test_files)
        if not test_segments:
            raise RuntimeError("No usable test segments found after filtering")
    else:
        train_segments, test_segments = _segment_train_test_split(train_segments, float(args.test_ratio))
        if not train_segments or not test_segments:
            raise RuntimeError("Could not form train/test split from the provided segments")

    train_frame = _concat_prepared(train_segments)
    test_frame = _concat_prepared(test_segments)
    if len(train_frame) == 0 or len(test_frame) == 0:
        raise RuntimeError("Prepared train/test data is empty after segment preprocessing")

    models: list[BaseModel] = [
        HoldVelocityModel(),
        LinearMIMOModel(),
        QuadraticSelfModel(),
        LaggedQuadraticSelfModel(),
        PressureQuadraticSelfModel(min_valid_frac=float(args.pressure_min_valid_frac)),
        HammersteinLinearMIMOModel(min_active_per_sign=int(args.hammerstein_min_active)),
    ]

    results = [_run_model(model, train_frame, test_frame, test_segments) for model in models]
    score_rows = _results_to_score_rows(results)
    score_df = pd.DataFrame(score_rows)
    score_df.to_csv(out_dir / "model_scores.csv", index=False)

    payload = {
        "config": {
            "input": [str(p) for p in train_files],
            "test_input": [str(p) for p in test_files],
            "dt": float(args.dt),
            "gap_factor": float(args.gap_factor),
            "test_ratio": float(args.test_ratio),
            "min_segment_sec": float(args.min_segment_sec),
            "pressure_min_valid_frac": float(args.pressure_min_valid_frac),
            "hammerstein_min_active": int(args.hammerstein_min_active),
            "units": {
                "joint_position": "rad",
                "joint_velocity": "rad/s",
                "command": "normalized_-1_to_1",
            },
        },
        "dataset": {
            "train_segments": int(len(train_segments)),
            "test_segments": int(len(test_segments)),
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "pressure_valid_fraction_train": _pressure_valid_fraction(train_frame),
            "active_counts_train": _active_counts(train_frame),
        },
        "results": [
            {
                "model": result.name,
                "enabled": result.enabled,
                "reason": result.reason,
                "metrics": result.metrics,
                "artifacts": result.artifacts,
            }
            for result in results
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Model benchmark complete")
    print(f"- train segments: {len(train_segments)}")
    print(f"- test segments:  {len(test_segments)}")
    print(f"- outputs:        {out_dir}")
    print()
    for row in score_rows:
        if not row["enabled"]:
            print(f"{row['model']:<24} skipped ({row['reason']})")
            continue
        print(
            f"{row['model']:<24} "
            f"one-step rmse={row['one_step_rmse_mean']:.3f} "
            f"mae={row['one_step_mae_mean']:.3f} "
            f"r2={row['one_step_r2_mean']:.3f} | "
            f"rollout rmse={row['rollout_rmse_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
