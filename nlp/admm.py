from __future__ import annotations

import multiprocessing as mp
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

Array = np.ndarray


# ------------------------------- utilities -------------------------------- #


def soft_threshold(v: Array, tau: float) -> Array:
    """Elementwise soft-thresholding: prox_{tau ||·||_1}(v)."""
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


def _sqnorm(x: Array) -> float:
    v = float(np.dot(x.ravel(), x.ravel()))
    return v if np.isfinite(v) else np.inf


def _nan_guard(*xs: Array) -> bool:
    """Return True if any array contains NaN or Inf."""
    for x in xs:
        if x is None or (isinstance(x, np.ndarray) and x.size == 0):
            continue
        if not np.isfinite(x).all():
            return True
    return False


def _robust_median_aggregation(
    values: List[Array], weights: Optional[Array] = None
) -> Array:
    """Coordinate-wise median (weights ignored by design)."""
    stacked = np.stack(values, axis=0)
    return np.median(stacked, axis=0)


def _trimmed_mean_aggregation(
    values: List[Array], weights: Optional[Array], trim_ratio: float = 0.1
) -> Array:
    """
    Coordinate-wise unweighted trimmed mean.
    We intentionally ignore 'weights' because per-coordinate sorting reorders sites differently.
    """
    stacked = np.stack(values, axis=0)  # (K, ...)
    K = stacked.shape[0]
    n_trim = int(K * trim_ratio)
    if n_trim == 0 or 2 * n_trim >= K:
        # Fallback to median if trimming would remove too much
        return np.median(stacked, axis=0)
    sorted_vals = np.sort(stacked, axis=0)
    trimmed = sorted_vals[n_trim : K - n_trim]
    return np.mean(trimmed, axis=0)


# ----------------------------- logging structs ---------------------------- #


@dataclass
class History:
    r_norm: List[float]
    s_norm: List[float]
    eps_pri: List[float]
    eps_dual: List[float]
    rho: List[float]
    obj: List[float]
    timing: Dict[str, List[float]]
    iters: int = 0
    converged: bool = False
    reason: str = ""
    # Optional relative residuals (added)
    rel_r: List[float] = None
    rel_s: List[float] = None

    def __post_init__(self):
        if not hasattr(self, "timing") or self.timing is None:
            self.timing = {
                "local_updates": [],
                "global_updates": [],
                "communication": [],
            }
        if self.rel_r is None:
            self.rel_r = []
        if self.rel_s is None:
            self.rel_s = []

    def as_arrays(self) -> Dict[str, Array]:
        return {
            "r_norm": np.asarray(self.r_norm, dtype=float),
            "s_norm": np.asarray(self.s_norm, dtype=float),
            "eps_pri": np.asarray(self.eps_pri, dtype=float),
            "eps_dual": np.asarray(self.eps_dual, dtype=float),
            "rho": np.asarray(self.rho, dtype=float),
            "obj": np.asarray(self.obj, dtype=float),
        }


# --------------------------- Anderson acceleration ------------------------ #


class _Anderson:
    """
    Minimal Anderson acceleration (AA-I) for fixed-point iteration on z.
    Stores last m residuals and points. Works on a concatenated vector; we
    re-pack/unpack dict[str->ndarray] by a stable block order (sorted keys).
    """

    def __init__(self, m: int = 5, reg: float = 1e-8):
        self.m = int(m)
        self.reg = float(reg)
        self.F: List[np.ndarray] = []
        self.G: List[np.ndarray] = []

    def reset(self):
        self.F.clear()
        self.G.clear()

    def update(
        self, g_new: Dict[str, Array], f_new: Dict[str, Array]
    ) -> Dict[str, Array]:
        # Pack
        block_names = sorted(g_new)
        g_flat = np.concatenate([g_new[n].ravel() for n in block_names])
        f_flat = np.concatenate([f_new[n].ravel() for n in block_names])

        self.G.append(g_flat.copy())
        self.F.append(f_flat.copy())
        if len(self.F) > self.m:
            self.F.pop(0)
            self.G.pop(0)

        k = len(self.F)
        if k == 1:
            return g_new

        F = np.column_stack(self.F)  # (D, k)
        FtF = F.T @ F + self.reg * np.eye(k)
        rhs = np.ones(k)
        try:
            c = np.linalg.solve(FtF, rhs)
            c /= c.sum()
        except np.linalg.LinAlgError:
            return g_new

        g_acc = np.zeros_like(g_flat)
        for j in range(k):
            g_acc += c[j] * self.G[j]

        # Unpack
        out: Dict[str, Array] = {}
        offset = 0
        for n in block_names:
            sz = g_new[n].size
            out[n] = g_acc[offset : offset + sz].reshape(g_new[n].shape)
            offset += sz
        return out


# ======================= Enhanced ADMM Framework ========================= #


class FederatedConsensusADMM:
    """
    Enhanced Generic consensus-ADMM for K sites and B consensus blocks.

    Features:
    - Parallelism support (threading/multiprocessing)
    - Robust aggregation (weighted avg, median, trimmed mean)
    - Better convergence detection (early stopping)
    - Timing and profiling
    - Communication compression (top-k) with round trip
    - Asynchronous updates support (optional early completion)
    - Better memory management hooks
    - Checkpointing hooks
    - Block-wise adaptive ρ with hysteresis and clamping
    - Optional Anderson acceleration on z
    - Optional Nesterov-style over-relaxation scheduling
    """

    # ------------------------------ init ---------------------------------- #
    def __init__(
        self,
        rho: Union[float, Dict[str, float]] = 1.0,
        max_iters: int = 300,
        abstol: float = 1e-4,
        reltol: float = 1e-3,
        adaptive_rho: bool = True,
        rho_mu: float = 10.0,
        rho_tau: float = 2.0,
        over_relaxation: float = 1.0,
        use_weights: bool = True,
        # Parallelism options
        parallel_mode: str = "none",  # 'none', 'threading', 'multiprocessing'
        max_workers: Optional[int] = None,
        # Robust aggregation
        aggregation_method: str = "weighted_avg",  # 'weighted_avg', 'median', 'trimmed_mean'
        trim_ratio: float = 0.1,
        # Advanced convergence
        patience: int = 10,
        min_improvement: float = 1e-6,
        convergence_window: int = 5,
        # Communication efficiency
        communication_rounds: int = 1,  # (kept for compatibility; not used internally)
        compression_ratio: Optional[
            float
        ] = None,  # fraction of entries to keep in top-k
        # Asynchronous support
        async_tolerance: float = 0.0,  # fraction of workers allowed to be stale
        # Memory and I/O
        checkpoint_freq: int = 0,  # 0 = no checkpointing (hook only)
        checkpoint_path: Optional[str] = None,
        low_memory_mode: bool = False,
        # Callback / logging
        callback: Optional[
            Callable[[int, "FederatedConsensusADMM", History], None]
        ] = None,
        random_state: int = 0,
        verbose: bool = False,
        # New: extras (rho clamping, hysteresis; Anderson; relax scheduling)
        rho_min: float = 1e-8,
        rho_max: float = 1e8,
        rho_hysteresis: int = 2,  # consecutive triggers required to change rho
        anderson: bool = False,
        anderson_m: int = 5,
        anderson_reg: float = 1e-8,
        relax_schedule: str = "const",  # {"const","nesterov"}
    ):
        # Seed (for any stochastic local updates that subclasses might add)
        np.random.seed(int(random_state))

        # Residual/stop parameters
        self.max_iters = int(max_iters)
        self.abstol = float(abstol)
        self.reltol = float(reltol)

        # ρ handling (scalar or dict per block)
        if isinstance(rho, dict):
            self._rho_is_dict = True
            self.rho: Union[float, Dict[str, float]] = {
                k: float(v) for k, v in rho.items()
            }
        else:
            self._rho_is_dict = False
            self.rho = float(rho)

        self.adaptive_rho = bool(adaptive_rho)
        self.rho_mu = float(rho_mu)
        self.rho_tau = float(rho_tau)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.rho_hysteresis = int(rho_hysteresis)
        self._hyst_pri = 0
        self._hyst_dual = 0

        # Over-relaxation
        self.alpha = float(over_relaxation)
        self.relax_schedule = str(relax_schedule)
        self._theta_k = 1.0  # for nesterov schedule

        # Weights & misc
        self.use_weights = bool(use_weights)
        self.callback = callback
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        # Parallelism
        self.parallel_mode = str(parallel_mode)
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self._executor = None
        self._lock = threading.Lock() if self.parallel_mode == "threading" else None

        # Aggregation
        self.aggregation_method = str(aggregation_method)
        self.trim_ratio = float(trim_ratio)
        if self.aggregation_method in {"median", "trimmed_mean"} and self.verbose:
            print("[info] Robust aggregation ignores per-site weights by design.")

        # Advanced convergence
        self.patience = int(patience)
        self.min_improvement = float(min_improvement)
        self.convergence_window = int(convergence_window)
        self._best_obj = np.inf
        self._patience_counter = 0

        # Communication
        self.communication_rounds = int(communication_rounds)
        self.compression_ratio = compression_ratio
        self.async_tolerance = float(async_tolerance)
        self.async_mode = self.async_tolerance > 0.0

        # Memory / I/O
        self.checkpoint_freq = int(checkpoint_freq)
        self.checkpoint_path = checkpoint_path
        self.low_memory_mode = bool(low_memory_mode)

        # State
        self.blocks_: List[str] = []
        self.z_: Dict[str, Array] = {}
        self.x_local_: List[Dict[str, Array]] = []
        self.u_: List[Dict[str, Array]] = []
        self.weights_: Optional[Array] = None
        self.history_: Optional[History] = None

        # Timing/profiling
        self._timers: Dict[str, float] = {}

        # Anderson acceleration
        self.anderson = bool(anderson)
        self.anderson_m = int(anderson_m)
        self.anderson_reg = float(anderson_reg)
        self._aa = (
            _Anderson(self.anderson_m, self.anderson_reg) if self.anderson else None
        )

    def __enter__(self):
        """Context manager entry: create executor if needed."""
        if self.parallel_mode == "threading":
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        elif self.parallel_mode == "multiprocessing":
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: shutdown executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ------------------- methods expected from subclasses ------------------ #

    def _init_blocks(self, X_list, y_list) -> Dict[str, Array]:
        """Return initial z per block. Shapes define the problem."""
        raise NotImplementedError

    def _local_update(self, i: int, z_minus_u: Dict[str, Array]) -> Dict[str, Array]:
        """Solve site's proximalized local subproblem (for all blocks)."""
        raise NotImplementedError

    def _objective(self, X_list, y_list, x_locals: List[Dict[str, Array]]) -> float:
        """Return current (monitoring) objective value."""
        raise NotImplementedError

    # ----------------------- optional extension points --------------------- #

    def _prox(self, name: str, v: Array) -> Array:
        """Per-block prox on the aggregated vector. Default: identity."""
        return v

    def _prox_arg_scale(self, name: str) -> float:
        """Scalar multiplier if prox needs scaling (e.g., λ/ρ)."""
        return 1.0

    def _rho_for_block(self, name: str) -> float:
        """Block-wise rho (default: global scalar)."""
        return self._get_rho_block(name)

    # -------------------------- rho helpers -------------------------------- #

    def _get_rho_block(self, name: str) -> float:
        return self.rho[name] if self._rho_is_dict else float(self.rho)

    def _set_rho_block(self, name: str, val: float) -> None:
        v = float(np.clip(val, self.rho_min, self.rho_max))
        if self._rho_is_dict:
            self.rho[name] = v
        else:
            self.rho = v

    # ---------------------- aggregation & compression ---------------------- #

    def _aggregate_block(
        self,
        name: str,
        x_locals: List[Dict[str, Array]],
        u_locals: List[Dict[str, Array]],
        weights: Array,
    ) -> Array:
        """Enhanced aggregation with multiple methods."""
        values = [x_locals[i][name] + u_locals[i][name] for i in range(len(x_locals))]

        if self.aggregation_method == "weighted_avg":
            acc = np.zeros_like(values[0])
            for i, val in enumerate(values):
                acc += weights[i] * val
            return acc
        elif self.aggregation_method == "median":
            return _robust_median_aggregation(values)
        elif self.aggregation_method == "trimmed_mean":
            return _trimmed_mean_aggregation(values, weights, self.trim_ratio)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def _compress_topk(self, g: Array, ratio: Optional[float]) -> Tuple[Array, Any]:
        """
        Top-k compression (round trip).
        Returns (packet, meta) where packet is a 2xk array [idx; vals], meta is original shape.
        If ratio is None or >= 1, returns (g, None).
        """
        if ratio is None or ratio >= 1.0:
            return g, None
        flat = g.ravel()
        k = max(1, int(len(flat) * ratio))
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        vals = flat[idx]
        packet = np.stack([idx.astype(np.int64), vals], axis=0)
        return packet, g.shape

    def _decompress_topk(self, packet: Array, meta: Any) -> Array:
        """Inverse of _compress_topk."""
        if meta is None:
            return packet
        idx, vals = packet
        out = np.zeros(int(np.prod(meta)), dtype=vals.dtype)
        out[idx.astype(np.int64)] = vals
        return out.reshape(meta)

    # --------------------------- parallel updates -------------------------- #

    def _parallel_local_updates(
        self,
        X_list: Sequence[Array],
        y_list: Sequence[Optional[Array]],
        z: Dict[str, Array],
    ) -> List[Dict[str, Array]]:
        """Perform local updates in parallel or sequentially."""
        K = len(X_list)

        if self.parallel_mode == "none" or self._executor is None:
            return [
                self._local_update(
                    i, {name: z[name] - self.u_[i][name] for name in self.blocks_}
                )
                for i in range(K)
            ]

        def update_site(i: int):
            z_minus_u = {name: z[name] - self.u_[i][name] for name in self.blocks_}
            return i, self._local_update(i, z_minus_u)

        futures = [self._executor.submit(update_site, i) for i in range(K)]
        results: List[Optional[Dict[str, Array]]] = [None] * K
        completed_count = 0
        min_required = (
            max(1, int(K * (1 - self.async_tolerance))) if self.async_mode else K
        )

        for future in as_completed(futures):
            try:
                i, x_i = future.result()
                results[i] = x_i
                completed_count += 1
                if self.async_mode and completed_count >= min_required:
                    break
            except Exception as e:
                if self.verbose:
                    print(f"Local update failed at a site: {e}")

        # Fill missing with previous or zeros
        for i in range(K):
            if results[i] is None:
                if hasattr(self, "x_local_") and i < len(self.x_local_):
                    results[i] = {k: v.copy() for k, v in self.x_local_[i].items()}
                else:
                    results[i] = {name: np.zeros_like(z[name]) for name in self.blocks_}

        return results  # type: ignore

    # ----------------------- enhanced convergence -------------------------- #

    def _check_convergence(
        self,
        hist: History,
        r_norm: float,
        s_norm: float,
        eps_pri: float,
        eps_dual: float,
        obj_val: float,
    ) -> Tuple[bool, str]:
        """Enhanced convergence detection with multiple criteria."""
        # Standard ADMM stopping
        if (r_norm <= eps_pri) and (s_norm <= eps_dual):
            return True, "primal & dual residual tolerances met"

        # Early stopping on objective plateau
        if len(hist.obj) >= self.convergence_window:
            if abs(obj_val - self._best_obj) < self.min_improvement:
                self._patience_counter += 1
                if self._patience_counter >= self.patience:
                    return (
                        True,
                        f"early stopping: no improvement for {self.patience} iterations",
                    )
            else:
                self._patience_counter = 0

        if obj_val < self._best_obj:
            self._best_obj = obj_val

        return False, ""

    # ------------------------------ checkpoints ---------------------------- #

    def _save_checkpoint(self, iteration: int):
        """Save checkpoint to disk (hook)."""
        if self.checkpoint_freq <= 0 or self.checkpoint_path is None:
            return
        if iteration % self.checkpoint_freq != 0:
            return

        checkpoint = {
            "iteration": iteration,
            "rho": self.rho,
            "blocks": self.blocks_,
            "z": self.z_,
            "x_local": self.x_local_,
            "u": self.u_,
            "weights": self.weights_,
            "history": self.history_,
            "best_obj": self._best_obj,
            "patience_counter": self._patience_counter,
            "rho_is_dict": self._rho_is_dict,
            "anderson": self.anderson,
            "aa_F": None if not self.anderson else [f.copy() for f in self._aa.F],
            "aa_G": None if not self.anderson else [g.copy() for g in self._aa.G],
        }

        import os
        import pickle

        os.makedirs(self.checkpoint_path, exist_ok=True)
        checkpoint_file = f"{self.checkpoint_path}/checkpoint_{iteration}.pkl"
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint, f)
        if self.verbose:
            print(f"Saved checkpoint at iteration {iteration} -> {checkpoint_file}")

    def load_checkpoint(self, checkpoint_file: str) -> int:
        """Load checkpoint from disk and return start iteration."""
        import pickle

        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)

        self.rho = checkpoint["rho"]
        self._rho_is_dict = bool(
            checkpoint.get("rho_is_dict", isinstance(self.rho, dict))
        )
        self.blocks_ = checkpoint["blocks"]
        self.z_ = checkpoint["z"]
        self.x_local_ = checkpoint["x_local"]
        self.u_ = checkpoint["u"]
        self.weights_ = checkpoint["weights"]
        self.history_ = checkpoint["history"]
        self._best_obj = checkpoint.get("best_obj", np.inf)
        self._patience_counter = checkpoint.get("patience_counter", 0)

        if checkpoint.get("anderson", False):
            if self._aa is None:
                self._aa = _Anderson(self.anderson_m, self.anderson_reg)
                self.anderson = True
            self._aa.reset()
            F = checkpoint.get("aa_F", None)
            G = checkpoint.get("aa_G", None)
            if F is not None and G is not None:
                self._aa.F = [f.copy() for f in F]
                self._aa.G = [g.copy() for g in G]

        return int(checkpoint["iteration"])

    # -------------------------------- API --------------------------------- #

    def snapshot(self) -> Dict[str, object]:
        """Return a shallow snapshot (for experimentation / restarts)."""
        return {
            "rho": self.rho,
            "blocks": list(self.blocks_),
            "z": {k: v.copy() for k, v in self.z_.items()},
            "x": [{k: v.copy() for k, v in d.items()} for d in self.x_local_],
            "u": [{k: v.copy() for k, v in d.items()} for d in self.u_],
            "weights": None if self.weights_ is None else self.weights_.copy(),
            "best_obj": self._best_obj,
            "patience_counter": self._patience_counter,
            "rho_is_dict": self._rho_is_dict,
            "anderson": self.anderson,
            "aa_F": None if not self.anderson else [f.copy() for f in self._aa.F],
            "aa_G": None if not self.anderson else [g.copy() for g in self._aa.G],
        }

    def restore(self, state: Dict[str, object]) -> None:
        """Restore from a snapshot produced by `snapshot()`."""
        self.rho = state["rho"]
        self._rho_is_dict = bool(state.get("rho_is_dict", isinstance(self.rho, dict)))
        self.blocks_ = list(state["blocks"])
        self.z_ = {k: v.copy() for k, v in state["z"].items()}
        self.x_local_ = [{k: v.copy() for k, v in d.items()} for d in state["x"]]
        self.u_ = [{k: v.copy() for k, v in d.items()} for d in state["u"]]
        self.weights_ = None if state["weights"] is None else state["weights"].copy()
        self._best_obj = state.get("best_obj", np.inf)
        self._patience_counter = state.get("patience_counter", 0)
        if state.get("anderson", False):
            if self._aa is None:
                self._aa = _Anderson(self.anderson_m, self.anderson_reg)
                self.anderson = True
            self._aa.reset()
            F = state.get("aa_F", None)
            G = state.get("aa_G", None)
            if F is not None and G is not None:
                self._aa.F = [f.copy() for f in F]
                self._aa.G = [g.copy() for g in G]

    # -------------------------------- fit --------------------------------- #

    def fit(
        self,
        X_list: Sequence[Array],
        y_list: Sequence[Optional[Array]],
        z0: Optional[Dict[str, Array]] = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Run consensus-ADMM with enhanced features.

        Args:
            X_list: list of site datasets
            y_list: list of site targets (or None if unsupervised)
            z0: optional warm start dictionary for blocks
            resume_from_checkpoint: path to checkpoint file to resume from

        Returns:
            self (with enhanced state and history)
        """
        start_iter = 0

        if resume_from_checkpoint is not None:
            start_iter = self.load_checkpoint(resume_from_checkpoint)
            if self.verbose:
                print(f"Resumed from checkpoint at iteration {start_iter}")
        else:
            K = len(X_list)
            assert K == len(
                y_list
            ), "X_list and y_list must have same length (K sites)."

            # weights: by sample count (rows) if enabled, else uniform
            n_i = np.array(
                [getattr(X, "shape", (0,))[0] or 1 for X in X_list], dtype=float
            )
            weights = (
                n_i / n_i.sum() if self.use_weights else np.ones(K, dtype=float) / K
            )
            self.weights_ = weights

            # initialize globals/blocks
            z = (
                self._init_blocks(X_list, y_list)
                if z0 is None
                else {k: v.copy() for k, v in z0.items()}
            )
            self.blocks_ = list(z.keys())
            self.z_ = {name: z[name].copy() for name in self.blocks_}

            # locals and duals
            self.x_local_ = []
            self.u_ = []
            for _ in range(K):
                self.x_local_.append(
                    {name: np.zeros_like(z[name]) for name in self.blocks_}
                )
                self.u_.append({name: np.zeros_like(z[name]) for name in self.blocks_})

            self.history_ = History(
                r_norm=[],
                s_norm=[],
                eps_pri=[],
                eps_dual=[],
                rho=[],
                obj=[],
                timing={"local_updates": [], "global_updates": [], "communication": []},
                rel_r=[],
                rel_s=[],
            )
            if self.anderson and self._aa is not None:
                self._aa.reset()

        hist = self.history_
        z = self.z_
        K = len(X_list)

        # basic shape cache for diagnostics
        dim_total = sum(z[name].size for name in self.blocks_)
        sqrtK = np.sqrt(K)
        sqrtKD = np.sqrt(K * dim_total)

        with self:
            # --------------- iterations --------------- #
            for it in range(start_iter, self.max_iters):
                iter_start_time = time.time()

                # Optional Nesterov-style schedule (updates self.alpha)
                if self.relax_schedule == "nesterov":
                    theta_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * self._theta_k**2))
                    self.alpha = (self._theta_k - 1.0) / theta_next
                    self._theta_k = theta_next

                # (1) Local updates (possibly parallel)
                local_start = time.time()
                updated_locals = self._parallel_local_updates(X_list, y_list, z)

                # Update local states, with checks
                for i, x_i in enumerate(updated_locals):
                    if x_i is not None:
                        for name in self.blocks_:
                            if x_i[name].shape != z[name].shape:
                                raise ValueError(
                                    f"Local update shape mismatch for block '{name}' at site {i}"
                                )
                        self.x_local_[i] = x_i

                local_time = time.time() - local_start
                hist.timing["local_updates"].append(local_time)

                # (2) Global step (aggregation -> prox -> (optional) Anderson)
                global_start = time.time()
                z_old = {name: z[name].copy() for name in self.blocks_}

                # Compute aggregated values per block (with over-relaxation)
                aggregated: Dict[str, Array] = {}
                for name in self.blocks_:
                    if self.alpha != 1.0:
                        xbar_plus_u = []
                        for i in range(K):
                            xbar = (
                                self.alpha * self.x_local_[i][name]
                                + (1.0 - self.alpha) * z_old[name]
                            )
                            xbar_plus_u.append(xbar + self.u_[i][name])
                        if self.aggregation_method == "weighted_avg":
                            avg = sum(
                                self.weights_[i] * xbar_plus_u[i] for i in range(K)
                            )
                        else:
                            temp_x = [
                                {name: xbar_plus_u[i] - self.u_[i][name]}
                                for i in range(K)
                            ]
                            temp_u = [{name: self.u_[i][name]} for i in range(K)]
                            avg = self._aggregate_block(
                                name, temp_x, temp_u, self.weights_
                            )
                    else:
                        avg = self._aggregate_block(
                            name, self.x_local_, self.u_, self.weights_
                        )

                    # Optional: communication compression round trip (receive decompressed)
                    if self.compression_ratio is not None:
                        packet, meta = self._compress_topk(avg, self.compression_ratio)
                        avg = self._decompress_topk(packet, meta)

                    # Prox step (per block); subclasses can scale internally via _prox_arg_scale if needed
                    aggregated[name] = self._prox(name, avg)

                # Optional Anderson acceleration on z
                if self.anderson and self._aa is not None:
                    z_proposed = {
                        name: aggregated[name].copy() for name in self.blocks_
                    }
                    residual = {
                        name: z_proposed[name] - z_old[name] for name in self.blocks_
                    }
                    z_acc = self._aa.update(z_proposed, residual)

                    # Safeguard: accept only if residual decreased
                    new_r = 0.0
                    old_r = 0.0
                    for i in range(K):
                        for n in self.blocks_:
                            new_r += _sqnorm(self.x_local_[i][n] - z_acc[n])
                            old_r += _sqnorm(self.x_local_[i][n] - z_proposed[n])
                    if new_r <= old_r:
                        for n in self.blocks_:
                            aggregated[n] = z_acc[n]

                # Commit z
                for name in self.blocks_:
                    z[name] = aggregated[name]

                global_time = time.time() - global_start
                hist.timing["global_updates"].append(global_time)

                # (3) Dual updates
                comm_start = time.time()
                for i in range(K):
                    for name in self.blocks_:
                        self.u_[i][name] = self.u_[i][name] + (
                            self.x_local_[i][name] - z[name]
                        )

                # Sync public z_
                self.z_ = {name: z[name].copy() for name in self.blocks_}

                comm_time = time.time() - comm_start
                hist.timing["communication"].append(comm_time)

                # (4) Diagnostics
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    r_sq = s_sq = x_norm_sq = z_norm_sq = sum_u_sq = 0.0
                    sum_u_block: Dict[str, Array] = {
                        name: np.zeros_like(z[name]) for name in self.blocks_
                    }

                    for i in range(K):
                        for name in self.blocks_:
                            r = self.x_local_[i][name] - z[name]
                            r_sq += _sqnorm(r)
                            x_norm_sq += _sqnorm(self.x_local_[i][name])
                            sum_u_block[name] += self.u_[i][name]

                    for name in self.blocks_:
                        dz = z[name] - z_old[name]
                        rho_b = self._get_rho_block(name)
                        s_sq += (rho_b**2) * K * _sqnorm(dz)
                        z_norm_sq += _sqnorm(z[name])
                        sum_u_sq += _sqnorm(sum_u_block[name])

                    r_norm = np.sqrt(r_sq)
                    s_norm = np.sqrt(s_sq)
                    eps_pri = sqrtKD * self.abstol + self.reltol * max(
                        np.sqrt(x_norm_sq), sqrtK * np.sqrt(z_norm_sq)
                    )
                    # For eps_dual we use average rho if dict
                    if self._rho_is_dict:
                        rho_mean = float(
                            np.mean([self._get_rho_block(n) for n in self.blocks_])
                        )
                    else:
                        rho_mean = float(self.rho)
                    eps_dual = sqrtKD * self.abstol + self.reltol * rho_mean * np.sqrt(
                        sum_u_sq
                    )

                    # Relative residuals (optional)
                    x_ref = max(1.0, np.sqrt(x_norm_sq))
                    z_ref = max(1.0, np.sqrt(z_norm_sq))
                    rel_r = r_norm / x_ref
                    rel_s = s_norm / (
                        rho_mean * z_ref if rho_mean > 0 else max(1.0, z_ref)
                    )

                    # Objective (subclass)
                    obj_val = self._objective(X_list, y_list, self.x_local_)

                # Update history
                hist.r_norm.append(r_norm)
                hist.s_norm.append(s_norm)
                hist.eps_pri.append(eps_pri)
                hist.eps_dual.append(eps_dual)
                # Store scalar rho (mean if dict) for logging continuity
                hist.rho.append(rho_mean)
                hist.obj.append(obj_val)
                hist.rel_r.append(rel_r)
                hist.rel_s.append(rel_s)

                if self.verbose:
                    iter_time = time.time() - iter_start_time
                    print(
                        f"[{it:4d}] r={r_norm:.3e} (≤{eps_pri:.3e}) | "
                        f"s={s_norm:.3e} (≤{eps_dual:.3e}) | ρ≈{rho_mean:.3g} | "
                        f"obj={obj_val:.6g} | t={iter_time:.3f}s "
                        f"(loc:{local_time:.3f}, glob:{global_time:.3f})"
                    )

                # (5) Block-wise adaptive rho with hysteresis & clamping
                if (
                    self.adaptive_rho
                    and it > 0
                    and np.isfinite(r_norm)
                    and np.isfinite(s_norm)
                ):
                    # per-block residual norms
                    r_block = {name: 0.0 for name in self.blocks_}
                    s_block = {name: 0.0 for name in self.blocks_}
                    for i in range(K):
                        for name in self.blocks_:
                            r_block[name] += _sqnorm(self.x_local_[i][name] - z[name])
                    for name in self.blocks_:
                        dz = z[name] - z_old[name]
                        rho_b = self._get_rho_block(name)
                        s_block[name] = (rho_b**2) * K * _sqnorm(dz)

                    for name in self.blocks_:
                        r_b, s_b = np.sqrt(r_block[name]), np.sqrt(s_block[name])
                        if r_b > self.rho_mu * s_b:
                            self._hyst_pri += 1
                            self._hyst_dual = 0
                            if self._hyst_pri >= self.rho_hysteresis:
                                old = self._get_rho_block(name)
                                new = np.clip(
                                    old * self.rho_tau, self.rho_min, self.rho_max
                                )
                                if new != old:
                                    self._set_rho_block(name, new)
                                    # dual scaling when rho increases: u <- u / tau
                                    for i in range(K):
                                        self.u_[i][name] /= self.rho_tau
                                    if self.verbose:
                                        print(f"[iter {it}]  ↑ rho[{name}] -> {new:g}")
                                self._hyst_pri = 0
                        elif s_b > self.rho_mu * r_b:
                            self._hyst_dual += 1
                            self._hyst_pri = 0
                            if self._hyst_dual >= self.rho_hysteresis:
                                old = self._get_rho_block(name)
                                new = np.clip(
                                    old / self.rho_tau, self.rho_min, self.rho_max
                                )
                                if new != old:
                                    self._set_rho_block(name, new)
                                    # dual scaling when rho decreases: u <- u * tau
                                    for i in range(K):
                                        self.u_[i][name] *= self.rho_tau
                                    if self.verbose:
                                        print(f"[iter {it}]  ↓ rho[{name}] -> {new:g}")
                                self._hyst_dual = 0
                        else:
                            self._hyst_pri = self._hyst_dual = 0

                # (6) NaN/Inf guard
                if _nan_guard(*(list(z.values()))):
                    hist.converged = False
                    hist.iters = it + 1
                    hist.reason = "NaN/Inf encountered"
                    if self.verbose:
                        print("Stopping: NaN/Inf encountered.")
                    break

                # (7) Convergence checks
                converged, reason = self._check_convergence(
                    hist, r_norm, s_norm, eps_pri, eps_dual, obj_val
                )
                if converged:
                    hist.converged = True
                    hist.iters = it + 1
                    hist.reason = reason
                    if self.verbose:
                        print(f"Converged at iter {it}: {reason}")
                    break

                # (8) Checkpoint hook
                self._save_checkpoint(it)

                # (9) Callback (early stopping by raising StopIteration)
                if self.callback is not None:
                    try:
                        self.callback(it, self, hist)
                    except StopIteration as e:
                        hist.converged = True
                        hist.iters = it + 1
                        hist.reason = f"stopped by callback: {e}"
                        if self.verbose:
                            print(f"Stopped by callback at iter {it}.")
                        break
            else:
                # Hit max_iters
                hist.converged = False
                hist.iters = self.max_iters
                hist.reason = "max_iters reached"

        # finalize state
        self.z_ = {name: z[name].copy() for name in self.blocks_}
        self.history_ = hist
        return self
