"""Constraint-aware alternating optimizer for UAV-ISAC warm starts.

The public class names are retained for checkpoint/data-pipeline compatibility,
but this module intentionally describes the implemented algorithm accurately:
it is a deterministic, constraint-aware alternating optimizer, not a formal
convex SCA/FP implementation.

The optimizer has four invariants:

* model-provided Q/A/P are used by the first deployment update;
* communication gains are recomputed when Q changes;
* association, power, mobility and separation constraints are checked together;
* the same physical model is used for optimization, ranking and evaluation.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

from ..env.uav_channel import ISACChannel


@dataclass
class SCAFPConfig:
    """Configuration for the constraint-aware alternating optimizer."""

    max_outer_iters: int = 30
    max_inner_iters: int = 5
    # Deprecated compatibility alias.  When supplied, it overrides
    # ``max_outer_iters``; None avoids the old accidental 100-iteration default.
    max_iters: Optional[int] = None
    tol: float = 1e-4
    lambda_sensing: float = 0.5
    lambda_idle_penalty: float = 0.0
    sinr_c_min: float = 1.0
    sinr_s_min: float = 10.0
    rate_min_bps: float = 1e6
    min_separation_m: float = 10.0
    constraint_penalty: float = 100.0
    power_update_rate: float = 0.5
    warm_start_hold_iters: int = 1
    # Retained only for config compatibility.  Hard separation projection is
    # authoritative; this optional term merely breaks symmetric local optima.
    lambda_repel: float = 0.0
    epsilon_min_repel: float = 1e-6
    ground_clutter_db: float = 0.0
    verbose: bool = False


@dataclass
class SCAFPSolution:
    Q: np.ndarray
    A: np.ndarray
    W_c_power: np.ndarray
    W_s_power: np.ndarray
    utility: float
    iterations: int
    converged: bool
    solve_time: float
    feasible: bool = False
    raw_utility: float = float("-inf")
    initial_utility: float = float("-inf")
    constraint_violations: Dict[str, float] = field(default_factory=dict)
    algorithm: str = "constraint_aware_alternating_optimization"


class SCAFPOptimizer:
    """Constraint-aware alternating optimizer with Q/A/P warm-start support."""

    def __init__(
        self,
        config: SCAFPConfig,
        M: int = 4,
        K: int = 20,
        T: int = 6,
        N_t: int = 8,
        N_r: Optional[int] = None,
        carrier_freq_ghz: float = 5.8,
        bandwidth_mhz: float = 20.0,
        noise_figure_db: float = 9.0,
        area_size: Tuple[float, float] = (1000.0, 1000.0),
        altitude_range: Tuple[float, float] = (50.0, 300.0),
        p_max: float = 1.0,
        noise_power: float = 1e-12,
        load_cap: int = 10,
        v_max: float = 15.0,
        slot_duration: float = 1.0,
    ):
        self.cfg = config
        self.M = int(M)
        self.K = int(K)
        self.T = int(T)
        self.N_t = int(N_t)
        self.N_r = int(N_r if N_r is not None else N_t)
        self.carrier_freq_ghz = float(carrier_freq_ghz)
        self.area_w, self.area_h = map(float, area_size)
        self.H_min, self.H_max = map(float, altitude_range)
        self.P_max = float(p_max)
        self.N0 = max(float(noise_power), 1e-30)
        self.K_max = int(load_cap)
        self.v_max = float(v_max)
        self.slot_duration = float(slot_duration)
        self.max_displacement = self.v_max * self.slot_duration
        self.min_separation_m = float(config.min_separation_m)
        p_max_dbm = 10.0 * np.log10(max(self.P_max, 1e-30)) + 30.0
        self.channel = ISACChannel(
            carrier_freq_ghz=self.carrier_freq_ghz,
            bandwidth_mhz=bandwidth_mhz,
            num_antennas_tx=self.N_t,
            num_antennas_rx=self.N_r,
            p_max_dbm=p_max_dbm,
            noise_figure_db=noise_figure_db,
        )
        self.wavelength = self.channel.wavelength
        if self.cfg.rate_min_bps < 0.0:
            raise ValueError("rate_min_bps must be non-negative")
        rate_sinr_min = (
            2.0 ** (float(self.cfg.rate_min_bps) / self.channel.B) - 1.0
        )
        self.comm_sinr_min = max(
            float(self.cfg.sinr_c_min), float(rate_sinr_min)
        )
        self.rng = np.random.RandomState()

        if self.M <= 0 or self.K <= 0 or self.T <= 0:
            raise ValueError("M, K and T must be positive")
        if self.K_max <= 0 or self.M * self.K_max < self.K:
            raise ValueError(
                "association capacity is infeasible: "
                f"M={self.M}, K={self.K}, K_max={self.K_max}"
            )
        if not 0.0 < self.cfg.power_update_rate <= 1.0:
            raise ValueError("power_update_rate must be in (0, 1]")

    # ------------------------------------------------------------------
    # Public solve/evaluation interface
    # ------------------------------------------------------------------

    def solve(
        self,
        environment: Dict,
        warm_start: Optional[Dict] = None,
        seed: Optional[int] = None,
    ) -> SCAFPSolution:
        """Optimize one environment from either a cold or model warm start."""
        env = self._validate_environment(environment)
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        started = time.time()
        is_warm = warm_start is not None
        if is_warm:
            Q, A, P_comm, P_sense = self._warmstart_to_init(warm_start, env)
        else:
            Q, A, P_comm, P_sense = self._random_init(env)

        initial = self.evaluate_solution(Q, A, P_comm, P_sense, env)
        initial_utility = initial["utility"]
        previous_utility = initial_utility
        best = initial
        best_state = (
            Q.copy(),
            A.copy(),
            P_comm.copy(),
            P_sense.copy(),
        )
        converged = False
        max_iters = (
            int(self.cfg.max_iters)
            if self.cfg.max_iters is not None
            else int(self.cfg.max_outer_iters)
        )
        if max_iters <= 0:
            raise ValueError("max_outer_iters/max_iters must be positive")

        for outer_iter in range(max_iters):
            # The first deployment update consumes the supplied A and P.  This
            # ordering fixes the old behavior that overwrote both before they
            # could influence any downstream optimization.
            Q_next = self._optimize_deployment_sca(
                Q, A, P_comm, P_sense, env
            )
            gains_next = self._communication_gains(Q_next, env)

            hold_warm_start = (
                is_warm and outer_iter < max(0, self.cfg.warm_start_hold_iters)
            )
            if hold_warm_start:
                A_next = A.copy()
                P_comm_next, P_sense_next = self._sanitize_power(
                    P_comm, P_sense, A_next
                )
            else:
                A_next = self._optimize_association(
                    gains_next, env["user_weights"]
                )
                P_opt_comm, P_opt_sense = self._optimize_beamforming(
                    Q_next,
                    A_next,
                    gains_next,
                    env["target_positions"],
                    env["user_weights"],
                    env["target_detected"],
                )
                alpha = self.cfg.power_update_rate
                P_comm_next = (1.0 - alpha) * P_comm + alpha * P_opt_comm
                P_sense_next = (1.0 - alpha) * P_sense + alpha * P_opt_sense
                P_comm_next, P_sense_next = self._sanitize_power(
                    P_comm_next, P_sense_next, A_next
                )

            current = self.evaluate_solution(
                Q_next, A_next, P_comm_next, P_sense_next, env
            )
            utility = current["utility"]
            if self.cfg.verbose:
                print(
                    f"  AO iter {outer_iter + 1}: utility={utility:.6f}, "
                    f"feasible={current['feasible']}"
                )

            Q, A = Q_next, A_next
            P_comm, P_sense = P_comm_next, P_sense_next
            if not np.isfinite(utility):
                break
            if self._candidate_is_better(current, best):
                best = current
                best_state = (
                    Q.copy(),
                    A.copy(),
                    P_comm.copy(),
                    P_sense.copy(),
                )

            relative_change = abs(utility - previous_utility) / (
                1.0 + abs(previous_utility)
            )
            if (
                outer_iter >= max(0, self.cfg.warm_start_hold_iters)
                and relative_change < self.cfg.tol
            ):
                converged = True
                break
            previous_utility = utility

        # Alternating updates are not guaranteed to be monotone.  Returning
        # the last iterate silently discarded an earlier feasible optimum and
        # made otherwise deterministic Oracle labels depend on when the loop
        # happened to stop.  Always return the best visited state, preferring
        # feasibility before the penalized objective.
        Q, A, P_comm, P_sense = best_state
        final = best
        return SCAFPSolution(
            Q=Q.astype(np.float32),
            A=A.astype(np.float32),
            W_c_power=P_comm.astype(np.float32),
            W_s_power=P_sense.astype(np.float32),
            utility=float(final["utility"]),
            raw_utility=float(final["raw_utility"]),
            initial_utility=float(initial_utility),
            iterations=outer_iter + 1,
            converged=converged and np.isfinite(final["utility"]),
            feasible=bool(final["feasible"]),
            constraint_violations=final["constraint_violations"],
            solve_time=time.time() - started,
        )

    @staticmethod
    def _candidate_is_better(candidate: Dict, incumbent: Dict) -> bool:
        candidate_feasible = bool(candidate["feasible"])
        incumbent_feasible = bool(incumbent["feasible"])
        if candidate_feasible != incumbent_feasible:
            return candidate_feasible
        candidate_utility = float(candidate["utility"])
        incumbent_utility = float(incumbent["utility"])
        candidate_finite = np.isfinite(candidate_utility)
        incumbent_finite = np.isfinite(incumbent_utility)
        if candidate_finite != incumbent_finite:
            return bool(candidate_finite)
        if not candidate_finite:
            return False
        return candidate_utility > incumbent_utility

    def evaluate_solution(
        self,
        Q: np.ndarray,
        A: np.ndarray,
        P_comm: np.ndarray,
        P_sense: np.ndarray,
        environment: Dict,
    ) -> Dict[str, object]:
        """Evaluate utility and all constraints with the solver's own model."""
        env = self._validate_environment(environment)
        Q = np.asarray(Q, dtype=np.float64)
        A = np.asarray(A, dtype=np.float64)
        P_comm = np.asarray(P_comm, dtype=np.float64)
        P_sense = np.asarray(P_sense, dtype=np.float64)
        gains_comm = self._communication_gains(Q, env)
        gains_sense = self._sensing_gains(Q, env["target_positions"])
        raw_utility = self._raw_utility(
            A,
            P_comm,
            P_sense,
            gains_comm,
            gains_sense,
            env["user_weights"],
            env["target_detected"],
        )
        violations = self._constraint_violations(
            Q,
            A,
            P_comm,
            P_sense,
            gains_comm,
            gains_sense,
            env["q_current"],
            env["target_detected"],
        )
        penalty = self.cfg.constraint_penalty * sum(violations.values())
        feasible = all(value <= 1e-5 for value in violations.values())
        return {
            "utility": float(raw_utility - penalty),
            "raw_utility": float(raw_utility),
            "feasible": feasible,
            "constraint_violations": {
                key: float(value) for key, value in violations.items()
            },
            "communication_gains": gains_comm,
            "sensing_gains": gains_sense,
        }

    # ------------------------------------------------------------------
    # Environment and physical models
    # ------------------------------------------------------------------

    def _validate_environment(self, environment: Dict) -> Dict:
        env = dict(environment)
        defaults = {
            "q_current": np.zeros((self.M, 3), dtype=np.float64),
            "user_positions": np.zeros((self.K, 2), dtype=np.float64),
            "target_positions": np.zeros((self.T, 2), dtype=np.float64),
            "channel_gains": np.ones((self.M, self.K), dtype=np.float64),
            "user_weights": np.ones(self.K, dtype=np.float64),
            "target_detected": np.ones(self.T, dtype=bool),
        }
        shapes = {
            "q_current": (self.M, 3),
            "user_positions": (self.K, 2),
            "target_positions": (self.T, 2),
            "channel_gains": (self.M, self.K),
            "user_weights": (self.K,),
            "target_detected": (self.T,),
        }
        for key, default in defaults.items():
            dtype = bool if key == "target_detected" else np.float64
            value = np.asarray(env.get(key, default), dtype=dtype)
            if value.shape != shapes[key]:
                raise ValueError(
                    f"{key} must have shape {shapes[key]}, got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"{key} contains NaN or infinity")
            env[key] = value
        env["channel_gains"] = np.maximum(env["channel_gains"], 1e-30)
        env["user_weights"] = np.maximum(env["user_weights"], 0.0)
        return env

    def _expected_comm_gains(
        self, Q: np.ndarray, user_positions: np.ndarray
    ) -> np.ndarray:
        gains = np.empty((self.M, self.K), dtype=np.float64)
        for m in range(self.M):
            for k in range(self.K):
                gains[m, k] = self.channel.expected_channel_gain(
                    Q[m], user_positions[k]
                )
        return np.maximum(gains, 1e-30)

    def _communication_gains(self, Q: np.ndarray, env: Dict) -> np.ndarray:
        """Move sampled CSI with Q using a large-scale geometry ratio."""
        base_expected = self._expected_comm_gains(
            env["q_current"], env["user_positions"]
        )
        moved_expected = self._expected_comm_gains(Q, env["user_positions"])
        geometry_ratio = np.clip(moved_expected / base_expected, 1e-3, 1e3)
        return np.maximum(env["channel_gains"] * geometry_ratio, 1e-30)

    def _sensing_gains(
        self, Q: np.ndarray, target_positions: np.ndarray
    ) -> np.ndarray:
        gains = np.empty((self.M, self.T), dtype=np.float64)
        for m in range(self.M):
            for target_idx in range(self.T):
                gains[m, target_idx] = self.channel.sensing_path_gain(
                    Q[m], target_positions[target_idx]
                )
        return np.maximum(gains, 1e-30)

    # ------------------------------------------------------------------
    # Initialization and projections
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_optional_batch(value: np.ndarray, expected_ndim: int) -> np.ndarray:
        value = np.asarray(value)
        if value.ndim == expected_ndim + 1 and value.shape[0] == 1:
            value = value[0]
        return value

    def _random_init(self, env: Dict) -> Tuple[np.ndarray, ...]:
        q_current = env["q_current"]
        directions = self.rng.normal(size=(self.M, 3))
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        radii = self.rng.uniform(0.0, self.max_displacement, size=(self.M, 1))
        Q = self._project_deployment_feasible(
            q_current + directions * radii, q_current
        )

        distances = np.linalg.norm(
            Q[:, None, :2] - env["user_positions"][None, :, :], axis=-1
        )
        A = self._capacity_constrained_assignment(-distances)
        gains = self._communication_gains(Q, env)
        P_comm, P_sense = self._optimize_beamforming(
            Q,
            A,
            gains,
            env["target_positions"],
            env["user_weights"],
            env["target_detected"],
        )
        return Q, A, P_comm, P_sense

    def _warmstart_to_init(
        self, warm_start: Dict, env: Dict
    ) -> Tuple[np.ndarray, ...]:
        delta_q = self._remove_optional_batch(
            warm_start.get("delta_q", np.zeros((self.M, 3))), 2
        ).astype(np.float64)
        delta_a = self._remove_optional_batch(
            warm_start.get("delta_a", np.zeros((self.M, self.K))), 2
        ).astype(np.float64)
        delta_p = self._remove_optional_batch(
            warm_start.get("delta_p", np.zeros((self.M, self.K + 1))), 2
        ).astype(np.float64)
        expected = {
            "delta_q": ((self.M, 3), delta_q),
            "delta_a": ((self.M, self.K), delta_a),
            "delta_p": ((self.M, self.K + 1), delta_p),
        }
        for name, (shape, value) in expected.items():
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity")

        norms = np.linalg.norm(delta_q, axis=1, keepdims=True)
        delta_q *= np.minimum(1.0, self.max_displacement / np.maximum(norms, 1e-12))
        Q = self._project_deployment_feasible(
            env["q_current"] + delta_q, env["q_current"]
        )
        A = self._capacity_constrained_assignment(delta_a)
        P_comm, P_sense = self._sanitize_power(
            delta_p[:, : self.K], delta_p[:, self.K], A
        )
        return Q, A, P_comm, P_sense

    def _capacity_constrained_assignment(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (self.M, self.K):
            raise ValueError(
                f"association scores must have shape {(self.M, self.K)}, got {scores.shape}"
            )
        slot_to_uav = np.repeat(np.arange(self.M), self.K_max)
        assigned_slots, assigned_users = linear_sum_assignment(-scores[slot_to_uav])
        A = np.zeros((self.M, self.K), dtype=np.float64)
        A[slot_to_uav[assigned_slots], assigned_users] = 1.0
        return A

    def _sanitize_power(
        self, P_comm: np.ndarray, P_sense: np.ndarray, A: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        P_comm = np.asarray(P_comm, dtype=np.float64).copy()
        P_sense = np.asarray(P_sense, dtype=np.float64).copy()
        if P_comm.shape != (self.M, self.K):
            raise ValueError(
                f"P_comm must have shape {(self.M, self.K)}, got {P_comm.shape}"
            )
        if P_sense.shape != (self.M,):
            raise ValueError(f"P_sense must have shape {(self.M,)}, got {P_sense.shape}")
        P_comm = np.maximum(P_comm, 0.0)
        P_sense = np.maximum(P_sense, 0.0)
        P_comm *= A > 0.5
        for m in range(self.M):
            total = float(P_comm[m].sum() + P_sense[m])
            if total > self.P_max:
                scale = self.P_max / max(total, 1e-30)
                P_comm[m] *= scale
                P_sense[m] *= scale
        return P_comm, P_sense

    def _project_deployment_feasible(
        self, Q: np.ndarray, q_current: np.ndarray
    ) -> np.ndarray:
        Q = np.asarray(Q, dtype=np.float64).copy()
        q_current = np.asarray(q_current, dtype=np.float64)

        def project_individual() -> None:
            Q[:, 0] = np.clip(Q[:, 0], 0.0, self.area_w)
            Q[:, 1] = np.clip(Q[:, 1], 0.0, self.area_h)
            Q[:, 2] = np.clip(Q[:, 2], self.H_min, self.H_max)
            displacement = Q - q_current
            norms = np.linalg.norm(displacement, axis=1, keepdims=True)
            displacement *= np.minimum(
                1.0, self.max_displacement / np.maximum(norms, 1e-12)
            )
            Q[:] = q_current + displacement
            Q[:, 0] = np.clip(Q[:, 0], 0.0, self.area_w)
            Q[:, 1] = np.clip(Q[:, 1], 0.0, self.area_h)
            Q[:, 2] = np.clip(Q[:, 2], self.H_min, self.H_max)

        project_individual()
        for _ in range(100):
            changed = False
            for first in range(self.M):
                for second in range(first + 1, self.M):
                    diff = Q[first] - Q[second]
                    distance = float(np.linalg.norm(diff))
                    if distance + 1e-7 >= self.min_separation_m:
                        continue
                    if distance < 1e-12:
                        angle = 2.0 * np.pi * (first + 1) / (self.M + 1)
                        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
                    else:
                        direction = diff / distance
                    correction = 0.5 * (self.min_separation_m - distance + 1e-6)
                    Q[first] += correction * direction
                    Q[second] -= correction * direction
                    changed = True
            project_individual()
            if not changed:
                break
        return Q

    # ------------------------------------------------------------------
    # Alternating updates
    # ------------------------------------------------------------------

    def _optimize_association(
        self, channel_gains: np.ndarray, user_weights: np.ndarray
    ) -> np.ndarray:
        gains = np.asarray(channel_gains, dtype=np.float64)
        weights = np.asarray(user_weights, dtype=np.float64)
        if gains.shape != (self.M, self.K):
            raise ValueError(
                f"channel_gains must have shape {(self.M, self.K)}, got {gains.shape}"
            )
        if weights.shape != (self.K,):
            raise ValueError(
                f"user_weights must have shape {(self.K,)}, got {weights.shape}"
            )
        candidate_power = self.P_max / float(self.K_max + 1)
        sinr = np.maximum(gains, 0.0) * candidate_power / self.N0
        required_power = (
            self.comm_sinr_min * self.N0 / np.maximum(gains, 1e-30)
        )
        # Reward rate while discouraging links whose QoS requirement alone
        # would consume a large fraction of the UAV budget.
        scores = weights[None, :] * np.log2(1.0 + sinr)
        scores -= self.cfg.constraint_penalty * np.maximum(
            required_power / self.P_max - 1.0, 0.0
        )
        return self._capacity_constrained_assignment(scores)

    def _optimize_beamforming(
        self,
        Q: np.ndarray,
        A: np.ndarray,
        channel_gains: np.ndarray,
        target_positions: np.ndarray,
        user_weights: Optional[np.ndarray] = None,
        target_detected: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Allocate power from QoS requirements plus marginal utility.

        Unlike the previous fixed 70/30 split, this update is association-aware,
        sets inactive communication entries to zero, and explicitly attempts to
        meet both configured SINR thresholds before distributing spare power.
        """
        weights = (
            np.ones(self.K, dtype=np.float64)
            if user_weights is None
            else np.asarray(user_weights, dtype=np.float64)
        )
        sensing_gains = self._sensing_gains(Q, target_positions)
        detected = (
            np.ones(self.T, dtype=bool)
            if target_detected is None
            else np.asarray(target_detected, dtype=bool)
        )
        if detected.shape != (self.T,):
            raise ValueError(
                f"target_detected must have shape {(self.T,)}, "
                f"got {detected.shape}"
            )
        target_owner = np.full(self.T, -1, dtype=np.int64)
        if np.any(detected):
            target_owner[detected] = np.argmax(
                sensing_gains[:, detected], axis=0
            )
        P_comm = np.zeros((self.M, self.K), dtype=np.float64)
        P_sense = np.zeros(self.M, dtype=np.float64)

        for m in range(self.M):
            active = np.flatnonzero(A[m] > 0.5)
            owned_targets = np.flatnonzero(target_owner == m)
            comm_required = np.array(
                [
                    self.comm_sinr_min * self.N0
                    / max(channel_gains[m, k], 1e-30)
                    for k in active
                ],
                dtype=np.float64,
            )
            if owned_targets.size:
                sense_required = max(
                    self.cfg.sinr_s_min * self.N0
                    / max(
                        sensing_gains[m, target_idx] * self.N_t * self.N_r,
                        1e-30,
                    )
                    for target_idx in owned_targets
                )
            else:
                sense_required = 0.0

            required_total = float(comm_required.sum() + sense_required)
            if required_total > self.P_max:
                scale = self.P_max / required_total
                comm_required *= scale
                sense_required *= scale
                spare = 0.0
            else:
                spare = self.P_max - required_total

            if active.size:
                comm_scores = weights[active] * np.sqrt(
                    np.maximum(channel_gains[m, active], 1e-30)
                )
            else:
                comm_scores = np.empty(0, dtype=np.float64)
            sensing_score = self.cfg.lambda_sensing * float(
                np.sqrt(sensing_gains[m, detected]).sum()
            )
            score_total = float(comm_scores.sum() + sensing_score)
            if score_total <= 0.0:
                sense_extra = spare if np.any(detected) else 0.0
                comm_extra = np.zeros_like(comm_scores)
            else:
                comm_extra = spare * comm_scores / score_total
                sense_extra = spare * sensing_score / score_total

            if active.size:
                P_comm[m, active] = comm_required + comm_extra
            P_sense[m] = sense_required + sense_extra

        return self._sanitize_power(P_comm, P_sense, A)

    def _optimize_deployment_sca(
        self,
        Q_init: np.ndarray,
        A: np.ndarray,
        P_comm: np.ndarray,
        P_sense: np.ndarray,
        environment: Dict,
    ) -> np.ndarray:
        """Locally update Q using the same utility/constraint model as ranking."""
        Q = np.asarray(Q_init, dtype=np.float64).copy()
        q_current = environment["q_current"]
        inner_iters = max(1, int(self.cfg.max_inner_iters))

        for _ in range(inner_iters):
            before = Q.copy()
            for m in range(self.M):
                q0 = q_current[m]
                bounds = [
                    (max(0.0, q0[0] - self.max_displacement),
                     min(self.area_w, q0[0] + self.max_displacement)),
                    (max(0.0, q0[1] - self.max_displacement),
                     min(self.area_h, q0[1] + self.max_displacement)),
                    (max(self.H_min, q0[2] - self.max_displacement),
                     min(self.H_max, q0[2] + self.max_displacement)),
                ]

                def objective(candidate: np.ndarray) -> float:
                    Q_candidate = Q.copy()
                    Q_candidate[m] = candidate
                    displacement_excess = max(
                        float(np.linalg.norm(candidate - q0)) - self.max_displacement,
                        0.0,
                    ) / max(self.max_displacement, 1e-12)
                    evaluated = self.evaluate_solution(
                        Q_candidate, A, P_comm, P_sense, environment
                    )
                    return float(
                        -evaluated["utility"]
                        + self.cfg.constraint_penalty * displacement_excess ** 2
                    )

                result = minimize(
                    objective,
                    Q[m],
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 20, "ftol": 1e-8},
                )
                if result.success and np.isfinite(result.x).all():
                    Q[m] = result.x
            Q = self._project_deployment_feasible(Q, q_current)
            if np.linalg.norm(Q - before) < 1e-5:
                break
        return Q

    # ------------------------------------------------------------------
    # Objective and feasibility
    # ------------------------------------------------------------------

    def _raw_utility(
        self,
        A: np.ndarray,
        P_comm: np.ndarray,
        P_sense: np.ndarray,
        gains_comm: np.ndarray,
        gains_sense: np.ndarray,
        user_weights: np.ndarray,
        target_detected: np.ndarray,
    ) -> float:
        comm_sinr = gains_comm * P_comm / self.N0
        comm_utility = float(
            np.sum(A * user_weights[None, :] * np.log2(1.0 + comm_sinr))
        )
        sensing_sinr = (
            P_sense[:, None]
            * gains_sense
            * self.N_t
            * self.N_r
            / self.N0
        )
        # Each target is considered served by its best UAV.  A logarithmic
        # term keeps communication and sensing on comparable numerical scales.
        detected = np.asarray(target_detected, dtype=bool)
        sensing_utility = (
            float(
                np.log2(
                    1.0 + np.max(sensing_sinr[:, detected], axis=0)
                ).sum()
            )
            if np.any(detected)
            else 0.0
        )
        idle_count = float(np.sum(A.sum(axis=1) < 0.5))
        return (
            comm_utility
            + self.cfg.lambda_sensing * sensing_utility
            - self.cfg.lambda_idle_penalty * idle_count
        )

    def _constraint_violations(
        self,
        Q: np.ndarray,
        A: np.ndarray,
        P_comm: np.ndarray,
        P_sense: np.ndarray,
        gains_comm: np.ndarray,
        gains_sense: np.ndarray,
        q_current: np.ndarray,
        target_detected: np.ndarray,
    ) -> Dict[str, float]:
        column_error = float(np.max(np.abs(A.sum(axis=0) - 1.0)))
        load_excess = float(
            np.max(np.maximum(A.sum(axis=1) - self.K_max, 0.0))
            / max(self.K_max, 1)
        )
        integrality = float(np.max(np.minimum(np.abs(A), np.abs(A - 1.0))))
        inactive_power = float(np.max(np.abs(P_comm[A < 0.5]))) if np.any(A < 0.5) else 0.0
        power_excess = float(
            np.max(
                np.maximum(P_comm.sum(axis=1) + P_sense - self.P_max, 0.0)
            ) / max(self.P_max, 1e-30)
        )
        negative_power = float(
            max(-float(np.min(P_comm)), -float(np.min(P_sense)), 0.0)
            / max(self.P_max, 1e-30)
        )

        movement = np.linalg.norm(Q - q_current, axis=1)
        movement_excess = float(
            np.max(np.maximum(movement - self.max_displacement, 0.0))
            / max(self.max_displacement, 1e-30)
        )
        boundary_excess = max(
            float(np.max(np.maximum(-Q[:, 0], 0.0))),
            float(np.max(np.maximum(Q[:, 0] - self.area_w, 0.0))),
            float(np.max(np.maximum(-Q[:, 1], 0.0))),
            float(np.max(np.maximum(Q[:, 1] - self.area_h, 0.0))),
            float(np.max(np.maximum(self.H_min - Q[:, 2], 0.0))),
            float(np.max(np.maximum(Q[:, 2] - self.H_max, 0.0))),
        ) / max(self.area_w, self.area_h, self.H_max, 1.0)

        min_distance = float("inf")
        for first in range(self.M):
            for second in range(first + 1, self.M):
                min_distance = min(
                    min_distance, float(np.linalg.norm(Q[first] - Q[second]))
                )
        separation_shortfall = (
            max(self.min_separation_m - min_distance, 0.0)
            / max(self.min_separation_m, 1e-30)
            if self.M > 1
            else 0.0
        )

        assigned_sinr = gains_comm * P_comm / self.N0
        active_sinr = assigned_sinr[A > 0.5]
        if active_sinr.size:
            comm_shortfall = float(
                np.max(
                    np.maximum(self.comm_sinr_min - active_sinr, 0.0)
                ) / max(self.comm_sinr_min, 1e-30)
            )
        else:
            comm_shortfall = 1.0
        sensing_sinr = (
            P_sense[:, None]
            * gains_sense
            * self.N_t
            * self.N_r
            / self.N0
        )
        detected = np.asarray(target_detected, dtype=bool)
        if np.any(detected):
            best_sensing_sinr = np.max(
                sensing_sinr[:, detected], axis=0
            )
            sensing_shortfall = float(
                np.max(
                    np.maximum(
                        self.cfg.sinr_s_min - best_sensing_sinr, 0.0
                    )
                )
                / max(self.cfg.sinr_s_min, 1e-30)
            )
        else:
            sensing_shortfall = 0.0
        return {
            "association_column_error": column_error,
            "association_load_excess": load_excess,
            "association_integrality": integrality,
            "inactive_power_leakage": inactive_power / max(self.P_max, 1e-30),
            "power_budget_excess": power_excess,
            "negative_power": negative_power,
            "movement_excess": movement_excess,
            "boundary_excess": boundary_excess,
            "separation_shortfall": float(separation_shortfall),
            "communication_sinr_shortfall": comm_shortfall,
            "sensing_sinr_shortfall": sensing_shortfall,
        }

    # Backward-compatible private utility entry used by a few external scripts.
    def _compute_utility(
        self,
        Q: np.ndarray,
        A: np.ndarray,
        P_comm: np.ndarray,
        P_sense: np.ndarray,
        channel_gains: np.ndarray,
        target_positions: np.ndarray,
        user_weights: np.ndarray,
        target_detected: Optional[np.ndarray] = None,
    ) -> float:
        gains_sense = self._sensing_gains(Q, target_positions)
        return self._raw_utility(
            A,
            P_comm,
            P_sense,
            channel_gains,
            gains_sense,
            user_weights,
            (
                np.ones(self.T, dtype=bool)
                if target_detected is None
                else target_detected
            ),
        )
