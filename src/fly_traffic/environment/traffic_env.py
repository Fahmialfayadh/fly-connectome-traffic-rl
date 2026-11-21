"""Gymnasium-compatible 4-way, 2-lane traffic intersection environment."""

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from fly_traffic.config import TrafficEnvConfig
from fly_traffic.environment.intersection_state import (
    LaneId,
    LaneQueue,
    PHASE_GREEN_LANES,
    PHASE_LABELS,
    SignalController,
    SignalPhase,
    SignalStage,
)
from fly_traffic.environment.traffic_profiles import TrafficProfile, TrafficProfileFactory


LANE_TO_PHASE: dict[int, SignalPhase] = {
    int(lane_id): phase
    for phase, green_lanes in PHASE_GREEN_LANES.items()
    for lane_id in green_lanes
}

# Symmetric physical path-intersection adjacency inside the 4-way box.
# Excludes same-phase partners and non-crossing cross-phase turn/straight paths.
CONFLICTING_LANES: dict[int, frozenset[int]] = {
    int(LaneId.N_STRAIGHT): frozenset(
        {
            int(LaneId.S_TURN),
            int(LaneId.E_STRAIGHT),
            int(LaneId.W_STRAIGHT),
            int(LaneId.W_TURN),
        }
    ),
    int(LaneId.N_TURN): frozenset(
        {
            int(LaneId.S_STRAIGHT),
            int(LaneId.E_STRAIGHT),
            int(LaneId.E_TURN),
            int(LaneId.W_TURN),
        }
    ),
    int(LaneId.S_STRAIGHT): frozenset(
        {
            int(LaneId.N_TURN),
            int(LaneId.E_STRAIGHT),
            int(LaneId.E_TURN),
            int(LaneId.W_STRAIGHT),
        }
    ),
    int(LaneId.S_TURN): frozenset(
        {
            int(LaneId.N_STRAIGHT),
            int(LaneId.E_TURN),
            int(LaneId.W_STRAIGHT),
            int(LaneId.W_TURN),
        }
    ),
    int(LaneId.E_STRAIGHT): frozenset(
        {
            int(LaneId.N_STRAIGHT),
            int(LaneId.N_TURN),
            int(LaneId.S_STRAIGHT),
            int(LaneId.W_TURN),
        }
    ),
    int(LaneId.E_TURN): frozenset(
        {
            int(LaneId.N_TURN),
            int(LaneId.S_STRAIGHT),
            int(LaneId.S_TURN),
            int(LaneId.W_STRAIGHT),
        }
    ),
    int(LaneId.W_STRAIGHT): frozenset(
        {
            int(LaneId.N_STRAIGHT),
            int(LaneId.S_STRAIGHT),
            int(LaneId.S_TURN),
            int(LaneId.E_TURN),
        }
    ),
    int(LaneId.W_TURN): frozenset(
        {
            int(LaneId.N_STRAIGHT),
            int(LaneId.N_TURN),
            int(LaneId.S_TURN),
            int(LaneId.E_STRAIGHT),
        }
    ),
}


class TrafficIntersectionEnv(gym.Env):
    """Single-intersection 4-phase traffic signal control environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    NUM_LANES = 8
    NUM_PHASES = 4
    OBS_DIM = 22

    def __init__(
        self,
        config: TrafficEnvConfig | None = None,
        profile_name: str | None = None,
        continuous_mode: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or TrafficEnvConfig()
        self.profile_name = profile_name or self.config.profile_name
        self._continuous_mode: bool = bool(continuous_mode)

        self.observation_space = spaces.Box(
            low=0.0,
            high=2.0,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.NUM_PHASES)

        self._all_red_steps: int = max(0, int(self.config.all_red_steps))
        self._lane_capacity_limit: int = int(self.config.max_lane_capacity)
        self._lanes: list[LaneQueue] = [
            LaneQueue(max_capacity=self._lane_capacity_limit)
            for _ in range(self.NUM_LANES)
        ]
        self._signal = SignalController(
            min_green_steps=self.config.min_green_steps,
            yellow_steps=self.config.yellow_steps,
            all_red_steps=self._all_red_steps,
        )
        self._active_profile: TrafficProfile = TrafficProfileFactory.create("normal")
        self._current_step: int = 0
        self._total_throughput: int = 0
        self._total_switches: int = 0
        self._total_collisions: int = 0
        self._cumulative_wait_seconds: float = 0.0
        self._episode_reward: float = 0.0
        self._last_step_arrivals: list[int] = [0] * self.NUM_LANES
        self._last_step_departures: list[int] = [0] * self.NUM_LANES
        self._last_manual_spawns: list[int] = [0] * self.NUM_LANES
        self._mutation_seq: int = 0
        self._box_stragglers: list[list[int]] = []
        self._last_step_collisions: list[dict[str, Any]] = []
        self._last_executed_phase: SignalPhase = SignalPhase.NS_STRAIGHT
        self._last_executed_stage: SignalStage = SignalStage.GREEN

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset lane queues, signal controller, and episode traffic profile."""
        super().reset(seed=seed)

        for lane in self._lanes:
            lane.max_capacity = self._lane_capacity_limit
            lane.reset()

        initial_phase = SignalPhase.NS_STRAIGHT
        if options and "initial_phase" in options:
            initial_phase = SignalPhase(int(options["initial_phase"]))
        self._signal.all_red_steps = self._all_red_steps
        self._signal.reset(initial_phase=initial_phase)

        chosen_profile = self.profile_name
        if options and "profile_name" in options:
            chosen_profile = str(options["profile_name"])
            
        self._active_profile = TrafficProfileFactory.create(
            chosen_profile,
            rng=self.np_random,
        )

        self._current_step = 0
        self._total_throughput = 0
        self._total_switches = 0
        self._total_collisions = 0
        self._cumulative_wait_seconds = 0.0
        self._episode_reward = 0.0
        self._last_step_arrivals = [0] * self.NUM_LANES
        self._last_step_departures = [0] * self.NUM_LANES
        self._last_manual_spawns = [0] * self.NUM_LANES
        self._mutation_seq += 1
        self._box_stragglers = []
        self._last_step_collisions = []
        self._last_executed_phase = initial_phase
        self._last_executed_stage = self._signal.stage

        obs = self._build_observation()
        info = self._build_info(
            step_throughput=0,
            step_wait_times=[],
            phase_switched=False,
        )
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance traffic simulation by one discrete decision interval."""
        target_phase = SignalPhase(int(action))
        previous_departures = list(self._last_step_departures)
        phase_switched = self._apply_signal_control(target_phase)
        executed_phase = self._signal.active_phase
        executed_stage = self._signal.stage

        step_throughput, step_wait_times, departures_per_lane = self._step_departures()
        step_collision_events, violation_waits = self._evaluate_collisions(
            phase_switched=phase_switched,
            previous_departures=previous_departures,
            departures_per_lane=departures_per_lane,
        )
        if violation_waits:
            step_wait_times.extend(violation_waits)

        arrivals_per_lane = self._step_arrivals()
        self._signal.step()

        self._last_executed_phase = executed_phase
        self._last_executed_stage = executed_stage
        self._last_step_departures = departures_per_lane
        self._last_step_arrivals = arrivals_per_lane
        self._last_manual_spawns = [0] * self.NUM_LANES
        self._mutation_seq += 1
        self._last_step_collisions = step_collision_events

        self._current_step += 1
        self._total_throughput += step_throughput
        if phase_switched:
            self._total_switches += 1
        self._total_collisions += len(step_collision_events)
        self._cumulative_wait_seconds += float(sum(step_wait_times))

        reward = self._compute_reward(
            phase_switched=phase_switched,
            collision_count=len(step_collision_events),
        )
        self._episode_reward += reward
        terminated = False
        truncated = (
            False
            if self._continuous_mode
            else (self._current_step >= self.config.max_steps)
        )

        obs = self._build_observation()
        info = self._build_info(
            step_throughput=step_throughput,
            step_wait_times=step_wait_times,
            phase_switched=phase_switched,
        )
        return obs, reward, terminated, truncated, info

    def set_profile(self, profile_name: str) -> None:
        """Update the traffic profile immediately without resetting queues."""
        self.profile_name = profile_name
        self._active_profile = TrafficProfileFactory.create(
            profile_name,
            rng=self.np_random,
        )

    def set_safety_buffer(self, all_red_steps: int) -> None:
        """Update intergreen ALL_RED clearance duration clamped to >= 0 steps."""
        clamped_steps = max(0, int(all_red_steps))
        self._all_red_steps = clamped_steps
        self._signal.all_red_steps = clamped_steps

    def set_continuous_mode(self, enabled: bool) -> None:
        """Enable or disable endless stepping without max_steps episode truncation."""
        self._continuous_mode = bool(enabled)

    def set_lane_capacity_limit(self, capacity: int) -> int:
        """Clamp and apply per-lane queue capacity across all 8 lanes."""
        if isinstance(capacity, bool):
            raise ValueError("capacity must be an integer.")
        hard_cap = max(10, int(self.config.hard_max_lane_capacity))
        applied_limit = min(max(int(capacity), 10), hard_cap)
        self._lane_capacity_limit = applied_limit
        for lane in self._lanes:
            lane.max_capacity = applied_limit
        self._mutation_seq += 1
        return applied_limit

    def inject_vehicles(self, lane_counts: dict[int, int]) -> list[int]:
        """Inject vehicles into specified lanes respecting each lane's max_capacity."""
        admitted_per_lane = [0] * self.NUM_LANES
        for raw_lane_id, raw_count in lane_counts.items():
            if isinstance(raw_lane_id, bool):
                raise ValueError(f"Invalid lane_id '{raw_lane_id}'. Expected 0..7.")
            lane_id = int(raw_lane_id)
            if not (0 <= lane_id < self.NUM_LANES):
                raise ValueError(
                    f"lane_id {lane_id} out of range [0, {self.NUM_LANES - 1}]."
                )
            if isinstance(raw_count, bool):
                raise ValueError("Vehicle count must be an integer.")
            requested_count = max(0, int(raw_count))
            admitted = self._lanes[lane_id].add_vehicles(
                count=requested_count,
                arrival_step=self._current_step,
            )
            admitted_per_lane[lane_id] += admitted

        self._last_manual_spawns = list(admitted_per_lane)
        self._mutation_seq += 1
        return admitted_per_lane

    def get_remaining_wait_times(self) -> list[float]:
        """Return wait durations (s) for all vehicles still queued across all lanes."""
        waits: list[float] = []
        for lane in self._lanes:
            waits.extend(
                lane.queued_wait_seconds(self._current_step, self.config.dt_seconds)
            )
        return waits

    def get_render_snapshot(self) -> dict[str, Any]:
        """Return immutable state snapshot for 2D UI rendering."""
        lane_queues = [lane.length for lane in self._lanes]
        lane_waits = [
            lane.max_wait_seconds(self._current_step, self.config.dt_seconds)
            for lane in self._lanes
        ]
        mean_wait = (
            self._cumulative_wait_seconds / self._total_throughput
            if self._total_throughput > 0
            else 0.0
        )
        executed_green_lanes = (
            tuple(int(lid) for lid in PHASE_GREEN_LANES[self._last_executed_phase])
            if self._last_executed_stage is SignalStage.GREEN
            else ()
        )
        return {
            "step": self._current_step,
            "max_steps": self.config.max_steps,
            "continuous_mode": bool(self._continuous_mode),
            "profile_name": self._active_profile.name,
            "active_phase": int(self._last_executed_phase),
            "phase_label": PHASE_LABELS[self._last_executed_phase],
            "signal_stage": self._last_executed_stage.value,
            "green_lanes": executed_green_lanes,
            "phase_green_lanes": tuple(
                int(lid) for lid in PHASE_GREEN_LANES[self._last_executed_phase]
            ),
            "lane_queues": lane_queues,
            "lane_max_waits_s": lane_waits,
            "step_arrivals_per_lane": list(self._last_step_arrivals),
            "step_departures_per_lane": list(self._last_step_departures),
            "manual_spawns_per_lane": list(self._last_manual_spawns),
            "max_lane_capacity": int(self._lane_capacity_limit),
            "max_total_capacity": int(self._lane_capacity_limit * self.NUM_LANES),
            "mutation_seq": int(self._mutation_seq),
            "total_queue": sum(lane_queues),
            "max_wait_time_s": max(lane_waits) if lane_waits else 0.0,
            "total_throughput": self._total_throughput,
            "total_switches": self._total_switches,
            "mean_wait_time_s": mean_wait,
            "episode_reward": float(self._episode_reward),
            "dt_seconds": float(self.config.dt_seconds),
            "all_red_steps": int(self._all_red_steps),
            "step_collisions": int(len(self._last_step_collisions)),
            "total_collisions": int(self._total_collisions),
            "step_collision_events": [dict(event) for event in self._last_step_collisions],
            "active_straggler_lanes": [
                int(straggler_lane) for straggler_lane, _ in self._box_stragglers
            ],
        }

    def _apply_signal_control(self, target_phase: SignalPhase) -> bool:
        """Forward phase command to the safety-interlocked signal controller."""
        return self._signal.request_phase(target_phase)

    def _step_departures(self) -> tuple[int, list[float], list[int]]:
        """Discharge queued vehicles from lanes currently receiving a GREEN signal."""
        departures_per_lane = [0] * self.NUM_LANES
        green_lanes = self._signal.green_lanes()
        if not green_lanes:
            return 0, [], departures_per_lane

        total_discharged = 0
        all_wait_times: list[float] = []

        for lane_id in green_lanes:
            is_turn = lane_id in (
                LaneId.N_TURN,
                LaneId.S_TURN,
                LaneId.E_TURN,
                LaneId.W_TURN,
            )
            capacity = self._sample_discharge_capacity(is_turn=is_turn)
            discharged_waits = self._lanes[int(lane_id)].discharge(
                capacity=capacity,
                current_step=self._current_step,
                dt_seconds=self.config.dt_seconds,
            )
            departed_count = len(discharged_waits)
            departures_per_lane[int(lane_id)] = departed_count
            total_discharged += departed_count
            all_wait_times.extend(discharged_waits)

        return total_discharged, all_wait_times, departures_per_lane

    def _sample_discharge_capacity(self, is_turn: bool) -> int:
        """Convert continuous vehicles/step discharge capacity to integer departures."""
        rate = (
            self.config.turn_discharge_rate
            if is_turn
            else self.config.straight_discharge_rate
        )
        base_capacity = int(rate)
        fractional_part = rate - base_capacity
        if fractional_part > 0.0 and float(self.np_random.random()) < fractional_part:
            return base_capacity + 1
        return base_capacity

    def _step_arrivals(self) -> list[int]:
        """Sample Poisson vehicle arrivals for all 8 incoming lanes and return admitted counts."""
        arrival_rates = self._active_profile.sample_rates(
            self._current_step,
            self.config.max_steps,
        )
        sampled_arrivals = self.np_random.poisson(arrival_rates)
        admitted_per_lane = [0] * self.NUM_LANES

        for lane_index in range(self.NUM_LANES):
            admitted_per_lane[lane_index] = self._lanes[lane_index].add_vehicles(
                count=int(sampled_arrivals[lane_index]),
                arrival_step=self._current_step,
            )

        return admitted_per_lane

    def _evaluate_collisions(
        self,
        phase_switched: bool,
        previous_departures: list[int],
        departures_per_lane: list[int],
    ) -> tuple[list[dict[str, Any]], list[float]]:
        """Evaluate box-clearance straggler and red-light violation conflicts."""
        if not self.config.enable_collisions:
            return [], []

        discharging_lanes = [
            lane_index for lane_index, count in enumerate(departures_per_lane) if count > 0
        ]
        events = self._evaluate_box_clearance_conflicts(
            phase_switched=phase_switched,
            previous_departures=previous_departures,
            discharging_lanes=discharging_lanes,
        )
        red_light_events, violation_waits = self._evaluate_red_light_conflicts(
            discharging_lanes=discharging_lanes,
        )
        events.extend(red_light_events)
        return events, violation_waits

    def _evaluate_box_clearance_conflicts(
        self,
        phase_switched: bool,
        previous_departures: list[int],
        discharging_lanes: list[int],
    ) -> list[dict[str, Any]]:
        """Track dilemma-zone stragglers across clearance stages and detect cross-phase hits."""
        if phase_switched:
            for lane_index, departed in enumerate(previous_departures):
                if (
                    departed > 0
                    and float(self.np_random.random()) < self.config.dilemma_zone_prob
                ):
                    self._box_stragglers.append([lane_index, 2])

        events: list[dict[str, Any]] = []
        remaining_stragglers: list[list[int]] = []

        for straggler_lane, remaining_steps in self._box_stragglers:
            if remaining_steps <= 0:
                continue

            conflicting_set = CONFLICTING_LANES[straggler_lane]
            conflicting_victims = [
                lane_index
                for lane_index in discharging_lanes
                if lane_index in conflicting_set
            ]
            if conflicting_victims:
                victim_index = int(
                    conflicting_victims[
                        int(self.np_random.integers(0, len(conflicting_victims)))
                    ]
                )
                events.append(
                    {
                        "step": int(self._current_step + 1),
                        "kind": "box_clearance_conflict",
                        "culprit_lane": int(straggler_lane),
                        "victim_lane": victim_index,
                        "culprit_wait_s": 0.0,
                    }
                )
                continue

            next_remaining = remaining_steps - 1
            if next_remaining > 0:
                remaining_stragglers.append([straggler_lane, next_remaining])

        self._box_stragglers = remaining_stragglers
        return events

    def _evaluate_red_light_conflicts(
        self,
        discharging_lanes: list[int],
    ) -> tuple[list[dict[str, Any]], list[float]]:
        """Evaluate starvation-induced red-light violations against discharging green lanes."""
        if self._signal.stage is not SignalStage.GREEN or not discharging_lanes:
            return [], []

        green_phase_lanes = {
            int(lane_id) for lane_id in PHASE_GREEN_LANES[self._signal.active_phase]
        }
        events: list[dict[str, Any]] = []
        violation_waits: list[float] = []

        for lane_index in range(self.NUM_LANES):
            if lane_index in green_phase_lanes or self._lanes[lane_index].length <= 0:
                continue

            conflicting_set = CONFLICTING_LANES[lane_index]
            conflicting_victims = [
                victim for victim in discharging_lanes if victim in conflicting_set
            ]
            if not conflicting_victims:
                continue

            head_wait_s = self._lanes[lane_index].max_wait_seconds(
                self._current_step,
                self.config.dt_seconds,
            )
            if head_wait_s <= self.config.red_light_wait_threshold_s:
                continue

            excess_ratio = (
                head_wait_s - self.config.red_light_wait_threshold_s
            ) / 140.0
            violation_prob = min(
                self.config.red_light_max_prob,
                excess_ratio * self.config.red_light_max_prob,
            )
            if float(self.np_random.random()) < violation_prob:
                discharged_waits = self._lanes[lane_index].discharge(
                    capacity=1,
                    current_step=self._current_step,
                    dt_seconds=self.config.dt_seconds,
                )
                violation_waits.extend(discharged_waits)
                victim_lane = int(
                    conflicting_victims[
                        int(self.np_random.integers(0, len(conflicting_victims)))
                    ]
                )
                events.append(
                    {
                        "step": int(self._current_step + 1),
                        "kind": "red_light_conflict",
                        "culprit_lane": int(lane_index),
                        "victim_lane": victim_lane,
                        "culprit_wait_s": round(float(head_wait_s), 1),
                    }
                )

        return events, violation_waits

    def _compute_reward(self, phase_switched: bool, collision_count: int = 0) -> float:
        """Compute scaled traffic control reward including switch and collision penalties."""
        total_queue = sum(lane.length for lane in self._lanes)
        max_wait_s = max(
            lane.max_wait_seconds(self._current_step, self.config.dt_seconds)
            for lane in self._lanes
        )
        raw_reward = (
            -float(total_queue)
            - self.config.wait_penalty_alpha * float(max_wait_s)
            - self.config.switch_penalty_beta * float(phase_switched)
            - self.config.collision_penalty * float(collision_count)
        )
        return float(raw_reward * self.config.reward_scale)

    def _build_observation(self) -> np.ndarray:
        """Build 22-dim normalized observation vector [queues, waits, phase, stage, elapsed]."""
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        queue_norm = float(self.config.queue_obs_norm_capacity)
        wait_norm = float(self.config.max_wait_norm_seconds)
        elapsed_norm = float(self.config.max_elapsed_norm_steps)

        for idx, lane in enumerate(self._lanes):
            obs[idx] = min(float(lane.length) / queue_norm, 2.0)
            wait_s = lane.max_wait_seconds(self._current_step, self.config.dt_seconds)
            obs[self.NUM_LANES + idx] = min(wait_s / wait_norm, 2.0)

        phase_offset = 2 * self.NUM_LANES
        obs[phase_offset + int(self._signal.active_phase)] = 1.0
        obs[phase_offset + self.NUM_PHASES] = 1.0 if self._signal.is_transitioning else 0.0
        obs[phase_offset + self.NUM_PHASES + 1] = min(
            float(self._signal.phase_elapsed) / elapsed_norm,
            2.0,
        )
        return obs

    def _build_info(
        self,
        step_throughput: int,
        step_wait_times: list[float],
        phase_switched: bool,
    ) -> dict[str, Any]:
        """Build factual step diagnostics dictionary."""
        lane_queues = [lane.length for lane in self._lanes]
        lane_waits = [
            lane.max_wait_seconds(self._current_step, self.config.dt_seconds)
            for lane in self._lanes
        ]
        return {
            "step": self._current_step,
            "profile_name": self._active_profile.name,
            "total_queue": sum(lane_queues),
            "max_wait_time_s": max(lane_waits) if lane_waits else 0.0,
            "step_throughput": int(step_throughput),
            "step_wait_times": list(step_wait_times),
            "phase_switched": bool(phase_switched),
            "total_throughput": int(self._total_throughput),
            "total_switches": int(self._total_switches),
            "active_phase": int(self._signal.active_phase),
            "signal_stage": self._signal.stage.value,
            "can_switch": bool(
                not self._signal.is_transitioning
                and self._signal.phase_elapsed >= self.config.min_green_steps
            ),
            "lane_queues": lane_queues,
            "lane_max_waits_s": lane_waits,
            "step_arrivals_per_lane": list(self._last_step_arrivals),
            "step_departures_per_lane": list(self._last_step_departures),
            "all_red_steps": int(self._all_red_steps),
            "step_collisions": int(len(self._last_step_collisions)),
            "total_collisions": int(self._total_collisions),
            "step_collision_events": [dict(event) for event in self._last_step_collisions],
            "active_straggler_lanes": [
                int(straggler_lane) for straggler_lane, _ in self._box_stragglers
            ],
        }
