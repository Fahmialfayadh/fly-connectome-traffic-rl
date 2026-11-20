"""Unit tests for TrafficIntersectionEnv, signal safety interlocks, and profiles."""

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from fly_traffic.config import TrafficEnvConfig
from fly_traffic.environment.intersection_state import SignalPhase, SignalStage
from fly_traffic.environment.traffic_env import CONFLICTING_LANES, TrafficIntersectionEnv


def test_gymnasium_env_checker_passes() -> None:
    """Validate Gymnasium API contract via official check_env utility."""
    env = TrafficIntersectionEnv(TrafficEnvConfig(max_steps=40, profile_name="normal"))
    check_env(env, skip_render_check=True)


def test_min_green_and_yellow_all_red_safety_interlocks() -> None:
    """Verify phase switches obey min_green_steps and enforce YELLOW/ALL_RED clearance."""
    cfg = TrafficEnvConfig(
        max_steps=50,
        min_green_steps=3,
        yellow_steps=2,
        all_red_steps=1,
        profile_name="normal",
    )
    env = TrafficIntersectionEnv(config=cfg)
    obs, info = env.reset(seed=123)
    assert obs.shape == (22,)
    assert info["active_phase"] == int(SignalPhase.NS_STRAIGHT)

    # Attempt immediate switch on steps 0, 1, 2 -> rejected until min_green=3 steps finish
    for _ in range(3):
        _, _, _, _, info = env.step(int(SignalPhase.EW_STRAIGHT))
        assert not info["phase_switched"]
        assert info["signal_stage"] == SignalStage.GREEN.value

    # Step 4 (after 3 green steps): switch initiates YELLOW (2 steps) -> ALL_RED (1 step)
    _, _, _, _, info = env.step(int(SignalPhase.EW_STRAIGHT))
    assert info["phase_switched"]
    assert info["signal_stage"] == SignalStage.YELLOW.value
    assert info["step_throughput"] == 0

    _, _, _, _, info = env.step(int(SignalPhase.EW_STRAIGHT))
    assert info["signal_stage"] == SignalStage.ALL_RED.value
    assert info["step_throughput"] == 0

    _, _, _, _, info = env.step(int(SignalPhase.EW_STRAIGHT))
    assert info["signal_stage"] == SignalStage.GREEN.value
    assert info["active_phase"] == int(SignalPhase.EW_STRAIGHT)


def test_all_traffic_profiles_and_wait_times() -> None:
    """Verify all supported traffic profiles run cleanly and track non-negative wait times."""
    for profile in ("normal", "asymmetric", "rush_hour", "mixed"):
        env = TrafficIntersectionEnv(TrafficEnvConfig(max_steps=30, profile_name=profile))
        obs, _ = env.reset(seed=77)
        assert np.all(obs >= 0.0) and np.all(obs <= 2.0)

        for step_idx in range(30):
            action = (step_idx // 6) % 4
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward)
            assert obs.shape == (22,)
            assert info["max_wait_time_s"] >= 0.0
            assert (step_idx == 29) == truncated
            assert not terminated


def test_red_light_conflict_triggered_when_red_lane_starves() -> None:
    """Verify red_light_conflict occurs when a red lane exceeds red_light_wait_threshold_s."""
    cfg = TrafficEnvConfig(
        max_steps=120,
        dt_seconds=2.0,
        red_light_wait_threshold_s=10.0,
        red_light_max_prob=1.0,
        dilemma_zone_prob=0.0,
        enable_collisions=True,
        profile_name="rush_hour",
    )
    env = TrafficIntersectionEnv(config=cfg)
    _, init_info = env.reset(seed=7)
    assert init_info["total_collisions"] == 0

    red_light_events: list[dict] = []
    for _ in range(100):
        _, _, _, _, info = env.step(int(SignalPhase.NS_STRAIGHT))
        for event in info["step_collision_events"]:
            if event["kind"] == "red_light_conflict":
                red_light_events.append(event)

    assert len(red_light_events) > 0
    assert info["total_collisions"] == len(red_light_events)
    sample_event = red_light_events[0]
    assert sample_event["culprit_lane"] not in (0, 2)
    assert sample_event["victim_lane"] in (0, 2)
    assert sample_event["culprit_wait_s"] > 10.0


def test_box_clearance_conflict_and_all_red_safety_buffer() -> None:
    """Verify box_clearance_conflict occurs with all_red_steps=0 and clears with all_red_steps=1."""
    unsafe_cfg = TrafficEnvConfig(
        max_steps=20,
        min_green_steps=2,
        yellow_steps=1,
        all_red_steps=0,
        dilemma_zone_prob=1.0,
        red_light_max_prob=0.0,
        enable_collisions=True,
        profile_name="rush_hour",
    )
    unsafe_env = TrafficIntersectionEnv(config=unsafe_cfg)
    unsafe_env.reset(seed=99)
    unsafe_env._lanes[0].add_vehicles(count=6, arrival_step=0)
    unsafe_env._lanes[4].add_vehicles(count=6, arrival_step=0)

    unsafe_env.step(int(SignalPhase.NS_STRAIGHT))
    unsafe_env.step(int(SignalPhase.NS_STRAIGHT))
    _, _, _, _, yellow_info = unsafe_env.step(int(SignalPhase.EW_STRAIGHT))
    assert yellow_info["phase_switched"]
    assert yellow_info["step_throughput"] == 0
    assert yellow_info["signal_stage"] == SignalStage.GREEN.value

    _, _, _, _, collision_info = unsafe_env.step(int(SignalPhase.EW_STRAIGHT))
    assert collision_info["signal_stage"] == SignalStage.GREEN.value
    assert collision_info["step_throughput"] > 0
    assert collision_info["step_collisions"] >= 1
    assert any(
        event["kind"] == "box_clearance_conflict"
        for event in collision_info["step_collision_events"]
    )
    snapshot = unsafe_env.get_render_snapshot()
    assert snapshot["all_red_steps"] == 0
    assert snapshot["total_collisions"] >= 1

    safe_env = TrafficIntersectionEnv(config=unsafe_cfg)
    safe_env.set_safety_buffer(1)
    safe_env.reset(seed=99)
    safe_env._lanes[0].add_vehicles(count=6, arrival_step=0)
    safe_env._lanes[4].add_vehicles(count=6, arrival_step=0)

    safe_env.step(int(SignalPhase.NS_STRAIGHT))
    safe_env.step(int(SignalPhase.NS_STRAIGHT))
    _, _, _, _, safe_yellow_info = safe_env.step(int(SignalPhase.EW_STRAIGHT))
    assert safe_yellow_info["phase_switched"]
    assert safe_yellow_info["step_throughput"] == 0
    assert safe_yellow_info["signal_stage"] == SignalStage.ALL_RED.value

    _, _, _, _, all_red_info = safe_env.step(int(SignalPhase.EW_STRAIGHT))
    assert all_red_info["signal_stage"] == SignalStage.GREEN.value
    assert all_red_info["step_throughput"] == 0
    assert all_red_info["step_collisions"] == 0

    _, _, _, _, green_info = safe_env.step(int(SignalPhase.EW_STRAIGHT))
    assert green_info["signal_stage"] == SignalStage.GREEN.value
    assert green_info["step_throughput"] > 0
    assert green_info["total_collisions"] == 0
    assert green_info["all_red_steps"] == 1


def test_collision_penalty_reduces_step_reward() -> None:
    """Verify collision_penalty reduces reward on steps with collision events."""
    base_kwargs = dict(
        max_steps=20,
        min_green_steps=2,
        yellow_steps=1,
        all_red_steps=0,
        dilemma_zone_prob=1.0,
        red_light_max_prob=0.0,
        enable_collisions=True,
        profile_name="rush_hour",
    )
    env_penalized = TrafficIntersectionEnv(
        config=TrafficEnvConfig(collision_penalty=15.0, **base_kwargs)
    )
    env_unpenalized = TrafficIntersectionEnv(
        config=TrafficEnvConfig(collision_penalty=0.0, **base_kwargs)
    )

    for env in (env_penalized, env_unpenalized):
        env.reset(seed=99)
        env._lanes[0].add_vehicles(count=6, arrival_step=0)
        env._lanes[4].add_vehicles(count=6, arrival_step=0)
        env.step(int(SignalPhase.NS_STRAIGHT))
        env.step(int(SignalPhase.NS_STRAIGHT))
        env.step(int(SignalPhase.EW_STRAIGHT))

    _, reward_penalized, _, _, info_penalized = env_penalized.step(
        int(SignalPhase.EW_STRAIGHT)
    )
    _, reward_unpenalized, _, _, info_unpenalized = env_unpenalized.step(
        int(SignalPhase.EW_STRAIGHT)
    )

    assert info_penalized["step_collisions"] >= 1
    assert info_penalized["step_collisions"] == info_unpenalized["step_collisions"]
    expected_delta = (
        15.0 * float(info_penalized["step_collisions"]) * env_penalized.config.reward_scale
    )
    assert np.isclose(reward_unpenalized - reward_penalized, expected_delta)
    assert reward_penalized < reward_unpenalized


def test_lane_capacity_limit_manual_injection_and_observation_bounds() -> None:
    """Verify dynamic lane capacity clamping, manual injection caps, and Box(0, 2) obs."""
    env = TrafficIntersectionEnv(TrafficEnvConfig(max_steps=40, profile_name="normal"))
    initial_obs, _ = env.reset(seed=11)
    assert initial_obs.shape == (22,)

    initial_snapshot = env.get_render_snapshot()
    assert initial_snapshot["max_lane_capacity"] == 30
    assert initial_snapshot["max_total_capacity"] == 240
    assert initial_snapshot["manual_spawns_per_lane"] == [0] * 8
    seq_after_reset = initial_snapshot["mutation_seq"]
    assert seq_after_reset >= 1

    assert env.set_lane_capacity_limit(999) == 80
    assert env.set_lane_capacity_limit(2) == 10
    assert env.set_lane_capacity_limit(80) == 80

    # reset() must preserve the active lane capacity limit across all 8 lanes.
    reset_obs, _ = env.reset(seed=11)
    assert np.all(reset_obs >= 0.0) and np.all(reset_obs <= 2.0)
    preserved_snapshot = env.get_render_snapshot()
    assert preserved_snapshot["max_lane_capacity"] == 80
    assert preserved_snapshot["max_total_capacity"] == 640
    assert all(lane.max_capacity == 80 for lane in env._lanes)

    admitted = env.inject_vehicles({0: 80, 2: 120, 5: -3})
    assert admitted == [80, 0, 80, 0, 0, 0, 0, 0]

    saturated_admitted = env.inject_vehicles({0: 15, 2: 10})
    assert saturated_admitted == [0] * 8

    saturated_obs = env._build_observation()
    assert saturated_obs.shape == (22,)
    assert env.observation_space.contains(saturated_obs)
    assert np.all(saturated_obs >= 0.0) and np.all(saturated_obs <= 2.0)
    assert np.isclose(saturated_obs[0], 2.0)
    assert np.isclose(saturated_obs[2], 2.0)

    inject_snapshot = env.get_render_snapshot()
    assert inject_snapshot["lane_queues"][0] == 80
    assert inject_snapshot["lane_queues"][2] == 80
    assert inject_snapshot["manual_spawns_per_lane"] == [0] * 8
    assert inject_snapshot["mutation_seq"] > preserved_snapshot["mutation_seq"]

    # Injecting into an unsaturated lane populates manual_spawns_per_lane until next step.
    admitted_lane_1 = env.inject_vehicles({1: 7})
    assert admitted_lane_1[1] == 7
    assert env.get_render_snapshot()["manual_spawns_per_lane"][1] == 7

    seq_before_step = env.get_render_snapshot()["mutation_seq"]
    env.step(int(SignalPhase.NS_STRAIGHT))
    after_step_snapshot = env.get_render_snapshot()
    assert after_step_snapshot["manual_spawns_per_lane"] == [0] * 8
    assert after_step_snapshot["mutation_seq"] == seq_before_step + 1

    with pytest.raises(ValueError):
        env.inject_vehicles({8: 1})
    with pytest.raises(ValueError):
        env.inject_vehicles({-1: 1})


def test_collision_pairs_respect_geometric_conflict_matrix_and_snapshot_stages() -> None:
    """Verify CONFLICTING_LANES symmetry, non-crossing exclusions, and executed snapshot stages."""
    assert set(CONFLICTING_LANES.keys()) == set(range(8))
    for lane_id, partners in CONFLICTING_LANES.items():
        assert lane_id not in partners
        assert len(partners) == 4
        for partner in partners:
            assert lane_id in CONFLICTING_LANES[partner]

    # Same-phase pairs and geometrically separated cross-phase turns/straights must not conflict.
    non_crossing_pairs = (
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 1),
        (0, 5),
        (1, 6),
        (2, 3),
        (2, 7),
        (3, 4),
        (4, 5),
        (6, 7),
    )
    for lane_a, lane_b in non_crossing_pairs:
        assert lane_b not in CONFLICTING_LANES[lane_a]

    # Box-clearance straggler on lane 0 (N Straight) must not collide with lane 5 (E Turn),
    # because E Turn turns south onto W exit without crossing N Straight.
    cfg_no_red = TrafficEnvConfig(
        max_steps=30,
        min_green_steps=2,
        yellow_steps=1,
        all_red_steps=0,
        dilemma_zone_prob=1.0,
        red_light_max_prob=0.0,
        enable_collisions=True,
        profile_name="normal",
    )
    env_non_crossing = TrafficIntersectionEnv(config=cfg_no_red)
    env_non_crossing.reset(seed=55)
    env_non_crossing._lanes[0].add_vehicles(count=6, arrival_step=0)
    env_non_crossing.step(int(SignalPhase.NS_STRAIGHT))
    env_non_crossing.step(int(SignalPhase.NS_STRAIGHT))
    env_non_crossing.step(int(SignalPhase.EW_TURN))
    env_non_crossing._lanes[7].reset()
    env_non_crossing._lanes[5].add_vehicles(count=6, arrival_step=3)
    _, _, _, _, non_cross_info = env_non_crossing.step(int(SignalPhase.EW_TURN))
    assert non_cross_info["step_departures_per_lane"][5] > 0
    assert non_cross_info["step_departures_per_lane"][7] == 0
    assert non_cross_info["step_collisions"] == 0

    # Straggler on lane 0 MUST collide when crossing lane 4 (E Straight) discharges.
    env_crossing = TrafficIntersectionEnv(config=cfg_no_red)
    env_crossing.reset(seed=55)
    env_crossing._lanes[0].add_vehicles(count=6, arrival_step=0)
    env_crossing._lanes[4].add_vehicles(count=6, arrival_step=0)
    env_crossing.step(int(SignalPhase.NS_STRAIGHT))
    env_crossing.step(int(SignalPhase.NS_STRAIGHT))
    env_crossing.step(int(SignalPhase.EW_STRAIGHT))
    env_crossing._lanes[6].reset()
    _, _, _, _, cross_info = env_crossing.step(int(SignalPhase.EW_STRAIGHT))
    assert cross_info["step_collisions"] >= 1
    assert any(
        event["kind"] == "box_clearance_conflict"
        and event["culprit_lane"] == 0
        and event["victim_lane"] == 4
        for event in cross_info["step_collision_events"]
    )

    # Executed snapshot stages must preserve YELLOW and ALL_RED instead of skipping to post-step GREEN.
    cfg_safe = TrafficEnvConfig(
        max_steps=30,
        min_green_steps=2,
        yellow_steps=1,
        all_red_steps=1,
        dilemma_zone_prob=1.0,
        red_light_max_prob=0.0,
        enable_collisions=True,
        profile_name="rush_hour",
    )
    safe_env = TrafficIntersectionEnv(config=cfg_safe)
    safe_env.reset(seed=99)
    safe_env._lanes[0].add_vehicles(count=6, arrival_step=0)
    assert safe_env.get_render_snapshot()["signal_stage"] == SignalStage.GREEN.value

    safe_env.step(int(SignalPhase.NS_STRAIGHT))
    safe_env.step(int(SignalPhase.NS_STRAIGHT))

    safe_env.step(int(SignalPhase.EW_STRAIGHT))
    yellow_snapshot = safe_env.get_render_snapshot()
    assert yellow_snapshot["signal_stage"] == SignalStage.YELLOW.value
    assert yellow_snapshot["active_phase"] == int(SignalPhase.NS_STRAIGHT)
    assert yellow_snapshot["green_lanes"] == ()
    assert 0 in yellow_snapshot["active_straggler_lanes"]

    safe_env.step(int(SignalPhase.EW_STRAIGHT))
    all_red_snapshot = safe_env.get_render_snapshot()
    assert all_red_snapshot["signal_stage"] == SignalStage.ALL_RED.value
    assert all_red_snapshot["green_lanes"] == ()
    assert all_red_snapshot["total_collisions"] == 0

    safe_env.step(int(SignalPhase.EW_STRAIGHT))
    green_snapshot = safe_env.get_render_snapshot()
    assert green_snapshot["signal_stage"] == SignalStage.GREEN.value
    assert green_snapshot["active_phase"] == int(SignalPhase.EW_STRAIGHT)
    assert green_snapshot["green_lanes"] == (4, 6)
    assert green_snapshot["active_straggler_lanes"] == []
