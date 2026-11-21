"""Dynamic Poisson traffic arrival profiles for training and evaluation."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrafficProfile:
    """Named traffic arrival schedule returning per-lane Poisson rates at each step."""

    name: str
    description: str
    arrival_rates_fn: Callable[[int, int], np.ndarray]

    def sample_rates(self, step: int, max_steps: int) -> np.ndarray:
        """Evaluate 8-element arrival rate array (veh/step) at a given simulation step."""
        return self.arrival_rates_fn(step, max_steps)


class TrafficProfileFactory:
    """Factory creating standardized traffic arrival profiles across 8 intersection lanes."""

    AVAILABLE_PROFILES: tuple[str, ...] = (
        "normal",
        "asymmetric",
        "asymmetric_ew",
        "rush_hour",
        "mixed",
        'stress_test',
    )

    @classmethod
    def create(cls, name: str, rng: np.random.Generator | None = None) -> TrafficProfile:
        """Build a traffic arrival profile by name; 'mixed' randomizes per episode."""
        if name == "normal":
            return cls._build_constant_profile(
                name="normal",
                description="Balanced light-to-moderate flow across all four approaches",
                ns_straight=0.12,
                ns_turn=0.06,
                ew_straight=0.12,
                ew_turn=0.06,
            )

        if name == "asymmetric":
            return cls._build_constant_profile(
                name="asymmetric",
                description="Sustained heavy North-South corridor vs light East-West flow",
                ns_straight=0.34,
                ns_turn=0.14,
                ew_straight=0.06,
                ew_turn=0.04,
            )

        if name == "asymmetric_ew":
            return cls._build_constant_profile(
                name="asymmetric_ew",
                description="Sustained heavy East-West corridor vs light North-South flow",
                ns_straight=0.06,
                ns_turn=0.04,
                ew_straight=0.34,
                ew_turn=0.14,
            )

        if name == "rush_hour":
            return cls._build_rush_hour_profile()

        if name == "mixed":
            return cls._build_mixed_profile(rng=rng)

        if name == "stress_test":
            return cls._build_stress_test_profile()

        if name == "stress_train":
            return cls._build_stress_train_profile(rng=rng)
        
        raise ValueError(
            f"Unknown traffic profile '{name}'. Expected one of {cls.AVAILABLE_PROFILES}."
        )

    @staticmethod
    def _build_constant_profile(
        name: str,
        description: str,
        ns_straight: float,
        ns_turn: float,
        ew_straight: float,
        ew_turn: float,
        rate_scale: float = 1.0,
    ) -> TrafficProfile:
        """Create a stationary profile with lane order [N_S, N_T, S_S, S_T, E_S, E_T, W_S, W_T]."""
        rates = np.array(
            [
                ns_straight,
                ns_turn,
                ns_straight,
                ns_turn,
                ew_straight,
                ew_turn,
                ew_straight,
                ew_turn,
            ],
            dtype=np.float64,
        ) * rate_scale

        def arrival_fn(_step: int, _max_steps: int) -> np.ndarray:
            return rates.copy()

        return TrafficProfile(name=name, description=description, arrival_rates_fn=arrival_fn)

    @staticmethod
    def _build_rush_hour_profile(rate_scale: float = 1.0) -> TrafficProfile:
        """Create a 3-regime profile: baseline -> NS surge -> EW surge."""
        regime_moderate = np.array(
            [0.14, 0.07, 0.14, 0.07, 0.14, 0.07, 0.14, 0.07], dtype=np.float64
        ) * rate_scale
        regime_ns_surge = np.array(
            [0.38, 0.16, 0.38, 0.16, 0.08, 0.04, 0.08, 0.04], dtype=np.float64
        ) * rate_scale
        regime_ew_surge = np.array(
            [0.08, 0.04, 0.08, 0.04, 0.38, 0.16, 0.38, 0.16], dtype=np.float64
        ) * rate_scale

        def arrival_fn(step: int, max_steps: int) -> np.ndarray:
            cycle_len = max(3, max_steps)
            cycle_step = step % cycle_len
            first_cut = max(1, cycle_len // 3)
            second_cut = max(first_cut + 1, (2 * cycle_len) // 3)
            if cycle_step < first_cut:
                return regime_moderate.copy()
            if cycle_step < second_cut:
                return regime_ns_surge.copy()
            return regime_ew_surge.copy()

        return TrafficProfile(
            name="rush_hour",
            description="Dynamic surge shifting from balanced to NS heavy then EW heavy",
            arrival_rates_fn=arrival_fn,
        )

    @classmethod
    def _build_mixed_profile(
        cls,
        rng: np.random.Generator | None = None,
    ) -> TrafficProfile:
        """Sample a randomized traffic regime with mild rate jitter for RL training."""
        generator = rng or np.random.default_rng()
        choices = ("normal", "asymmetric", "asymmetric_ew", "rush_hour", "stress_train")
        weights = np.array([1.0, 1.0, 1.0, 1.0, 0.5], dtype=np.float64)
        probs = weights / weights.sum()

        chosen_name = str(generator.choice(choices, p=probs))
        scale = float(generator.uniform(0.85, 1.15))

        if chosen_name == "rush_hour":
            base_profile = cls._build_rush_hour_profile(rate_scale=scale)

        elif chosen_name == "stress_train":
            base_profile = cls._build_stress_train_profile(rng=generator)

        else:
            template = cls.create(chosen_name)
            base_rates = template.sample_rates(0, 300)
            base_profile = cls._build_constant_profile(
                name=f"mixed:{chosen_name}",
                description=f"Randomized training regime ({chosen_name}, scale={scale:.2f})",
                ns_straight=float(base_rates[0]),
                ns_turn=float(base_rates[1]),
                ew_straight=float(base_rates[4]),
                ew_turn=float(base_rates[5]),
                rate_scale=scale,
            )

        return TrafficProfile(
            name=f"mixed({chosen_name})",
            description=base_profile.description,
            arrival_rates_fn=base_profile.arrival_rates_fn,
        )

    @classmethod
    def _build_stress_test_profile(cls) -> TrafficProfile:
        """Create a high-volume profile for stress testing."""
        return cls._build_constant_profile(
            name="stress_test",
            description="Extreme traffic volume to test agent robustness",
            ns_straight=0.5,
            ns_turn=0.5,
            ew_straight=0.5,
            ew_turn=0.5,
        )

    @classmethod
    def _build_stress_train_profile(cls, rng: np.random.Generator | None = None) -> TrafficProfile:
        """Create a high-volume profile for stress testing."""
        generator = rng or np.random.default_rng()

        return cls._build_constant_profile(
            name="stress_train",
            description="Extreme traffic volume to test agent robustness",
            ns_straight=generator.uniform(0.35, 0.4),
            ns_turn=generator.uniform(0.35, 0.4),
            ew_straight=generator.uniform(0.35, 0.4),
            ew_turn=generator.uniform(0.35, 0.4),
        )