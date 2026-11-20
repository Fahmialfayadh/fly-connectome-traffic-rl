"""Static fixed-cycle baseline traffic signal controller."""

import numpy as np

from fly_traffic.agents.base import BaseTrafficAgent


class FixedTimerAgent(BaseTrafficAgent):
    """Cycles through the 4 signal phases on fixed green durations."""

    def __init__(
        self,
        phase_durations: tuple[int, int, int, int] = (8, 4, 8, 4),
        max_elapsed_norm_steps: float = 20.0,
        agent_name: str = "Fixed Timer",
    ) -> None:
        if len(phase_durations) != 4:
            raise ValueError("phase_durations must contain exactly 4 phase durations.")
        self.phase_durations = phase_durations
        self.max_elapsed_norm_steps = max_elapsed_norm_steps
        self._agent_name = agent_name

    @property
    def name(self) -> str:
        """Human-readable agent identifier."""
        return self._agent_name

    def reset(self) -> None:
        """Fixed timer is stateless given environment phase timers."""

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Advance to the next phase once the current green duration reaches its target."""
        _ = deterministic
        phase_one_hot = obs[16:20]
        active_phase = int(np.argmax(phase_one_hot))
        is_transitioning = float(obs[20]) > 0.5

        if is_transitioning:
            return active_phase

        elapsed_steps = round(float(obs[21]) * self.max_elapsed_norm_steps)
        if elapsed_steps >= self.phase_durations[active_phase]:
            return (active_phase + 1) % 4

        return active_phase
