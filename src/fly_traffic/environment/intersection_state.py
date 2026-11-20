"""Lane queues and safety-interlocked traffic signal state machine."""

from collections import deque
from enum import Enum, IntEnum


class LaneId(IntEnum):
    """Canonical ordering of the eight incoming lanes at the 4-way intersection."""

    N_STRAIGHT = 0
    N_TURN = 1
    S_STRAIGHT = 2
    S_TURN = 3
    E_STRAIGHT = 4
    E_TURN = 5
    W_STRAIGHT = 6
    W_TURN = 7


class SignalPhase(IntEnum):
    """Non-conflicting 4-phase movement plan."""

    NS_STRAIGHT = 0
    NS_TURN = 1
    EW_STRAIGHT = 2
    EW_TURN = 3


class SignalStage(Enum):
    """Active safety stage within a signal phase lifecycle."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ALL_RED = "ALL_RED"


PHASE_GREEN_LANES: dict[SignalPhase, tuple[LaneId, LaneId]] = {
    SignalPhase.NS_STRAIGHT: (LaneId.N_STRAIGHT, LaneId.S_STRAIGHT),
    SignalPhase.NS_TURN: (LaneId.N_TURN, LaneId.S_TURN),
    SignalPhase.EW_STRAIGHT: (LaneId.E_STRAIGHT, LaneId.W_STRAIGHT),
    SignalPhase.EW_TURN: (LaneId.E_TURN, LaneId.W_TURN),
}

PHASE_LABELS: dict[SignalPhase, str] = {
    SignalPhase.NS_STRAIGHT: "NS Straight",
    SignalPhase.NS_TURN: "NS Turn",
    SignalPhase.EW_STRAIGHT: "EW Straight",
    SignalPhase.EW_TURN: "EW Turn",
}


class LaneQueue:
    """FIFO queue of queued vehicle arrival steps enforcing finite lane capacity."""

    def __init__(self, max_capacity: int) -> None:
        self.max_capacity = max_capacity
        self._arrivals: deque[int] = deque()

    def reset(self) -> None:
        """Clear all queued vehicles."""
        self._arrivals.clear()

    @property
    def length(self) -> int:
        """Current number of queued vehicles in this lane."""
        return len(self._arrivals)

    def max_wait_seconds(self, current_step: int, dt_seconds: float) -> float:
        """Return wait duration in seconds of the head-of-line vehicle."""
        if not self._arrivals:
            return 0.0
        oldest_arrival = self._arrivals[0]
        return max(0.0, float(current_step - oldest_arrival) * dt_seconds)

    def queued_wait_seconds(self, current_step: int, dt_seconds: float) -> list[float]:
        """Return wait durations (s) of all vehicles currently waiting in this lane."""
        return [
            max(0.0, float(current_step - arrival_step) * dt_seconds)
            for arrival_step in self._arrivals
        ]

    def add_vehicles(self, count: int, arrival_step: int) -> int:
        """Append up to remaining lane capacity and return actual vehicles admitted."""
        if count <= 0:
            return 0

        available = max(0, self.max_capacity - len(self._arrivals))
        admitted = min(count, available)
        for _ in range(admitted):
            self._arrivals.append(arrival_step)
        return admitted

    def discharge(
        self,
        capacity: int,
        current_step: int,
        dt_seconds: float,
    ) -> list[float]:
        """Release up to `capacity` vehicles and return their individual wait times (s)."""
        if capacity <= 0 or not self._arrivals:
            return []

        wait_times: list[float] = []
        departed = min(capacity, len(self._arrivals))
        for _ in range(departed):
            arrival_step = self._arrivals.popleft()
            wait_times.append(max(0.0, float(current_step - arrival_step) * dt_seconds))

        return wait_times


class SignalController:
    """Enforces minimum green duration and yellow/all-red intergreen clearance buffers."""

    def __init__(
        self,
        min_green_steps: int,
        yellow_steps: int,
        all_red_steps: int,
    ) -> None:
        self.min_green_steps = min_green_steps
        self.yellow_steps = yellow_steps
        self.all_red_steps = all_red_steps

        self.active_phase: SignalPhase = SignalPhase.NS_STRAIGHT
        self.pending_phase: SignalPhase | None = None
        self.stage: SignalStage = SignalStage.GREEN
        self.phase_elapsed: int = 0
        self.stage_elapsed: int = 0

    def reset(self, initial_phase: SignalPhase = SignalPhase.NS_STRAIGHT) -> None:
        """Reset signal state to GREEN on the specified initial phase."""
        self.active_phase = initial_phase
        self.pending_phase = None
        self.stage = SignalStage.GREEN
        self.phase_elapsed = 0
        self.stage_elapsed = 0

    @property
    def is_transitioning(self) -> bool:
        """Return True during YELLOW or ALL_RED intergreen clearance stages."""
        return self.stage is not SignalStage.GREEN

    def green_lanes(self) -> tuple[LaneId, ...]:
        """Return lanes permitted to discharge during the current simulation step."""
        if self.stage is not SignalStage.GREEN:
            return ()
        return PHASE_GREEN_LANES[self.active_phase]

    def request_phase(self, target_phase: SignalPhase) -> bool:
        """Initiate a phase transition if min-green is satisfied; return True if switched."""
        if self.is_transitioning:
            return False
        if target_phase == self.active_phase:
            return False
        if self.phase_elapsed < self.min_green_steps:
            return False

        self.pending_phase = target_phase
        if self.yellow_steps > 0:
            self.stage = SignalStage.YELLOW
            self.stage_elapsed = 0
        elif self.all_red_steps > 0:
            self.stage = SignalStage.ALL_RED
            self.stage_elapsed = 0
        else:
            self._activate_pending_phase()
        return True

    def step(self) -> None:
        """Advance signal timers by one step and complete pending transitions."""
        if self.stage is SignalStage.GREEN:
            self.phase_elapsed += 1
            self.stage_elapsed += 1
            return

        self.stage_elapsed += 1
        if self.stage is SignalStage.YELLOW and self.stage_elapsed >= self.yellow_steps:
            if self.all_red_steps > 0:
                self.stage = SignalStage.ALL_RED
                self.stage_elapsed = 0
            else:
                self._activate_pending_phase()
            return

        if self.stage is SignalStage.ALL_RED and self.stage_elapsed >= self.all_red_steps:
            self._activate_pending_phase()

    def _activate_pending_phase(self) -> None:
        """Switch active phase to pending target and reset phase timers."""
        if self.pending_phase is not None:
            self.active_phase = self.pending_phase
        self.pending_phase = None
        self.stage = SignalStage.GREEN
        self.phase_elapsed = 0
        self.stage_elapsed = 0
