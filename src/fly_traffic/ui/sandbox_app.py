"""60 FPS Pygame interactive traffic simulation and in-silico pharmacology sandbox."""

from pathlib import Path

import numpy as np
import pygame

from fly_traffic.agents.base import BaseTrafficAgent, NeuralAgentAdapter
from fly_traffic.agents.baselines import load_policy_checkpoint
from fly_traffic.agents.fixed_timer import FixedTimerAgent
from fly_traffic.agents.fly_policy import FlyConnectomePolicy
from fly_traffic.config import (
    ConnectomeConfig,
    PPOConfig,
    SandboxUIConfig,
    TrafficEnvConfig,
)
from fly_traffic.connectome.artifact import ConnectomeArtifact
from fly_traffic.environment.traffic_env import TrafficIntersectionEnv
from fly_traffic.ui.connectome_panel import ConnectomePanel
from fly_traffic.ui.intersection_panel import IntersectionPanel
from fly_traffic.ui.widgets import ActionButton, Slider, UIPalette


class TrafficConnectomeSandboxApp:
    """Interactive Pygame application uniting 2D traffic control and live neural state."""

    SCENARIOS: tuple[str, ...] = ("normal", "asymmetric", "rush_hour")

    def __init__(
        self,
        ui_config: SandboxUIConfig | None = None,
        env_config: TrafficEnvConfig | None = None,
        connectome_path: Path | None = None,
        checkpoints_dir: Path | None = None,
    ) -> None:
        self.ui_config = ui_config or SandboxUIConfig()
        self.env_config = env_config or TrafficEnvConfig()
        self.connectome_path = connectome_path or ConnectomeConfig.output_path
        self.checkpoints_dir = checkpoints_dir or PPOConfig.checkpoints_dir
        self.palette = UIPalette()

        self.scenario_idx = 2  # Start on rush_hour for dynamic demonstration
        self.env = TrafficIntersectionEnv(
            config=self.env_config,
            profile_name=self.SCENARIOS[self.scenario_idx],
        )
        self.obs, _ = self.env.reset(seed=42)

        self.agents: list[BaseTrafficAgent] = self._load_available_agents()
        self.agent_idx = 0
        self.fly_policy = self._find_fly_policy()
        self.agents[self.agent_idx].reset()

        self.paused = False
        self.running = False
        self.status_message = "Ready | Drag sliders or click controls to perturb circuit."

        self._sliders, self._buttons = self._create_controls()
        self.left_panel = IntersectionPanel(
            panel_rect=pygame.Rect(
                0,
                0,
                self.ui_config.left_panel_width,
                self.ui_config.window_height,
            ),
            palette=self.palette,
        )
        self.right_panel = ConnectomePanel(
            panel_rect=pygame.Rect(
                self.ui_config.left_panel_width,
                0,
                self.ui_config.right_panel_width,
                self.ui_config.window_height,
            ),
            ui_config=self.ui_config,
            palette=self.palette,
            sliders=self._sliders,
            buttons=self._buttons,
        )

    @property
    def current_agent(self) -> BaseTrafficAgent:
        """Currently active controller driving the intersection."""
        return self.agents[self.agent_idx]

    @property
    def is_fly_active(self) -> bool:
        """Return True if the active controller wraps FlyConnectomePolicy."""
        agent = self.current_agent
        return isinstance(agent, NeuralAgentAdapter) and isinstance(
            agent.policy,
            FlyConnectomePolicy,
        )

    def step_simulation(self) -> None:
        """Advance traffic simulation by one discrete decision interval."""
        action = self.current_agent.select_action(self.obs, deterministic=True)
        next_obs, _reward, terminated, truncated, _info = self.env.step(action)
        if terminated or truncated:
            self.obs, _ = self.env.reset(
                options={"profile_name": self.SCENARIOS[self.scenario_idx]}
            )
            self.current_agent.reset()
        else:
            self.obs = next_obs

    def run(self) -> None:
        """Launch interactive 60 FPS Pygame window loop."""
        pygame.init()
        try:
            screen = pygame.display.set_mode(
                (self.ui_config.window_width, self.ui_config.window_height)
            )
            pygame.display.set_caption(self.ui_config.window_title)
            fonts = self._load_fonts()
            clock = pygame.time.Clock()

            self.running = True
            accumulator_s = 0.0
            step_interval_s = 1.0 / max(self.ui_config.sim_steps_per_second, 0.5)

            while self.running:
                dt_s = clock.tick(self.ui_config.target_fps) / 1000.0
                self._process_events()

                if not self.paused:
                    accumulator_s += dt_s
                    while accumulator_s >= step_interval_s:
                        self.step_simulation()
                        accumulator_s -= step_interval_s

                self._render_frame(screen, fonts)
                pygame.display.flip()
        finally:
            pygame.quit()

    def render_headless_screenshot(self, output_path: Path, steps: int = 45) -> Path:
        """Run N simulation steps headless and save a rendered PNG screenshot."""
        pygame.init()
        try:
            screen = pygame.Surface(
                (self.ui_config.window_width, self.ui_config.window_height)
            )
            fonts = self._load_fonts()
            for _ in range(steps):
                self.step_simulation()
            self._render_frame(screen, fonts)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(output_path))
            return output_path
        finally:
            pygame.quit()

    def _load_available_agents(self) -> list[BaseTrafficAgent]:
        loaded: list[BaseTrafficAgent] = []
        artifact = (
            ConnectomeArtifact.load(self.connectome_path)
            if self.connectome_path.exists()
            else None
        )

        fly_ckpt = self.checkpoints_dir / "fly_ppo.pt"
        if fly_ckpt.exists():
            policy, _ = load_policy_checkpoint(fly_ckpt, artifact=artifact)
            loaded.append(NeuralAgentAdapter(policy, "Fly Connectome PPO"))
        elif artifact is not None:
            loaded.append(
                NeuralAgentAdapter(
                    FlyConnectomePolicy(obs_dim=22, act_dim=4, artifact=artifact),
                    "Fly Connectome PPO (untrained)",
                )
            )

        for key, label in [("gru", "PPO-GRU"), ("mlp", "PPO-MLP")]:
            ckpt = self.checkpoints_dir / f"{key}_ppo.pt"
            if ckpt.exists():
                policy, _ = load_policy_checkpoint(ckpt)
                loaded.append(NeuralAgentAdapter(policy, label))

        loaded.append(FixedTimerAgent())
        return loaded

    def _find_fly_policy(self) -> FlyConnectomePolicy | None:
        for agent in self.agents:
            if isinstance(agent, NeuralAgentAdapter) and isinstance(
                agent.policy,
                FlyConnectomePolicy,
            ):
                return agent.policy
        return None

    def _create_controls(self) -> tuple[list[Slider], list[ActionButton]]:
        base_x = self.ui_config.left_panel_width + 24
        slider_width = self.ui_config.right_panel_width - 48

        g_exc_slider = Slider(
            rect=pygame.Rect(base_x, 474, slider_width, 20),
            label="g_exc (ACh Excitatory Gain)",
            min_val=self.ui_config.gain_min,
            max_val=self.ui_config.gain_max,
            initial_val=1.0,
            on_change=self._on_g_exc_change,
        )
        g_inh_slider = Slider(
            rect=pygame.Rect(base_x, 534, slider_width, 20),
            label="g_inh (GABA/Glu Inhibitory Gain)",
            min_val=self.ui_config.gain_min,
            max_val=self.ui_config.gain_max,
            initial_val=1.0,
            on_change=self._on_g_inh_change,
        )

        btn_w = 132
        btn_h = 38
        gap = 8
        lesion_specs = [
            ("Lesion CX 25%", self._on_lesion_cx),
            ("Lesion MB 25%", self._on_lesion_mb),
            ("Random 25%", self._on_lesion_random),
            ("Reset Brain", self._on_reset_brain),
        ]
        sim_specs = [
            ("Pause / Play", self._on_toggle_pause),
            ("Reset Sim", self._on_reset_sim),
            ("Next Profile", self._on_cycle_scenario),
            ("Next Agent", self._on_cycle_agent),
        ]

        buttons: list[ActionButton] = []
        for idx, (label, callback) in enumerate(lesion_specs):
            x = base_x + idx * (btn_w + gap)
            buttons.append(ActionButton(pygame.Rect(x, 588, btn_w, btn_h), label, callback))

        for idx, (label, callback) in enumerate(sim_specs):
            x = base_x + idx * (btn_w + gap)
            is_active = (lambda: self.paused) if idx == 0 else None
            buttons.append(
                ActionButton(
                    pygame.Rect(x, 646, btn_w, btn_h),
                    label,
                    callback,
                    is_active_fn=is_active,
                )
            )

        return [g_exc_slider, g_inh_slider], buttons

    def _on_g_exc_change(self, val: float) -> None:
        if self.fly_policy is not None:
            self.fly_policy.cell.set_pharmacology(val, self.fly_policy.cell.g_inh)
            self.status_message = f"Updated g_exc = {val:.2f}"

    def _on_g_inh_change(self, val: float) -> None:
        if self.fly_policy is not None:
            self.fly_policy.cell.set_pharmacology(self.fly_policy.cell.g_exc, val)
            self.status_message = f"Updated g_inh = {val:.2f}"

    def _on_lesion_cx(self) -> None:
        if self.fly_policy is not None:
            removed = self.fly_policy.cell.apply_region_lesion(
                circuit_groups=self.fly_policy.artifact.circuit_groups,
                target_group="CX",
                fraction=self.ui_config.lesion_fraction,
                seed=42,
            )
            self.status_message = f"Applied 25% CX lesion ({removed} neurons silenced)"

    def _on_lesion_mb(self) -> None:
        if self.fly_policy is not None:
            removed = self.fly_policy.cell.apply_region_lesion(
                circuit_groups=self.fly_policy.artifact.circuit_groups,
                target_group="MB",
                fraction=self.ui_config.lesion_fraction,
                seed=42,
            )
            self.status_message = f"Applied 25% MB lesion ({removed} neurons silenced)"

    def _on_lesion_random(self) -> None:
        if self.fly_policy is not None:
            removed = self.fly_policy.cell.apply_random_lesion(
                fraction=self.ui_config.lesion_fraction,
                seed=42,
            )
            self.status_message = f"Applied 25% random lesion ({removed} neurons silenced)"

    def _on_reset_brain(self) -> None:
        if self.fly_policy is not None:
            self.fly_policy.cell.set_pharmacology(1.0, 1.0)
            self.fly_policy.cell.reset_lesion()
        self._sliders[0].set_value(1.0, notify=False)
        self._sliders[1].set_value(1.0, notify=False)
        self.status_message = "Reset pharmacology (g_exc=1.0, g_inh=1.0) and cleared lesions"

    def _on_toggle_pause(self) -> None:
        self.paused = not self.paused
        self.status_message = "Simulation paused" if self.paused else "Simulation running"

    def _on_reset_sim(self) -> None:
        self.obs, _ = self.env.reset(
            seed=42,
            options={"profile_name": self.SCENARIOS[self.scenario_idx]},
        )
        self.current_agent.reset()
        self.status_message = f"Reset simulation ({self.SCENARIOS[self.scenario_idx]})"

    def _on_cycle_scenario(self) -> None:
        self.scenario_idx = (self.scenario_idx + 1) % len(self.SCENARIOS)
        profile_name = self.SCENARIOS[self.scenario_idx]
        self.env.set_profile(profile_name)
        self.status_message = f"Switched traffic profile to: {profile_name}"

    def _on_cycle_agent(self) -> None:
        self.agent_idx = (self.agent_idx + 1) % len(self.agents)
        self.current_agent.reset()
        self.status_message = f"Active controller: {self.current_agent.name}"

    def _process_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self._on_toggle_pause()
                continue

            for slider in self._sliders:
                if slider.handle_event(event):
                    break
            else:
                for button in self._buttons:
                    if button.handle_event(event):
                        break

    def _render_frame(
        self,
        surface: pygame.Surface,
        fonts: tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font],
    ) -> None:
        title_font, body_font, small_font = fonts
        snapshot = self.env.get_render_snapshot()
        mouse_pos = pygame.mouse.get_pos()

        hidden_np: np.ndarray | None = None
        if isinstance(self.current_agent, NeuralAgentAdapter) and self.is_fly_active:
            hidden_np = self.current_agent.latest_hidden_numpy

        self.left_panel.draw(
            surface=surface,
            snapshot=snapshot,
            agent_name=self.current_agent.name,
            title_font=title_font,
            body_font=body_font,
            small_font=small_font,
        )
        self.right_panel.draw(
            surface=surface,
            fly_policy=self.fly_policy,
            hidden_state=hidden_np,
            is_fly_active=self.is_fly_active,
            status_text=self.status_message,
            title_font=title_font,
            body_font=body_font,
            small_font=small_font,
            mouse_pos=mouse_pos,
        )

    @staticmethod
    def _load_fonts() -> tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font]:
        font_names = "dejavusans,liberationsans,arial"
        try:
            title_font = pygame.font.SysFont(font_names, 18, bold=True)
            body_font = pygame.font.SysFont(font_names, 15, bold=False)
            small_font = pygame.font.SysFont(font_names, 13, bold=False)
        except Exception:
            title_font = pygame.font.Font(None, 22)
            body_font = pygame.font.Font(None, 18)
            small_font = pygame.font.Font(None, 16)
        return title_font, body_font, small_font
