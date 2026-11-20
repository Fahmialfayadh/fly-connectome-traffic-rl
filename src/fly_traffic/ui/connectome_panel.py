"""Right-panel live 50x50 Drosophila connectome heatmap and interactive lab controls."""

import numpy as np
import pygame

from fly_traffic.agents.fly_policy import FlyConnectomePolicy
from fly_traffic.config import SandboxUIConfig
from fly_traffic.ui.widgets import ActionButton, Slider, UIPalette


class ConnectomePanel:
    """Displays 2,500-neuron neural heatmap and pharmacology/lesion sliders & buttons."""

    def __init__(
        self,
        panel_rect: pygame.Rect,
        ui_config: SandboxUIConfig,
        palette: UIPalette,
        sliders: list[Slider],
        buttons: list[ActionButton],
    ) -> None:
        self.rect = panel_rect
        self.ui_config = ui_config
        self.palette = palette
        self.sliders = sliders
        self.buttons = buttons

        self.heatmap_rect = pygame.Rect(
            panel_rect.x + 24,
            panel_rect.y + 68,
            ui_config.heatmap_display_size,
            ui_config.heatmap_display_size,
        )
        self._colorbar_surface = self._build_colorbar_surface(width=18, height=180)

    def draw(
        self,
        surface: pygame.Surface,
        fly_policy: FlyConnectomePolicy | None,
        hidden_state: np.ndarray | None,
        is_fly_active: bool,
        status_text: str,
        title_font: pygame.font.Font,
        body_font: pygame.font.Font,
        small_font: pygame.font.Font,
        mouse_pos: tuple[int, int],
    ) -> None:
        """Render connectome heatmap, colorbar, circuit summary, and interactive controls."""
        pygame.draw.rect(surface, self.palette.panel_bg, self.rect)

        header = title_font.render(
            "Connectome Activity (50 x 50 Neurons) & Lab Controls",
            True,
            self.palette.text,
        )
        surface.blit(header, (self.rect.x + 24, self.rect.y + 14))

        sub = small_font.render(
            "Top bands (rows 0-35): CX [PB/EB/FB/NO]  |  Bottom bands (rows 36-49): MB",
            True,
            self.palette.muted_text,
        )
        surface.blit(sub, (self.rect.x + 24, self.rect.y + 42))

        self._draw_heatmap(
            surface=surface,
            fly_policy=fly_policy,
            hidden_state=hidden_state,
            is_fly_active=is_fly_active,
            body_font=body_font,
        )
        self._draw_sidebar_legend(
            surface=surface,
            fly_policy=fly_policy,
            is_fly_active=is_fly_active,
            small_font=small_font,
        )

        for slider in self.sliders:
            slider.draw(surface, body_font, self.palette)

        for button in self.buttons:
            button.draw(surface, small_font, self.palette, mouse_pos)

        status_surf = small_font.render(status_text, True, self.palette.muted_text)
        surface.blit(status_surf, (self.rect.x + 24, self.rect.y + 716))

    def _draw_heatmap(
        self,
        surface: pygame.Surface,
        fly_policy: FlyConnectomePolicy | None,
        hidden_state: np.ndarray | None,
        is_fly_active: bool,
        body_font: pygame.font.Font,
    ) -> None:
        pygame.draw.rect(surface, (12, 15, 20), self.heatmap_rect)

        if is_fly_active and fly_policy is not None and hidden_state is not None:
            lesion_mask = fly_policy.cell.lesion_mask.squeeze(0).detach().cpu().numpy()
            rgb_grid = self._activations_to_rgb(
                activations=hidden_state,
                lesion_mask=lesion_mask,
                grid_side=self.ui_config.heatmap_grid_side,
            )
            # surfarray expects (width, height, 3), so transpose (H, W, 3) -> (W, H, 3)
            raw_surf = pygame.surfarray.make_surface(np.transpose(rgb_grid, (1, 0, 2)))
            scaled = pygame.transform.scale(
                raw_surf,
                (self.heatmap_rect.width, self.heatmap_rect.height),
            )
            surface.blit(scaled, self.heatmap_rect.topleft)

            # Subtle divider line between CX (1800 = 36 rows) and MB (700 = 14 rows)
            cx_rows = int(1800 // self.ui_config.heatmap_grid_side)
            split_y = self.heatmap_rect.y + int(
                (cx_rows / self.ui_config.heatmap_grid_side) * self.heatmap_rect.height
            )
            pygame.draw.line(
                surface,
                (210, 218, 230),
                (self.heatmap_rect.left, split_y),
                (self.heatmap_rect.right, split_y),
                width=1,
            )
        else:
            msg_1 = body_font.render("Baseline Agent Active", True, self.palette.text)
            msg_2 = body_font.render(
                "Switch to Fly Connectome PPO for heatmap",
                True,
                self.palette.muted_text,
            )
            surface.blit(
                msg_1,
                msg_1.get_rect(center=(self.heatmap_rect.centerx, self.heatmap_rect.centery - 14)),
            )
            surface.blit(
                msg_2,
                msg_2.get_rect(center=(self.heatmap_rect.centerx, self.heatmap_rect.centery + 14)),
            )

        pygame.draw.rect(surface, self.palette.border, self.heatmap_rect, width=1)

    def _draw_sidebar_legend(
        self,
        surface: pygame.Surface,
        fly_policy: FlyConnectomePolicy | None,
        is_fly_active: bool,
        small_font: pygame.font.Font,
    ) -> None:
        side_x = self.heatmap_rect.right + 18
        side_y = self.heatmap_rect.y

        surface.blit(self._colorbar_surface, (side_x, side_y + 8))
        for label, dy in [("+0.4", 4), (" 0.0", 90), ("-0.4", 176)]:
            lbl = small_font.render(label, True, self.palette.muted_text)
            surface.blit(lbl, (side_x + 26, side_y + dy))

        info_y = side_y + 212
        if is_fly_active and fly_policy is not None:
            active_cnt = fly_policy.cell.active_neuron_count()
            total_cnt = fly_policy.cell.num_neurons
            tau_val = float(fly_policy.cell.leak_rate.item())
            lines = [
                f"Active: {active_cnt}/{total_cnt}",
                f"g_exc:  {fly_policy.cell.g_exc:.2f}",
                f"g_inh:  {fly_policy.cell.g_inh:.2f}",
                f"tau:    {tau_val:.3f}",
                "CX:     1800 neurons",
                "MB:      700 neurons",
                "Dark:   Lesioned",
            ]
        else:
            lines = [
                "Active: Baseline",
                "Connectome controls",
                "apply to Fly PPO",
            ]

        for idx, text in enumerate(lines):
            txt_surf = small_font.render(text, True, self.palette.text)
            surface.blit(txt_surf, (side_x, info_y + idx * 20))

    @staticmethod
    def _activations_to_rgb(
        activations: np.ndarray,
        lesion_mask: np.ndarray,
        grid_side: int,
    ) -> np.ndarray:
        """Vectorized diverging colormap from [-0.4, +0.4] neural state to uint8 RGB."""
        total_cells = grid_side * grid_side
        vals = np.clip(activations[:total_cells] / 0.4, -1.0, 1.0)
        mask = lesion_mask[:total_cells] > 0.5

        base = np.array([24.0, 30.0, 41.0], dtype=np.float32)
        pos_color = np.array([237.0, 137.0, 54.0], dtype=np.float32)
        neg_color = np.array([49.0, 130.0, 206.0], dtype=np.float32)
        dead_color = np.array([10.0, 12, 16.0], dtype=np.float32)

        rgb = np.broadcast_to(base, (total_cells, 3)).copy()
        pos_idx = vals >= 0.0
        neg_idx = ~pos_idx

        rgb[pos_idx] = base + vals[pos_idx, None] * (pos_color - base)
        rgb[neg_idx] = base + (-vals[neg_idx, None]) * (neg_color - base)
        rgb[~mask] = dead_color

        return np.clip(rgb, 0.0, 255.0).astype(np.uint8).reshape(grid_side, grid_side, 3)

    @staticmethod
    def _build_colorbar_surface(width: int, height: int) -> pygame.Surface:
        """Pre-render vertical [-1, +1] legend gradient matching heatmap colors."""
        vals = np.linspace(1.0, -1.0, height, dtype=np.float32)
        base = np.array([24.0, 30.0, 41.0], dtype=np.float32)
        pos_color = np.array([237.0, 137.0, 54.0], dtype=np.float32)
        neg_color = np.array([49.0, 130.0, 206.0], dtype=np.float32)

        col = np.zeros((height, 3), dtype=np.float32)
        pos_mask = vals >= 0.0
        col[pos_mask] = base + vals[pos_mask, None] * (pos_color - base)
        col[~pos_mask] = base + (-vals[~pos_mask, None]) * (neg_color - base)
        img = np.repeat(col[:, None, :], width, axis=1).astype(np.uint8)
        surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        pygame.draw.rect(surf, (70, 82, 102), surf.get_rect(), width=1)
        return surf
