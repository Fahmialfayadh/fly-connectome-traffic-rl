"""Left-panel 2D intersection renderer showing vehicles, signals, and factual metrics."""

from typing import Any

import pygame

from fly_traffic.environment.intersection_state import LaneId
from fly_traffic.ui.widgets import UIPalette


class IntersectionPanel:
    """Renders the 4-way, 2-lane traffic intersection and live state counters."""

    ROAD_COLOR = (40, 44, 54)
    CENTER_BOX_COLOR = (48, 53, 64)
    LANE_LINE_COLOR = (120, 128, 144)
    STOP_LINE_COLOR = (224, 228, 236)
    CAR_STRAIGHT_COLOR = (90, 175, 235)
    CAR_TURN_COLOR = (232, 168, 72)

    SIGNAL_RED = (220, 68, 68)
    SIGNAL_YELLOW = (236, 194, 62)
    SIGNAL_GREEN = (68, 200, 120)

    def __init__(self, panel_rect: pygame.Rect, palette: UIPalette) -> None:
        self.rect = panel_rect
        self.palette = palette
        self.cx = panel_rect.x + panel_rect.width // 2
        self.cy = panel_rect.y + 340
        self.road_half = 80
        self.lane_width = 40
        self.stop_offset = 92

    def draw(
        self,
        surface: pygame.Surface,
        snapshot: dict[str, Any],
        agent_name: str,
        title_font: pygame.font.Font,
        body_font: pygame.font.Font,
        small_font: pygame.font.Font,
    ) -> None:
        """Render header, roads, signal heads, queued vehicles, and metrics readout."""
        pygame.draw.rect(surface, self.palette.panel_bg, self.rect)
        pygame.draw.line(
            surface,
            self.palette.border,
            (self.rect.right - 1, self.rect.top),
            (self.rect.right - 1, self.rect.bottom),
            width=1,
        )

        self._draw_header(surface, snapshot, agent_name, title_font, small_font)
        self._draw_roads(surface, small_font)
        self._draw_signals(surface, snapshot)
        self._draw_queued_vehicles(surface, snapshot, small_font)
        self._draw_metrics_card(surface, snapshot, agent_name, body_font, small_font)

    def _draw_header(
        self,
        surface: pygame.Surface,
        snapshot: dict[str, Any],
        agent_name: str,
        title_font: pygame.font.Font,
        small_font: pygame.font.Font,
    ) -> None:
        title_surf = title_font.render(
            f"Traffic Simulation | Agent: {agent_name}",
            True,
            self.palette.text,
        )
        surface.blit(title_surf, (self.rect.x + 20, self.rect.y + 14))

        sub_text = (
            f"Profile: {snapshot['profile_name']}  |  "
            f"Step: {snapshot['step']}/{snapshot['max_steps']}  |  "
            f"Phase: {snapshot['phase_label']} ({snapshot['signal_stage']})"
        )
        sub_surf = small_font.render(sub_text, True, self.palette.muted_text)
        surface.blit(sub_surf, (self.rect.x + 20, self.rect.y + 42))

    def _draw_roads(
        self,
        surface: pygame.Surface,
        small_font: pygame.font.Font,
    ) -> None:
        top_y = self.rect.y + 72
        bottom_y = self.rect.y + 608
        left_x = self.rect.x + 20
        right_x = self.rect.right - 20

        # Vertical road corridor
        v_road = pygame.Rect(
            self.cx - self.road_half,
            top_y,
            2 * self.road_half,
            bottom_y - top_y,
        )
        pygame.draw.rect(surface, self.ROAD_COLOR, v_road)

        # Horizontal road corridor
        h_road = pygame.Rect(
            left_x,
            self.cy - self.road_half,
            right_x - left_x,
            2 * self.road_half,
        )
        pygame.draw.rect(surface, self.ROAD_COLOR, h_road)

        # Intersection box
        center_box = pygame.Rect(
            self.cx - self.road_half,
            self.cy - self.road_half,
            2 * self.road_half,
            2 * self.road_half,
        )
        pygame.draw.rect(surface, self.CENTER_BOX_COLOR, center_box)

        # Center dividers
        pygame.draw.line(
            surface,
            self.SIGNAL_YELLOW,
            (self.cx, top_y),
            (self.cx, self.cy - self.stop_offset),
            width=2,
        )
        pygame.draw.line(
            surface,
            self.SIGNAL_YELLOW,
            (self.cx, self.cy + self.stop_offset),
            (self.cx, bottom_y),
            width=2,
        )
        pygame.draw.line(
            surface,
            self.SIGNAL_YELLOW,
            (left_x, self.cy),
            (self.cx - self.stop_offset, self.cy),
            width=2,
        )
        pygame.draw.line(
            surface,
            self.SIGNAL_YELLOW,
            (self.cx + self.stop_offset, self.cy),
            (right_x, self.cy),
            width=2,
        )

        # Lane separators inside incoming approaches
        for offset in (-self.lane_width, self.lane_width):
            pygame.draw.line(
                surface,
                self.LANE_LINE_COLOR,
                (self.cx + offset, top_y),
                (self.cx + offset, self.cy - self.stop_offset),
                width=1,
            )
            pygame.draw.line(
                surface,
                self.LANE_LINE_COLOR,
                (self.cx + offset, self.cy + self.stop_offset),
                (self.cx + offset, bottom_y),
                width=1,
            )
            pygame.draw.line(
                surface,
                self.LANE_LINE_COLOR,
                (left_x, self.cy + offset),
                (self.cx - self.stop_offset, self.cy + offset),
                width=1,
            )
            pygame.draw.line(
                surface,
                self.LANE_LINE_COLOR,
                (self.cx + self.stop_offset, self.cy + offset),
                (right_x, self.cy + offset),
                width=1,
            )

        # Stop lines
        pygame.draw.line(
            surface,
            self.STOP_LINE_COLOR,
            (self.cx - self.road_half, self.cy - self.stop_offset),
            (self.cx, self.cy - self.stop_offset),
            width=3,
        )
        pygame.draw.line(
            surface,
            self.STOP_LINE_COLOR,
            (self.cx, self.cy + self.stop_offset),
            (self.cx + self.road_half, self.cy + self.stop_offset),
            width=3,
        )
        pygame.draw.line(
            surface,
            self.STOP_LINE_COLOR,
            (self.cx - self.stop_offset, self.cy),
            (self.cx - self.stop_offset, self.cy + self.road_half),
            width=3,
        )
        pygame.draw.line(
            surface,
            self.STOP_LINE_COLOR,
            (self.cx + self.stop_offset, self.cy - self.road_half),
            (self.cx + self.stop_offset, self.cy),
            width=3,
        )

        # Cardinal labels placed clearly outside road corridors without overlapping lane badges
        for label, pos in [
            ("NORTH", (self.cx + self.road_half + 10, top_y + 2)),
            ("SOUTH", (self.cx + self.road_half + 10, bottom_y - 18)),
            ("WEST", (left_x, self.cy - self.road_half - 22)),
            ("EAST", (right_x - 38, self.cy - self.road_half - 22)),
        ]:
            lbl = small_font.render(label, True, self.palette.muted_text)
            surface.blit(lbl, pos)

    def _draw_signals(self, surface: pygame.Surface, snapshot: dict[str, Any]) -> None:
        stage = snapshot["signal_stage"]
        phase_lanes = set(snapshot["phase_green_lanes"])

        lane_light_positions = {
            int(LaneId.N_STRAIGHT): (self.cx - 60, self.cy - self.stop_offset - 8),
            int(LaneId.N_TURN): (self.cx - 20, self.cy - self.stop_offset - 8),
            int(LaneId.S_TURN): (self.cx + 20, self.cy + self.stop_offset + 8),
            int(LaneId.S_STRAIGHT): (self.cx + 60, self.cy + self.stop_offset + 8),
            int(LaneId.E_STRAIGHT): (self.cx + self.stop_offset + 8, self.cy - 60),
            int(LaneId.E_TURN): (self.cx + self.stop_offset + 8, self.cy - 20),
            int(LaneId.W_TURN): (self.cx - self.stop_offset - 8, self.cy + 20),
            int(LaneId.W_STRAIGHT): (self.cx - self.stop_offset - 8, self.cy + 60),
        }

        for lane_id, center in lane_light_positions.items():
            if lane_id in phase_lanes:
                if stage == "GREEN":
                    color = self.SIGNAL_GREEN
                elif stage == "YELLOW":
                    color = self.SIGNAL_YELLOW
                else:
                    color = self.SIGNAL_RED
            else:
                color = self.SIGNAL_RED

            pygame.draw.circle(surface, (15, 18, 24), center, 9)
            pygame.draw.circle(surface, color, center, 6)

    def _draw_queued_vehicles(
        self,
        surface: pygame.Surface,
        snapshot: dict[str, Any],
        small_font: pygame.font.Font,
    ) -> None:
        queues: list[int] = snapshot["lane_queues"]

        top_y = self.rect.y + 72
        bottom_y = self.rect.y + 608
        left_x = self.rect.x + 20
        right_x = self.rect.right - 20

        # (lane_id, anchor_x, anchor_y, dx, dy, badge_x, badge_y, is_horizontal, is_turn)
        lane_specs: list[tuple[int, int, int, int, int, int, int, bool, bool]] = [
            (int(LaneId.N_STRAIGHT), self.cx - 60, self.cy - self.stop_offset - 24, 0, -1, self.cx - 60, top_y + 14, False, False),
            (int(LaneId.N_TURN), self.cx - 20, self.cy - self.stop_offset - 24, 0, -1, self.cx - 20, top_y + 14, False, True),
            (int(LaneId.S_STRAIGHT), self.cx + 60, self.cy + self.stop_offset + 24, 0, 1, self.cx + 60, bottom_y - 14, False, False),
            (int(LaneId.S_TURN), self.cx + 20, self.cy + self.stop_offset + 24, 0, 1, self.cx + 20, bottom_y - 14, False, True),
            (int(LaneId.E_STRAIGHT), self.cx + self.stop_offset + 24, self.cy - 60, 1, 0, right_x - 16, self.cy - 60, True, False),
            (int(LaneId.E_TURN), self.cx + self.stop_offset + 24, self.cy - 20, 1, 0, right_x - 16, self.cy - 20, True, True),
            (int(LaneId.W_STRAIGHT), self.cx - self.stop_offset - 24, self.cy + 60, -1, 0, left_x + 16, self.cy + 60, True, False),
            (int(LaneId.W_TURN), self.cx - self.stop_offset - 24, self.cy + 20, -1, 0, left_x + 16, self.cy + 20, True, True),
        ]

        max_visible = 8
        spacing = 16

        for lane_id, ax, ay, dx, dy, badge_x, badge_y, is_horiz, is_turn in lane_specs:
            count = queues[lane_id]
            car_color = self.CAR_TURN_COLOR if is_turn else self.CAR_STRAIGHT_COLOR
            draw_count = min(count, max_visible)

            for idx in range(draw_count):
                cx = ax + idx * spacing * dx
                cy = ay + idx * spacing * dy
                w, h = (12, 18) if not is_horiz else (18, 12)
                car_rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
                pygame.draw.rect(surface, car_color, car_rect, border_radius=3)

            badge_surf = small_font.render(str(count), True, self.palette.text)
            badge_rect = badge_surf.get_rect(center=(badge_x, badge_y))
            surface.blit(badge_surf, badge_rect)

    def _draw_metrics_card(
        self,
        surface: pygame.Surface,
        snapshot: dict[str, Any],
        agent_name: str,
        body_font: pygame.font.Font,
        small_font: pygame.font.Font,
    ) -> None:
        card_rect = pygame.Rect(self.rect.x + 16, self.rect.y + 622, self.rect.width - 32, 122)
        pygame.draw.rect(surface, self.palette.bg, card_rect, border_radius=8)
        pygame.draw.rect(surface, self.palette.border, card_rect, width=1, border_radius=8)

        metrics = [
            ("Total Queue", f"{snapshot['total_queue']} veh"),
            ("Mean Wait Time", f"{snapshot['mean_wait_time_s']:.1f} s"),
            ("Max Head Wait", f"{snapshot['max_wait_time_s']:.1f} s"),
            ("Total Throughput", f"{snapshot['total_throughput']} veh"),
            ("Signal Switches", f"{snapshot['total_switches']}"),
            ("Active Controller", agent_name),
        ]

        col_w = card_rect.width // 3
        row_h = 50
        for idx, (label, val) in enumerate(metrics):
            col = idx % 3
            row = idx // 3
            x = card_rect.x + 16 + col * col_w
            y = card_rect.y + 14 + row * row_h

            lbl_surf = small_font.render(label, True, self.palette.muted_text)
            val_surf = body_font.render(val, True, self.palette.text)
            surface.blit(lbl_surf, (x, y))
            surface.blit(val_surf, (x, y + 18))
