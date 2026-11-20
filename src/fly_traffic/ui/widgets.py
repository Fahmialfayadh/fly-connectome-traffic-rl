"""Minimal reusable Pygame UI widgets (slider and action button) with zero tooltips."""

from collections.abc import Callable
from dataclasses import dataclass

import pygame


@dataclass(frozen=True)
class UIPalette:
    """Muted dark-slate scientific color scheme."""

    bg: tuple[int, int, int] = (18, 22, 30)
    panel_bg: tuple[int, int, int] = (25, 31, 42)
    border: tuple[int, int, int] = (54, 65, 84)
    text: tuple[int, int, int] = (232, 237, 244)
    muted_text: tuple[int, int, int] = (156, 168, 184)
    accent: tuple[int, int, int] = (72, 168, 132)
    accent_hover: tuple[int, int, int] = (92, 192, 152)
    button_bg: tuple[int, int, int] = (38, 46, 62)
    button_hover: tuple[int, int, int] = (52, 64, 86)
    button_active: tuple[int, int, int] = (46, 114, 92)


class Slider:
    """Horizontal numeric parameter slider with direct mouse drag support."""

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        min_val: float,
        max_val: float,
        initial_val: float,
        on_change: Callable[[float], None] | None = None,
    ) -> None:
        self.rect = rect
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = self._clamp(initial_val)
        self.on_change = on_change
        self.dragging = False

    def set_value(self, new_value: float, notify: bool = False) -> None:
        """Programmatically update slider value within [min_val, max_val]."""
        self.value = self._clamp(new_value)
        if notify and self.on_change is not None:
            self.on_change(self.value)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Process mouse press, drag, and release events; return True if consumed."""
        hit_rect = self.rect.inflate(12, 16)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if hit_rect.collidepoint(event.pos):
                self.dragging = True
                self._update_from_x(event.pos[0])
                return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True

        if event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_from_x(event.pos[0])
            return True

        return False

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        palette: UIPalette,
    ) -> None:
        """Render label, value readout, track, filled span, and handle knob."""
        label_surf = font.render(f"{self.label}: {self.value:.2f}", True, palette.text)
        surface.blit(label_surf, (self.rect.x, self.rect.y - 22))

        track_y = self.rect.centery
        track_rect = pygame.Rect(self.rect.x, track_y - 3, self.rect.width, 6)
        pygame.draw.rect(surface, palette.border, track_rect, border_radius=3)

        ratio = (self.value - self.min_val) / max(self.max_val - self.min_val, 1e-6)
        fill_width = max(0, int(round(ratio * self.rect.width)))
        if fill_width > 0:
            fill_rect = pygame.Rect(self.rect.x, track_y - 3, fill_width, 6)
            pygame.draw.rect(surface, palette.accent, fill_rect, border_radius=3)

        knob_x = self.rect.x + fill_width
        knob_color = palette.accent_hover if self.dragging else palette.text
        pygame.draw.circle(surface, knob_color, (knob_x, track_y), 8)
        pygame.draw.circle(surface, palette.bg, (knob_x, track_y), 4)

    def _update_from_x(self, mouse_x: int) -> None:
        """Map horizontal cursor pixel position to [min_val, max_val]."""
        ratio = (mouse_x - self.rect.x) / max(self.rect.width, 1)
        clamped_ratio = min(max(ratio, 0.0), 1.0)
        new_value = self.min_val + clamped_ratio * (self.max_val - self.min_val)
        self.set_value(new_value, notify=True)

    def _clamp(self, val: float) -> float:
        return min(max(float(val), self.min_val), self.max_val)


class ActionButton:
    """Clickable rectangular button with crisp hover state and text label."""

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        on_click: Callable[[], None],
        is_active_fn: Callable[[], bool] | None = None,
    ) -> None:
        self.rect = rect
        self.label = label
        self.on_click = on_click
        self.is_active_fn = is_active_fn

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Invoke callback on primary click inside button bounds."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.on_click()
                return True
        return False

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        palette: UIPalette,
        mouse_pos: tuple[int, int],
    ) -> None:
        """Draw button background, border, and centered caption."""
        hovered = self.rect.collidepoint(mouse_pos)
        active = self.is_active_fn() if self.is_active_fn is not None else False

        if active:
            bg_color = palette.button_active
        elif hovered:
            bg_color = palette.button_hover
        else:
            bg_color = palette.button_bg

        pygame.draw.rect(surface, bg_color, self.rect, border_radius=6)
        pygame.draw.rect(surface, palette.border, self.rect, width=1, border_radius=6)

        text_surf = font.render(self.label, True, palette.text)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
