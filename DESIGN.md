# Design System: Apple Monochrome Instrument Lab

## Color Strategy & Palette (Apple Monochrome OKLCH)

- **Strategy**: Apple macOS Pro / iOS Human Interface Guidelines-inspired monochrome workbench. Minimalist, restrained, instrument-grade neutral grayscale palette with high legibility, subtle diffuse shadows, and zero saturated color distractions.
- **Surface Hierarchy**:
  - `--bg-canvas`: `oklch(0.968 0.002 260)` (`#F5F5F7`) — Apple System Gray 6 pro workspace canvas.
  - `--bg-surface`: `oklch(1.000 0.000 0)` (`#FFFFFF`) — pure Apple card surface.
  - `--bg-elevated`: `oklch(0.938 0.003 260)` (`#EBEBED`) — Apple System Gray 5 segmented control tray and inset wells.
  - `--border-subtle`: `oklch(0.915 0.003 260)` — Apple hairline divider (`#E5E5EA`).
  - `--border-medium`: `oklch(0.850 0.004 260)` — Apple control perimeter (`#D2D2D7`).
  - `--border-strong`: `oklch(0.680 0.006 260)` — active boundary (`#86868B`).
- **Text Hierarchy**:
  - `--ink-primary`: `oklch(0.185 0.004 260)` (`#1D1D1F`) — Apple Midnight Ink, >= 14:1 contrast.
  - `--ink-secondary`: `oklch(0.485 0.006 260)` (`#6E6E73`) — Apple SF Gray, >= 5.5:1 contrast.
  - `--ink-muted`: `oklch(0.650 0.005 260)` (`#8E8E93`) — Apple Tertiary Gray.
  - `--ink-on-fill`: `oklch(0.995 0.000 0)` (`#FFFFFF`) — white text on filled primary buttons.
- **Brand & Action Elements**:
  - Primary button / action: Solid Apple Carbon (`#1D1D1F`) with white text.
  - Accent / secondary button: Space Gray / Titanium (`#48484A`).
  - Segmented tab control: elevated white capsule (`#FFFFFF`) with Apple subtle diffuse shadow over gray tray.
- **Semantic Traffic, Multi-Agent Benchmark, & Telemetry Palette**:
  - Benchmark bar charts:
    - Fly Connectome PPO (Hero model): Apple System Green (`#34C759` / `oklch(0.680 0.170 142)`)
    - PPO-MLP: Apple System Blue (`#007AFF` / `oklch(0.580 0.180 245)`)
    - PPO-GRU: Apple System Purple (`#AF52DE` / `oklch(0.620 0.180 300)`)
    - Fixed Timer: Apple System Slate (`#8E8E93` / `oklch(0.600 0.010 260)`)
  - Training loss & reward telemetry curves:
    - `mean_episode_reward`: Apple System Orange (`#FF9500`)
    - `value_loss`: Apple System Blue (`#007AFF`)
    - `entropy`: Apple System Green (`#34C759`)
  - Traffic signals & intersection:
    - `--signal-green`: Apple System Green (`#28CD41`)
    - `--signal-amber`: Apple System Amber (`#FF9500`)
    - `--signal-red`: Apple System Red (`#FF3B30`)
    - Lajur Lurus (Straight): Blue accent (`#007AFF`)
    - Lajur Belok Kiri (Turn): Amber accent (`#FF9500`)
    - Vehicles on road: diverse distinct finishes (`#2563EB` Sapphire, `#DC2626` Crimson, `#D97706` Amber, `#16A34A` Emerald, `#1D1D1F` Midnight, `#475569` Slate)
  - Connectome diverging heatmap: cool cobalt blue (`#2563EB`) for GABA/Glu inhibitory to warm vermilion (`#EA580C`) for ACh excitatory with dark graphite neutral zero.

## Typography

- **Primary UI Sans**: `-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'SF Pro Display', 'IBM Plex Sans', system-ui, sans-serif`
- **Tabular Telemetry Mono**: `'SF Mono', 'IBM Plex Mono', ui-monospace, 'SFMono-Regular', monospace` with `font-variant-numeric: tabular-nums`.
- **Fixed Scale (`rem`)**:
  - Caption / Table Metadata: `0.75rem`
  - Secondary Labels / Controls: `0.875rem`
  - Body Copy: `1.00rem`
  - Panel Subheadings: `1.125rem`
  - Header Product Title: `1.25rem`

## Spacing, Elevation, & Component Rules

- 4px base spacing scale (`4px`, `8px`, `12px`, `16px`, `24px`, `32px`).
- Panel border radius: `12px`; control/button radius: `6px`; segmented tabs: `9px`; status pill radius: `999px`.
- Elevation uses Apple diffuse multi-layer drop shadows (`--shadow-card`, `--shadow-subtle`, `--shadow-segmented`, `--shadow-tooltip`).
- Zero `border-left`/`border-right` accent stripes, zero gradient text, zero emojis, zero intrusive popups.

## Primary Views

1. **Live Simulation**: Split workspace (`minmax(540px, 1.15fr)` 2D Canvas Intersection + `minmax(400px, 0.85fr)` Connectome Activity, Action Policy Probabilities, and In-Silico Pharmacology/Lesion controls).
2. **Training Studio**: Transparent CPU PPO configuration form, live WebSocket status metrics, real-time dual Canvas charts (`mean_episode_reward` and loss/entropy), and per-update log table.
3. **Benchmark & Connectome**: Sub-connectome extraction summary (`FAFB v783`), multi-scenario comparison table + Canvas bar charts, pharmacology/lesion table, and variable-to-equation glossary.
