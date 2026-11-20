# Product

## Register

product

## Users

Machine learning researchers, computational neuroscience students, and reinforcement learning practitioners running local CPU experiments on a desktop or laptop monitor. Users open this workspace to inspect whether an extracted *Drosophila* sub-connectome can regulate traffic flow, watch live inference behavior, launch or monitor transparent PPO training runs, and test real-time neuro-pharmacological perturbations.

## Product Purpose

An interactive research workbench unifying a 2D traffic intersection simulation, real-time neural activation telemetry, transparent on-demand PPO training, and comparative benchmark inspection. Success means a user immediately understands whether the system is running Inference or Training, sees realistic vehicles obeying 3-lens traffic signals, reads clear variable-to-math documentation, and reproduces every experiment directly from the UI or CLI.

## Brand Personality

Precise, transparent, scientific. Instrument-grade clarity with zero decorative hype or opaque black boxes.

## Anti-references

- Generic dark-mode neon-glow "AI sci-fi" dashboards with unreadable low-contrast labels and decorative borders.
- Toy block visualizers where cars teleport instead of braking, queuing, and turning through real 3-lens signal heads.
- Opaque training scripts that hide hyperparameters, loss curves, and active checkpoint provenance behind terminal logs alone.

## Design Principles

1. **Unmistakable System State**: Always surface active mode (`INFERENCE` vs `TRAINING`), active checkpoint path, and simulation step timing in the primary header.
2. **Physical Legibility First**: Render real 3-lens traffic light housings, painted lane markings, brake lights, turn signals, and smooth vehicle kinematics matching backend environment counts 1-to-1.
3. **Zero Black-Box Training**: Expose exact hyperparameters, live episode reward/loss curves, and neural leak-rate evolution during CPU training.
4. **Honest Factual Telemetry**: Label every axis, table column, and slider strictly after the underlying variable or metric without speculative claims.

## Accessibility & Inclusion

- Target WCAG AA contrast (>= 4.5:1 body text, >= 3:1 UI controls) across all Light Instrument Lab surfaces.
- Preserve visible keyboard `:focus-visible` outlines, explicit `<label>` elements on all form inputs, `role="tablist"` keyboard navigation, and `@media (prefers-reduced-motion: reduce)` fallbacks.
- Distinguish traffic signals by both lens position (top=red, middle=amber, bottom=green) and color, and separate straight vs turn vehicle bodies visually.
