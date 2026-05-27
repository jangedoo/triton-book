# Shared assets

This directory holds assets shared across chapters: reusable SVG diagrams, Mermaid sources, and Observable JS (`ojs`) templates that chapter agents copy-paste into their own `.qmd` files.

## Layout

- `ojs/` — Reusable Observable JS snippets. Each file is a self-contained Quarto-friendly fragment you can drop into a chapter and edit in place. They are not rendered as standalone pages — they exist only as source material.

## Conventions

- Every snippet is written so it can be copied into a chapter with minimal edits — change input parameters at the top, leave the rendering code alone.
- Snippets do not depend on the rest of the chapter context — they should render on their own if you paste them into a fresh `.qmd`.
- No emojis. No marketing language. Comments explain what the visualization is teaching, not how clever it is.
- If you add a new shared asset, add a one-line entry below describing what it is for.

## Current shared assets

- `ojs/grid-mapping-template.qmd` — Slider over `BLOCK_SIZE` showing how a 1D tensor of length N is tiled by the grid. Use in any chapter that introduces a new grid layout.
- `ojs/online-softmax-template.qmd` — Slider stepping through the online-softmax / FlashAttention accumulator state (`m`, `l`, `acc`). Use in Chapter 14 and reference from Chapter 5 / Chapter 20.
