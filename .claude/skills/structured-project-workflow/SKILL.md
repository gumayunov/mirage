---
name: structured-project-workflow
description: >
  Structured naming and organization for designs, plans, phases, and tasks when using superpowers
  brainstorming/writing-plans workflow. Use INSTEAD of default superpowers file naming when creating
  design documents, implementation plans, phase files, or tracking progress. Activates when:
  (1) Starting a new design/brainstorm session, (2) Writing implementation plans,
  (3) Creating phase breakdowns, (4) Tracking task progress. Overrides default superpowers path
  conventions (docs/plans/YYYY-MM-DD-*) with numbered directory-based structure.
---

# Structured Project Workflow

Replaces default superpowers file naming with a numbered, directory-based structure.
All artifacts live under `docs/designs/` in the project root.

## Directory Structure

```
docs/designs/
├── 00-<design-slug>/
│   ├── design.md
│   ├── plan.md              # simple plan (no phases)
│   ├── phase-00-00.md       # phase 0 of design 00
│   ├── phase-00-01.md       # phase 1 of design 00
│   └── progress.md
├── 01-<design-slug>/
│   ├── design.md
│   ├── phase-01-00.md
│   ├── phase-01-01.md
│   └── progress.md
```

## Naming Rules

### Design directories

Format: `NN-<slug>` — NN is zero-padded sequence (00, 01, 02...), slug is lowercase-kebab-case.

To get next number: list `docs/designs/`, find highest NN, increment. If none exist, start at 00.

### Files inside design directory

| File | When |
|------|------|
| `design.md` | Always. Brainstorming output / spec. |
| `plan.md` | Simple single-phase plan. |
| `phase-DD-PP.md` | Multi-phase plan. DD=design#, PP=phase# (00,01...). |
| `progress.md` | Always. Task checklist with statuses. |

### Task IDs

Format: **`TDD-PP-TT`**
- DD = design number
- PP = phase number (00 for simple plans)
- TT = task number within phase (01, 02, 03...)

Examples: `T00-00-01`, `T01-02-05`, `T03-00-12`

Use task IDs in plan/phase files as section headers AND in progress.md as checklist items.

## Workflow

### On brainstorming

1. Scan `docs/designs/` → determine next NN
2. After design validation → `mkdir -p docs/designs/NN-<slug>/`
3. Write `design.md`
4. Create `progress.md` (empty template)
5. Commit design

### On writing-plans

1. Read `design.md` from the design directory
2. Single phase → `plan.md` with task IDs `TNN-00-TT`
3. Multiple phases → one `phase-NN-PP.md` per phase with task IDs `TNN-PP-TT`
4. Populate `progress.md` with ALL task IDs + short descriptions
5. Commit plan

### On task completion

Update `progress.md`: change `[ ]` → `[x]`, optionally add brief note.

## File Templates

See [references/templates.md](references/templates.md) for exact templates of design.md, plan.md, phase-NN-PP.md, and progress.md.
