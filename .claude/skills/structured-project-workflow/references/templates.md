# File Templates

## design.md

```markdown
# <Design Title>

> Design NN — created YYYY-MM-DD

## Problem

<What problem are we solving and why>

## Goals

<Specific, measurable outcomes>

## Non-Goals

<Explicitly out of scope>

## Design

<Technical design — architecture, data flow, key decisions>

## Alternatives Considered

<Other approaches and why they were rejected>

## Open Questions

<Unresolved items, if any>
```

## plan.md (simple, single-phase)

```markdown
# <Feature Name> — Implementation Plan

> Design NN | Plan created YYYY-MM-DD
> **For agent:** Execute tasks sequentially using task IDs.

## Overview

<Brief summary of what this plan implements from the design>

## Tasks

### TNN-00-01: <Task title>

**Files:** `path/to/file.ext`

<Detailed instructions: what to do, exact code, expected behavior>

**Verify:** `<command to verify>`
**Expected:** `<expected output>`

### TNN-00-02: <Task title>

...
```

## phase-DD-PP.md (multi-phase plan)

```markdown
# Phase PP: <Phase Title>

> Design DD | Phase PP | Created YYYY-MM-DD
> **For agent:** Execute tasks sequentially using task IDs.

## Phase Goal

<What this phase achieves>

## Prerequisites

<What must be done before this phase — reference previous phase if applicable>

## Tasks

### TDD-PP-01: <Task title>

**Files:** `path/to/file.ext`

<Detailed instructions>

**Verify:** `<command>`
**Expected:** `<output>`

### TDD-PP-02: <Task title>

...

## Phase Completion Criteria

<How to know this phase is done>
```

## progress.md

```markdown
# Progress — <Design Title>

> Design NN | Last updated: YYYY-MM-DD

## Summary

| Phase | Total | Done | Status |
|-------|-------|------|--------|
| 00    | 5     | 0    | 🔲 Not started |

## Tasks

### Phase 00: <Phase title or "Main">

- [ ] `T00-00-01` — <Short task description>
- [ ] `T00-00-02` — <Short task description>
- [ ] `T00-00-03` — <Short task description>

### Phase 01: <Phase title>

- [ ] `T00-01-01` — <Short task description>
- [ ] `T00-01-02` — <Short task description>
```

When marking a task done:

```markdown
- [x] `T00-00-01` — <Short task description> ✅ done
```

Update the Summary table counts and status accordingly:
- 🔲 Not started — no tasks done
- 🔄 In progress — some tasks done
- ✅ Complete — all tasks done
