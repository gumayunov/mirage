# miRAGe Implementation Plans

Полный план реализации разбит на 5 итеративных фаз для оптимального использования контекста агентов.

## Overview

```
┌─────────────────────┐
│  Phase 1: Foundation │  ← Начинать здесь
│  (6 tasks, ~5K tokens)│
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐  ┌─────────┐  ┌─────────────────┐
│ Phase 2 │  │ Phase 3 │  │    Phase 5      │
│   API   │  │ Indexer │  │ Infrastructure  │
│(4 tasks)│  │(5 tasks)│  │   (7 tasks)     │
└────┬────┘  └────┬────┘  │  (параллельно)  │
     │            │       └─────────────────┘
     └─────┬──────┘
           │
           ▼
┌─────────────────────┐
│ Phase 4: CLI + Int. │
│ (7 tasks, ~5K tokens)│
└─────────────────────┘
```

## Plans

| Phase | File | Tasks | Tokens | Dependencies |
|-------|------|-------|--------|--------------|
| 1 | [phase-1-foundation.md](phase-1-foundation.md) | 6 | ~5K | None |
| 2 | [phase-2-api.md](phase-2-api.md) | 4 | ~8K | Phase 1 |
| 3 | [phase-3-indexer.md](phase-3-indexer.md) | 5 | ~7K | Phase 1 |
| 4 | [phase-4-cli-integration.md](phase-4-cli-integration.md) | 7 | ~5K | Phase 1-3 |
| 5 | [phase-5-infrastructure.md](phase-5-infrastructure.md) | 7 | ~6K | Phase 1 (partial) |

## Execution Order

### Sequential (safest)
```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
```

### Parallel (faster)
```
Phase 1
   ├── Phase 2 (API)        ─┐
   ├── Phase 3 (Indexer)    ─┼── Phase 4 (CLI + Integration)
   └── Phase 5 (Infra)      ─┘
```

## Usage

Каждый план самодостаточен и содержит:
- Prerequisites (что должно быть готово)
- Deliverable (что будет создано)
- Tasks с TDD подходом (test first)
- Verification steps

### Запуск плана

```bash
# В новой сессии Claude Code:
# 1. Открыть план
# 2. Использовать superpowers:executing-plans

# Пример для Phase 1:
cat docs/plans/phase-1-foundation.md
```

## Architecture

Общая архитектура проекта:

```
miRAGe
├── API (FastAPI)           ← Phase 2
│   ├── /projects
│   ├── /documents
│   └── /search
├── Indexer (Worker)        ← Phase 3
│   ├── Parsers (MD, PDF, EPUB)
│   └── Chunking + Embeddings
├── CLI (Typer)             ← Phase 4
├── Shared                  ← Phase 1
│   ├── Config
│   ├── Models
│   ├── DB (SQLAlchemy + pgvector)
│   └── Embedding Client (Ollama)
└── Infrastructure          ← Phase 5
    ├── Dockerfile
    └── Helm Chart
```
