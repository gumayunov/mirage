# Alembic Migrations

> Design 00 — created 2026-01-31

## Problem

On every API startup, the `lifespan` hook calls `recreate_tables()` which runs `DROP ALL` + `CREATE ALL`. All data is lost. There is no mechanism to evolve the DB schema without losing data.

## Goals

- Data in PostgreSQL persists across restarts
- DB schema managed through versioned migrations (Alembic)
- Fresh DB can be set up from scratch with `make migrate`
- Ready for future k3s deployment (migration as a separate step, not in app startup)

## Non-Goals

- Changes to the current schema
- Automatic migration on app startup

## Design

### Alembic setup

Add `alembic` to dependencies. Initialize with async template. Config (`alembic.ini`) reads `database_url` from `MIRAGE_DATABASE_URL` environment variable, same as the application.

### File structure

```
alembic.ini              # config in project root
src/mirage/migrations/
├── env.py               # async engine, imports Base.metadata
├── script.py.mako       # migration template
└── versions/
    └── 001_baseline.py  # creates all tables (projects, documents, chunks, indexing_tasks)
```

### Baseline migration

`upgrade()` contains `CREATE TABLE` for all four tables + `CREATE EXTENSION IF NOT EXISTS vector`. `downgrade()` — `DROP TABLE` in reverse order. For an existing DB with data — `alembic stamp head`.

### Lifespan changes

Remove `recreate_tables(engine)` call from `api/main.py`. Lifespan no longer touches the schema. Remove `recreate_tables` and `create_tables` from `db.py` (keep `create_tables` only if used by test fixtures).

### Makefile

New target `make migrate` — runs `docker exec mirage-api alembic upgrade head` inside the API container.

### Tests

Current tests use SQLite in-memory and call `create_tables()` in fixtures. This stays unchanged — tests do not depend on Alembic. Fixtures continue using `create_tables()` directly for SQLite.

## Alternatives Considered

- **Just `create_tables` instead of `recreate_tables`** — does not support altering existing tables, dead end at the first ALTER TABLE.
- **Manual SQL migrations** — more manual work, no autogenerate, no SQLAlchemy integration.
- **Auto-migrate in lifespan** — race conditions with multiple replicas in k3s.

## Open Questions

None.
