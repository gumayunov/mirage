# Phase 03-03: Search Update

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a parent-summary search path to the existing search endpoint, merge with child results by `parent_id`, and never expose summary text to the user (always return original parent content).

**Architecture:** For each model, run two SQL queries — child-original (current) and parent-summary (new) — both returning `parent_id`, `distance`, and original `parent.content`. Python-side merge dedups by `parent_id`, retains minimum distance, and constructs `ChunkResult` with `content = parent.content` when the hit came via the summary path.

---

### Task T03-03-01: Parent summary SQL query

**Files:**
- Modify: `src/mirage/api/routers/search.py:64-114`

- [ ] **Step 1: Restrict child query to children**

In the existing child SQL block, add `AND child.parent_id IS NOT NULL` to the WHERE clause (currently the JOIN already excludes parents, but be explicit):

```sql
WHERE d.project_id = :project_id
  AND child.parent_id IS NOT NULL
  AND d.status IN ('ready', 'partial')
```

- [ ] **Step 2: Add parent-summary query inside the per-model loop**

Below the child query, add:

```python
            try:
                parent_sql = text(f"""
                    SELECT parent.id AS parent_id,
                           parent.content AS parent_content,
                           parent.structure AS parent_structure,
                           e.embedding <=> :embedding AS distance,
                           d.id AS doc_id, d.filename
                    FROM {table_name} e
                    JOIN chunks parent ON e.chunk_id = parent.id
                    JOIN documents d ON parent.document_id = d.id
                    WHERE d.project_id = :project_id
                      AND parent.parent_id IS NULL
                      AND d.status IN ('ready', 'partial')
                    ORDER BY e.embedding <=> :embedding
                    LIMIT :limit
                """)
                presult = await db.execute(
                    parent_sql,
                    {"embedding": str(query_embedding), "project_id": project_id, "limit": request.limit},
                )
                for row in presult.fetchall():
                    all_results.append((
                        row.parent_id,            # use parent.id as chunk_id for parent hits
                        row.parent_content,       # content
                        row.parent_content,       # parent_content
                        row.distance,
                        row.doc_id,
                        row.filename,
                        row.parent_structure,
                    ))
            except Exception as e:
                logger.warning(f"Parent-summary search failed for {model_name}: {e}")
```

- [ ] **Step 3: Commit (no behavior change beyond adding queries)**

```
git add src/mirage/api/routers/search.py
git commit -m "feat(api): parent-summary SQL path in search (T03-03-01)"
```

---

### Task T03-03-02: Merge by parent_id

**Files:**
- Modify: `src/mirage/api/routers/search.py:116-142`

- [ ] **Step 1: Replace dedup-by-chunk_id with dedup-by-parent_id**

The existing block dedups by `chunk_id`. Switch to keying by `parent_id` so a parent hit and its children collapse to one result with the lowest distance.

Add a side query before iterating: build `child_id → parent_id` map for chunks present in `all_results`:

```python
    child_ids = [r[0] for r in all_results if r[0] != r[2]]  # heuristic: child rows have content != parent_content potentially
    # More reliable: query DB once
    parent_lookup: dict[str, str] = {}
    if all_results:
        cids = list({r[0] for r in all_results})
        rows = (await db.execute(
            select(ChunkTable.id, ChunkTable.parent_id).where(ChunkTable.id.in_(cids))
        )).fetchall()
        for cid, pid in rows:
            parent_lookup[cid] = pid or cid  # parents map to themselves
```

Add `from mirage.shared.db import ChunkTable` to imports.

Then dedup:

```python
    seen: dict[str, tuple] = {}  # parent_id → (chunk_id, content, parent_content, distance, doc_id, filename, structure)
    for chunk_id, content, parent_content, distance, doc_id, filename, structure in all_results:
        pid = parent_lookup.get(chunk_id, chunk_id)
        cur = seen.get(pid)
        if cur is None or cur[3] > distance:
            seen[pid] = (chunk_id, content, parent_content, distance, doc_id, filename, structure)

    sorted_results = sorted(seen.values(), key=lambda x: x[3])[:request.limit]

    results = []
    for chunk_id, content, parent_content, distance, doc_id, filename, structure in sorted_results:
        score = 1 - distance
        if score >= request.threshold:
            results.append(ChunkResult(
                chunk_id=chunk_id,
                content=parent_content if chunk_id == parent_lookup.get(chunk_id) and content == parent_content else content,
                parent_content=parent_content,
                score=score,
                structure=structure,
                document={"id": doc_id, "filename": filename},
            ))
```

(The conditional collapses to: when the matched row was a parent hit, `content` is the parent original — already set in T03-03-01. The conditional is defensive.)

- [ ] **Step 2: Run existing search tests**

Run: `uv run pytest tests/api/test_search.py -v`
Expected: all pre-existing assertions still hold (child-only data: parent hits absent → behavior unchanged).

- [ ] **Step 3: Commit**

```
git add src/mirage/api/routers/search.py
git commit -m "feat(api): merge search results by parent_id (T03-03-02)"
```

---

### Task T03-03-03: Integration test — parent summary hit

**Files:**
- Modify: `tests/api/test_search.py`

- [ ] **Step 1: Add a test that seeds an embedding only on a parent (via summary path)**

```python
@pytest.mark.asyncio
async def test_search_returns_parent_when_summary_embedding_matches(client, seed_with_parent_summary):
    """Parent has summary embedding, no child embedding → parent hit returns parent.content."""
    r = client.post(
        f"/projects/{seed_with_parent_summary['project_id']}/search",
        headers={"X-API-Key": "test-key"},
        json={"query": "о чём глава", "limit": 5, "threshold": 0.0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"], "expected parent-summary hit"
    chunk_ids = [r["chunk_id"] for r in body["results"]]
    assert seed_with_parent_summary["parent_id"] in chunk_ids
    hit = next(r for r in body["results"] if r["chunk_id"] == seed_with_parent_summary["parent_id"])
    assert hit["content"] == seed_with_parent_summary["parent_content"]  # original, not summary
```

The fixture `seed_with_parent_summary` should: create project + doc + parent chunk with `summary_text`+`summary_status='done'`, insert one row in the model embeddings table for that parent (no children), and stub the query embedding to match.

- [ ] **Step 2: Build the fixture (mirror existing search test setup, adding the embeddings-table insert)**

Use a deterministic embedding (e.g., `[1.0, 0, …]`) and have the search request mock the embedding client to return the same vector so distance is 0.

- [ ] **Step 3: Run + commit**

```
uv run pytest tests/api/test_search.py -v
git add tests/api/test_search.py
git commit -m "test(api): parent-summary search integration (T03-03-03)"
```
