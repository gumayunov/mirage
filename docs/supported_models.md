# Supported Embedding Models

miRAGe supports the following embedding models. Each model has different characteristics suitable for different use cases.

## Model Comparison

| Model | Dimensions | Context | Languages | Speed | Quality |
|-------|------------|---------|-----------|-------|---------|
| nomic-embed-text | 768 | 8192 | EN (best), others | Fast | Good |
| bge-m3 | 1024 | 8192 | Multilingual (excellent) | Medium | Excellent |
| mxbai-embed-large | 1024 | 512 | EN (best), others | Fast | Excellent (short) |

## nomic-embed-text

**Dimensions:** 768
**Context Length:** 8192 tokens
**Ollama Model:** `nomic-embed-text`

### Strengths
- **Fast inference** — smallest model, quickest embedding generation
- **Compact storage** — 768 dimensions vs 1024, 25% less storage
- **Good English performance** — solid quality for English technical content
- **Long context** — handles full documents without truncation

### Weaknesses
- **Weaker multilingual** — not optimized for non-English content
- **Lower quality for complex semantics** — simpler model architecture

### Best For
- English-only documentation
- High-volume indexing where speed matters
- Storage-constrained environments

---

## bge-m3

**Dimensions:** 1024
**Context Length:** 8192 tokens
**Ollama Model:** `bge-m3`

### Strengths
- **Excellent multilingual** — specifically trained for multilingual retrieval (RU, EN, ZH, and 100+ languages)
- **High quality semantic search** — state-of-the-art for conceptual retrieval
- **Long context** — 8192 tokens handles most documents
- **Hybrid retrieval** — supports dense, sparse, and colbert modes

### Weaknesses
- **Larger model** — slower inference than nomic
- **More storage** — 1024 dimensions

### Best For
- Multilingual content (RU/EN mixed)
- Conceptual search (architecture patterns, best practices)
- Quality over speed

---

## mxbai-embed-large

**Dimensions:** 1024
**Context Length:** 512 tokens
**Ollama Model:** `mxbai-embed-large`

### Strengths
- **Excellent quality for short texts** — optimized for query-sized content
- **Fast inference** — efficient architecture
- **High accuracy** — top performance on MTEB benchmarks

### Weaknesses
- **Short context** — 512 tokens only, requires more chunks
- **Not optimized for long documents** — quality degrades with length

### Best For
- Short documents, summaries, abstracts
- Query-heavy workloads
- When chunk quality matters more than context preservation

---

## Recommendations by Use Case

### Technical Documentation (EN)
- **Primary:** bge-m3
- **Fallback:** nomic-embed-text

### Technical Documentation (RU/EN mixed)
- **Primary:** bge-m3
- **Add:** nomic-embed-text for faster preview search

### Books and Long-form Content
- **Primary:** bge-m3 (long context, quality)
- **Avoid:** mxbai-embed-large (short context)

### High-volume Quick Search
- **Primary:** nomic-embed-text (speed, storage)
- **Add:** bge-m3 for quality comparison

## Default Configuration

New projects use all three models by default. This enables:
- Quality comparison across models
- Fallback if one model performs poorly
- Future A/B testing capabilities

Disable models you don't need to save indexing time and storage.
