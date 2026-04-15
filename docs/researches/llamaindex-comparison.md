# Сравнение miRAGe и LlamaIndex

## Обзор

| Критерий | miRAGe | LlamaIndex |
|----------|--------|------------|
| **Тип** | Готовое приложение (API + CLI) | Фреймворк/библиотека |
| **Назначение** | RAG для книг/документации | Универсальный data framework для LLM |
| **Self-hosted** | Да, Docker/k3s | Интегрируется в ваш код |

## Форматы данных

| miRAGe | LlamaIndex |
|--------|------------|
| PDF, EPUB, Markdown | 300+ data loaders на LlamaHub: PDF, DOCX, SQL, APIs, S3, Notion, Slack, GitHub... |

## Embeddings

| miRAGe | LlamaIndex |
|--------|------------|
| Только Ollama (3 модели: nomic, bge-m3, mxbai) | OpenAI, HuggingFace, Cohere, Azure, local models, любой провайдер |

## Vector Stores

| miRAGe | LlamaIndex |
|--------|------------|
| Только pgvector | 40+ интеграций: Pinecone, Weaviate, Qdrant, Chroma, Milvus, pgvector... |

## LLM интеграция

| miRAGe | LlamaIndex |
|--------|------------|
| Нет (только embeddings) | Query engines с LLM: OpenAI, Anthropic, Llama, Mistral, локальные модели |
| Поиск возвращает только чанки | RAG pipeline: retrieval → synthesis → ответ |

## Ключевые возможности

### LlamaIndex (чего нет в miRAGe)

- Query engines (synthesis, summarization)
- Агенты и multi-step reasoning
- Graph indexes (knowledge graphs)
- Reranking modules
- Streaming responses
- Conversational memory
- Evaluation metrics

### miRAGe (чего нет "из коробки" в LlamaIndex)

- Готовый REST API
- CLI клиент
- Parent-child chunking (оптимизация контекста)
- Multi-model embeddings в параллель
- Per-project конфигурация
- Асинхронный indexer worker
- Статус индексации документов

## Вывод

**miRAGe** — специализированный self-hosted сервис для семантического поиска по книгам. Готов к деплою, минимальная настройка, но ограничен в возможностях.

**LlamaIndex** — мощный фреймворк для построения RAG-приложений любой сложности. Требует написания кода, но даёт полный контроль над pipeline.

**Когда что использовать:**
- miRAGe — нужен готовый сервис для поиска по своей библиотеке PDF/EPUB
- LlamaIndex — строите своё RAG-приложение, нужна интеграция с LLM, сложные query patterns, специфичные data sources

---

## Анализ: miRAGe как бэкенд для Opus/GLM

При архитектуре **miRAGe (retrieval) → Opus/GLM (synthesis)** многие преимущества LlamaIndex нивелируются.

### Нивелируется внешним LLM

| Возможность LlamaIndex | Почему не нужна |
|------------------------|-----------------|
| **Query engines / synthesis** | Opus/GLM сами синтезируют ответ из чанков |
| **Agents / multi-step reasoning** | Модель сама делает multi-hop, уточняет запросы |
| **Conversational memory** | Агент управляет контекстом сессии |
| **Streaming responses** | Агент стримит в UI |
| **Reranking modules** | Модель сама оценивает релевантность или простые эвристики |
| **Structured outputs** | Opus/GLM нативно поддерживают |

### Остаётся актуальным от LlamaIndex

- **Data loaders** (300+ vs 3 формата в miRAGe)
- **Гибкость vector stores** (миграция с pgvector)
- **Больше embedding провайдеров**

### Что miRAGe даёт "из коробки"

- Parent-child chunking (маленькие чанки для поиска → большой контекст для модели)
- Multi-model embeddings (сравнение качества)
- Готовый API + CLI
- Статус индексации (долгие документы)

### Итог

При связке с мощным LLM miRAGe закрывает 80% потребностей. LlamaIndex имеет смысл если:

- Нужны экзотические форматы (Notion, Slack, SQL)
- Планируется миграция между vector stores
- Хочется полностью локальный pipeline без API вызовов к внешним LLM
