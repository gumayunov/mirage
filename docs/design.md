# miRAGe — Design Document

Дата: 2026-01-28

## Обзор

miRAGe — локальная RAG-система для работы с книгами и документацией в контексте проектов. Интегрируется с Claude Code через skill и CLI.

**Ключевые особенности:**
- Изоляция документов по проектам
- Multi-hop retrieval на стороне агента
- Семантические чанки с сохранением структуры документа
- Деплой в k3s кластер

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                   k3s cluster (namespace: mirage)           │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   miRAGe     │    │   miRAGe     │    │   Ollama     │  │
│  │     API      │───▶│   Indexer    │───▶│  (embeddings)│  │
│  └──────┬───────┘    └──────┬───────┘    └──────────────┘  │
│         │                   │                               │
│         ▼                   ▼                               │
│  ┌─────────────────────────────────────┐                   │
│  │      PostgreSQL + pgvector          │                   │
│  │  ┌─────────┐ ┌─────────┐ ┌───────┐  │    ┌──────────┐  │
│  │  │ chunks  │ │  tasks  │ │ docs  │  │    │   PV     │  │
│  │  │+vectors │ │ (queue) │ │ meta  │  │    │ (files)  │  │
│  │  └─────────┘ └─────────┘ └───────┘  │    └──────────┘  │
│  └─────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
         ▲
         │ API Key auth
         │
┌────────┴────────┐
│  Claude Code    │
│  + miRAGe skill │
│  + CLI          │
└─────────────────┘
```

**Компоненты:**
- **API** — FastAPI, CRUD документов, поиск, аутентификация
- **Indexer** — worker для парсинга и индексации документов
- **PostgreSQL + pgvector** — хранение метаданных и векторов
- **Ollama** — генерация embeddings (модель mxbai-embed-large)
- **PV** — хранение оригиналов файлов

**Потоки данных:**
- **Добавление документа:** CLI → API → файл в PV → задача в БД → Indexer парсит → embeddings через Ollama → чанки в pgvector
- **Поиск:** CLI → API → embedding запроса через Ollama → vector search → чанки с метаданными
- **Multi-hop:** Skill делает несколько поисков, анализирует, уточняет запрос

## Модель данных

```sql
-- Проекты
projects (
    id UUID PRIMARY KEY,
    name VARCHAR(255) UNIQUE,      -- "miRAGe", "k3s-monitoring"
    created_at TIMESTAMP
)

-- Документы
documents (
    id UUID PRIMARY KEY,
    project_id UUID REFERENCES projects,
    filename VARCHAR(255),          -- "clean_code.pdf"
    original_path VARCHAR(512),     -- путь в PV
    file_type VARCHAR(50),          -- "pdf", "epub", "markdown"
    status VARCHAR(50),             -- "pending", "indexing", "ready", "error"
    error_message TEXT,
    metadata JSONB,                 -- название книги, автор, и т.д.
    created_at TIMESTAMP,
    indexed_at TIMESTAMP
)

-- Чанки с векторами
chunks (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents,
    content TEXT,                   -- текст чанка
    embedding vector(2048),         -- максимальный размер, оптимизируем позже
    position INTEGER,               -- порядок в документе
    structure JSONB,                -- {"book": "...", "chapter": "...", "section": "..."}
    metadata JSONB                  -- дополнительные данные
)

-- Очередь задач индексации
indexing_tasks (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents,
    task_type VARCHAR(50),          -- "index", "reindex", "delete"
    status VARCHAR(50),             -- "pending", "processing", "done", "failed"
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
)
```

**Индексы:**
- `chunks.embedding` — ivfflat или hnsw для vector search
- `chunks.document_id` — для получения чанков документа
- `documents.project_id` — фильтрация по проекту

## API

**Base URL:** `https://mirage.your-domain.ru/api/v1`

**Аутентификация:** заголовок `X-API-Key: <token>`

```
# Проекты
GET    /projects                    — список проектов
POST   /projects                    — создать проект {name}
DELETE /projects/{project_id}       — удалить проект и все документы

# Документы
GET    /projects/{project_id}/documents           — список документов
POST   /projects/{project_id}/documents           — загрузить документ (multipart/form-data)
GET    /projects/{project_id}/documents/{doc_id}  — статус и метаданные документа
DELETE /projects/{project_id}/documents/{doc_id}  — удалить документ
POST   /projects/{project_id}/documents/{doc_id}/reindex  — переиндексировать

# Чанки (для отладки)
GET    /projects/{project_id}/documents/{doc_id}/chunks  — список чанков документа

# Поиск
POST   /projects/{project_id}/search
       {query: "...", limit: 10, threshold: 0.7}
       → [{chunk_id, content, score, structure, document: {id, filename}}]
```

**Коды ответов:**
- `202 Accepted` — для загрузки документа (индексация асинхронная)
- `409 Conflict` — документ с таким именем уже есть
- `404` — проект/документ не найден

## Парсинг и Chunking

**Парсеры по типам файлов:**

| Формат | Библиотека | Извлечение структуры |
|--------|-----------|---------------------|
| PDF | PyMuPDF (fitz) | TOC если есть, иначе эвристики по шрифтам |
| EPUB | ebooklib | TOC + HTML-структура глав |
| Markdown | markdown-it-py | Заголовки `#`, `##`, `###` |

**Алгоритм chunking:**

1. **Извлечь структуру** — построить дерево (книга → главы → секции)
2. **Разбить на параграфы** — по двойным переносам строк
3. **Семантическая группировка** — объединять короткие параграфы, разбивать длинные
4. **Целевой размер чанка** — 500-1000 токенов (конфигурируемо)
5. **Overlap** — 50-100 токенов между соседними чанками

**Метаданные чанка (structure JSONB):**

```json
{
  "book": "Clean Code",
  "chapter": "Chapter 3: Functions",
  "section": "Do One Thing",
  "page": 35,
  "position_in_chapter": 3
}
```

**Обработка проблемных случаев:**
- PDF без TOC — разбиваем по страницам, определяем заголовки по размеру шрифта
- Сканированный PDF — не поддерживаем OCR, возвращаем ошибку
- Слабо структурированные книги — разбиваем по абзацам, structure минимальный

## CLI

```bash
# Список документов
mirage documents list --project <project_id>

# Добавить документ
mirage documents add --project <project_id> /path/to/file.pdf

# Удалить документ
mirage documents remove --project <project_id> <document_id>

# Статус документа
mirage documents status --project <project_id> <document_id>

# Поиск
mirage search --project <project_id> "запрос" --limit 10
```

**Конфигурация CLI:**
- `MIRAGE_API_KEY` — переменная окружения
- `MIRAGE_API_URL` — переменная окружения (default: `https://mirage.your-domain.ru/api/v1`)

## Skill для Claude Code

**Конфиг проекта (`.mirage.yaml`):**

```yaml
project_id: "my-project"
```

**Skill (`~/.claude/skills/mirage.md`):**

```markdown
---
name: mirage
description: Поиск по базе знаний проекта (книги, документация)
---

# miRAGe — база знаний проекта

Используй этот skill когда нужно найти информацию в книгах
и документации, привязанных к текущему проекту.

## Конфигурация

Читай project_id из файла `.mirage.yaml` в корне проекта.
API ключ в переменной окружения `MIRAGE_API_KEY`.

## Команды

mirage documents list --project <project_id>
mirage documents add --project <project_id> /path/to/file.pdf
mirage documents remove --project <project_id> <document_id>
mirage documents status --project <project_id> <document_id>
mirage search --project <project_id> "запрос" --limit 10

## Multi-hop поиск

При сложных вопросах:
1. Разбей на 2-4 подвопроса
2. Сделай поиск по каждому
3. Если информации недостаточно — уточни запрос
4. Синтезируй ответ с указанием источников
```

## Структура проекта

```
miRAGe/
├── mise.toml                # версии инструментов
├── pyproject.toml
├── Dockerfile
├── README.md
├── src/
│   ├── api/                 # FastAPI приложение
│   │   ├── main.py
│   │   ├── routers/
│   │   ├── models/
│   │   └── dependencies.py
│   ├── indexer/             # Worker индексации
│   │   ├── worker.py
│   │   ├── parsers/
│   │   └── chunking.py
│   ├── cli/                 # CLI клиент
│   │   └── main.py
│   └── shared/              # Общий код
│       ├── db.py
│       ├── embedding.py
│       └── config.py
└── helm/
    └── mirage/
        ├── Chart.yaml
        ├── values.yaml
        └── templates/
            ├── api-deployment.yaml
            ├── indexer-deployment.yaml
            ├── ollama-deployment.yaml
            ├── ollama-pvc.yaml
            ├── postgresql-deployment.yaml
            ├── postgresql-pvc.yaml
            ├── documents-pvc.yaml
            ├── configmap.yaml
            ├── secret.yaml
            ├── service.yaml
            └── ingress.yaml
```

**mise.toml:**

```toml
[tools]
python = "3.12"
uv = "latest"

[env]
VIRTUAL_ENV = ".venv"
UV_LINK_MODE = "copy"
```

## Деплой

**Всё в одном namespace `mirage`:**
- PostgreSQL + pgvector
- Ollama + PVC для моделей
- API
- Indexer
- PVC для документов

**Загрузка модели Ollama (вручную после деплоя):**

```bash
kubectl exec -n mirage deploy/ollama -- ollama pull mxbai-embed-large
```

**Helm values (основное):**

```yaml
api:
  replicas: 1

indexer:
  replicas: 1

ollama:
  model: "mxbai-embed-large"
  persistence:
    size: 10Gi

postgresql:
  persistence:
    size: 5Gi

documents:
  persistence:
    size: 10Gi

ingress:
  enabled: true
  host: mirage.your-domain.ru

auth:
  apiKey: ""  # через secret
```

## Порядок реализации

1. **Ollama** — деплой, PVC, загрузка модели
2. **PostgreSQL + pgvector** — деплой, схема БД
3. **API** — FastAPI, CRUD, аутентификация
4. **Indexer** — парсеры, chunking, embeddings
5. **Поиск** — vector search endpoint
6. **CLI** — клиент
7. **Helm** — чарт для всех компонентов
8. **Skill** — интеграция с Claude Code

## Ключевые решения

| Решение | Выбор | Причина |
|---------|-------|---------|
| Идентификация проекта | `.mirage.yaml` + project_id в API | Явность, дополнительные настройки |
| Multi-hop | На стороне агента | Гибче, проще сервис, токены не жалко |
| Chunking | Семантический + структурные метаданные | Качество поиска + контекст |
| Очередь задач | PostgreSQL | Меньше компонентов |
| Архитектура | API + Indexer worker | Не блокировать API при индексации |
| Интеграция | CLI (не MCP) | Проще, легче отлаживать |
| Стек | Python + FastAPI + uv | Экосистема для RAG, скорость |
| Хранение файлов | PersistentVolume | Простота |
| Аутентификация | Один API ключ | Достаточно для одного пользователя |
| Namespace | Один для всего | Ollama только для miRAGe |
