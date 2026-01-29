# miRAGe Phase 5: Infrastructure

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Создать Dockerfile и Helm chart для деплоя в k3s.

**Prerequisite:** Phase 1 (Foundation) завершена. Можно выполнять параллельно с Phase 2-4.

**Deliverable:** Docker образ и Helm chart готовы к деплою. `helm install` работает.

---

## Task 7.1: Dockerfile

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

**Step 1: Create Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv sync --no-dev --no-install-project

# Copy source code
COPY src/ ./src/

# Install the project
RUN uv sync --no-dev

# Set Python path
ENV PYTHONPATH=/app/src

# Default command (overridden by Helm)
CMD ["uvicorn", "mirage.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Create .dockerignore**

```
__pycache__/
*.py[cod]
.venv/
.env
*.egg-info/
dist/
.coverage
htmlcov/
.mypy_cache/
.pytest_cache/
.ruff_cache/
.git/
tests/
docs/
helm/
```

**Step 3: Commit**

```bash
git add .
git commit -m "feat: add Dockerfile"
```

---

## Task 7.2: Indexer Entrypoint

**Files:**
- Create: `src/mirage/indexer/__main__.py`

**Step 1: Create __main__.py**

```python
import asyncio
import logging

from mirage.shared.config import Settings
from mirage.shared.db import create_tables, get_engine
from mirage.indexer.worker import IndexerWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def main():
    settings = Settings()

    # Create tables if needed
    engine = get_engine(settings.database_url)
    await create_tables(engine)
    await engine.dispose()

    # Run worker
    worker = IndexerWorker(settings)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Commit**

```bash
git add .
git commit -m "feat: add indexer entrypoint"
```

---

## Task 7.3: Database Migrations (Alembic)

**Files:**
- Modify: `pyproject.toml` (add alembic dependency)
- Create: `alembic.ini`
- Create: `src/mirage/migrations/env.py`
- Create: `src/mirage/migrations/versions/001_initial.py`
- Modify: `src/mirage/indexer/__main__.py` (run migrations on start)

**Step 1: Add alembic dependency**

```bash
uv add alembic
```

**Step 2: Create alembic.ini**

`alembic.ini`:
```ini
[alembic]
script_location = src/mirage/migrations
prepend_sys_path = .

[post_write_hooks]

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

**Step 3: Create migrations/env.py**

`src/mirage/migrations/env.py`:
```python
import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from mirage.shared.config import Settings
from mirage.shared.db import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url():
    settings = Settings()
    return settings.database_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

**Step 4: Create script.py.mako**

`src/mirage/migrations/script.py.mako`:
```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

**Step 5: Create initial migration**

`src/mirage/migrations/versions/001_initial.py`:
```python
"""Initial migration

Revision ID: 001
Revises:
Create Date: 2026-01-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'projects',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), unique=True, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False),
    )

    op.create_table(
        'documents',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('project_id', sa.String(36), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('filename', sa.String(255), nullable=False),
        sa.Column('original_path', sa.String(512), nullable=False),
        sa.Column('file_type', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='pending'),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('indexed_at', sa.DateTime, nullable=True),
    )

    op.create_table(
        'chunks',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('document_id', sa.String(36), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('embedding', sa.JSON, nullable=True),
        sa.Column('position', sa.Integer, nullable=False),
        sa.Column('structure', sa.JSON, nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
    )

    op.create_table(
        'indexing_tasks',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('document_id', sa.String(36), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('task_type', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='pending'),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),
    )


def downgrade() -> None:
    op.drop_table('indexing_tasks')
    op.drop_table('chunks')
    op.drop_table('documents')
    op.drop_table('projects')
```

**Step 6: Update indexer entrypoint to run migrations**

Update `src/mirage/indexer/__main__.py`:
```python
import asyncio
import logging
import subprocess

from mirage.shared.config import Settings
from mirage.indexer.worker import IndexerWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_migrations():
    """Run alembic migrations."""
    logger.info("Running database migrations...")
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Migration failed: {result.stderr}")
        raise RuntimeError("Database migration failed")
    logger.info("Migrations completed successfully")


async def main():
    settings = Settings()

    # Run migrations
    run_migrations()

    # Run worker
    worker = IndexerWorker(settings)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 7: Commit**

```bash
git add .
git commit -m "feat: add Alembic database migrations"
```

---

## Task 6.1: Helm Chart Base

**Files:**
- Create: `helm/mirage/Chart.yaml`
- Create: `helm/mirage/values.yaml`

**Step 1: Create Chart.yaml**

`helm/mirage/Chart.yaml`:
```yaml
apiVersion: v2
name: mirage
description: miRAGe - Local RAG system for books and documentation
type: application
version: 0.1.0
appVersion: "0.1.0"
```

**Step 2: Create values.yaml**

`helm/mirage/values.yaml`:
```yaml
# API Configuration
api:
  replicas: 1
  image:
    repository: mirage
    tag: latest
    pullPolicy: IfNotPresent
  resources:
    requests:
      memory: "256Mi"
      cpu: "100m"
    limits:
      memory: "512Mi"
      cpu: "500m"

# Indexer Configuration
indexer:
  replicas: 1
  image:
    repository: mirage
    tag: latest
    pullPolicy: IfNotPresent
  resources:
    requests:
      memory: "512Mi"
      cpu: "200m"
    limits:
      memory: "1Gi"
      cpu: "1000m"

# Ollama Configuration
ollama:
  enabled: true
  image:
    repository: ollama/ollama
    tag: latest
  model: mxbai-embed-large
  persistence:
    enabled: true
    size: 10Gi
    storageClass: ""
  resources:
    requests:
      memory: "2Gi"
      cpu: "500m"
    limits:
      memory: "4Gi"
      cpu: "2000m"

# PostgreSQL Configuration
postgresql:
  enabled: true
  image:
    repository: pgvector/pgvector
    tag: pg16
  auth:
    database: mirage
    username: mirage
    password: ""  # Set via secret
  persistence:
    enabled: true
    size: 5Gi
    storageClass: ""
  resources:
    requests:
      memory: "256Mi"
      cpu: "100m"
    limits:
      memory: "512Mi"
      cpu: "500m"

# Documents Storage
documents:
  persistence:
    enabled: true
    size: 10Gi
    storageClass: ""

# Ingress Configuration
ingress:
  enabled: false
  className: ""
  host: mirage.local
  tls: []

# Authentication
auth:
  apiKey: ""  # Set via secret

# General Configuration
config:
  chunkSize: 800
  chunkOverlap: 100
```

**Step 3: Commit**

```bash
git add .
git commit -m "feat: add Helm chart base"
```

---

## Task 6.2: PostgreSQL Templates

**Files:**
- Create: `helm/mirage/templates/postgresql-deployment.yaml`
- Create: `helm/mirage/templates/postgresql-pvc.yaml`
- Create: `helm/mirage/templates/postgresql-service.yaml`

**Step 1: Create postgresql-pvc.yaml**

`helm/mirage/templates/postgresql-pvc.yaml`:
```yaml
{{- if and .Values.postgresql.enabled .Values.postgresql.persistence.enabled }}
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ .Release.Name }}-postgresql
  labels:
    app.kubernetes.io/name: postgresql
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  accessModes:
    - ReadWriteOnce
  {{- if .Values.postgresql.persistence.storageClass }}
  storageClassName: {{ .Values.postgresql.persistence.storageClass }}
  {{- end }}
  resources:
    requests:
      storage: {{ .Values.postgresql.persistence.size }}
{{- end }}
```

**Step 2: Create postgresql-deployment.yaml**

`helm/mirage/templates/postgresql-deployment.yaml`:
```yaml
{{- if .Values.postgresql.enabled }}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-postgresql
  labels:
    app.kubernetes.io/name: postgresql
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: postgresql
      app.kubernetes.io/instance: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: postgresql
        app.kubernetes.io/instance: {{ .Release.Name }}
    spec:
      containers:
        - name: postgresql
          image: "{{ .Values.postgresql.image.repository }}:{{ .Values.postgresql.image.tag }}"
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_DB
              value: {{ .Values.postgresql.auth.database }}
            - name: POSTGRES_USER
              value: {{ .Values.postgresql.auth.username }}
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: {{ .Release.Name }}-secrets
                  key: postgresql-password
          volumeMounts:
            - name: data
              mountPath: /var/lib/postgresql/data
          resources:
            {{- toYaml .Values.postgresql.resources | nindent 12 }}
      volumes:
        - name: data
          {{- if .Values.postgresql.persistence.enabled }}
          persistentVolumeClaim:
            claimName: {{ .Release.Name }}-postgresql
          {{- else }}
          emptyDir: {}
          {{- end }}
{{- end }}
```

**Step 3: Create postgresql-service.yaml**

`helm/mirage/templates/postgresql-service.yaml`:
```yaml
{{- if .Values.postgresql.enabled }}
apiVersion: v1
kind: Service
metadata:
  name: {{ .Release.Name }}-postgresql
  labels:
    app.kubernetes.io/name: postgresql
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  type: ClusterIP
  ports:
    - port: 5432
      targetPort: 5432
  selector:
    app.kubernetes.io/name: postgresql
    app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
```

**Step 4: Commit**

```bash
git add .
git commit -m "feat: add PostgreSQL Helm templates"
```

---

## Task 6.3: Ollama Templates

**Files:**
- Create: `helm/mirage/templates/ollama-deployment.yaml`
- Create: `helm/mirage/templates/ollama-pvc.yaml`
- Create: `helm/mirage/templates/ollama-service.yaml`

**Step 1: Create ollama-pvc.yaml**

`helm/mirage/templates/ollama-pvc.yaml`:
```yaml
{{- if and .Values.ollama.enabled .Values.ollama.persistence.enabled }}
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ .Release.Name }}-ollama
  labels:
    app.kubernetes.io/name: ollama
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  accessModes:
    - ReadWriteOnce
  {{- if .Values.ollama.persistence.storageClass }}
  storageClassName: {{ .Values.ollama.persistence.storageClass }}
  {{- end }}
  resources:
    requests:
      storage: {{ .Values.ollama.persistence.size }}
{{- end }}
```

**Step 2: Create ollama-deployment.yaml**

`helm/mirage/templates/ollama-deployment.yaml`:
```yaml
{{- if .Values.ollama.enabled }}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-ollama
  labels:
    app.kubernetes.io/name: ollama
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: ollama
      app.kubernetes.io/instance: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: ollama
        app.kubernetes.io/instance: {{ .Release.Name }}
    spec:
      containers:
        - name: ollama
          image: "{{ .Values.ollama.image.repository }}:{{ .Values.ollama.image.tag }}"
          ports:
            - containerPort: 11434
          volumeMounts:
            - name: models
              mountPath: /root/.ollama
          resources:
            {{- toYaml .Values.ollama.resources | nindent 12 }}
      volumes:
        - name: models
          {{- if .Values.ollama.persistence.enabled }}
          persistentVolumeClaim:
            claimName: {{ .Release.Name }}-ollama
          {{- else }}
          emptyDir: {}
          {{- end }}
{{- end }}
```

**Step 3: Create ollama-service.yaml**

`helm/mirage/templates/ollama-service.yaml`:
```yaml
{{- if .Values.ollama.enabled }}
apiVersion: v1
kind: Service
metadata:
  name: {{ .Release.Name }}-ollama
  labels:
    app.kubernetes.io/name: ollama
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  type: ClusterIP
  ports:
    - port: 11434
      targetPort: 11434
  selector:
    app.kubernetes.io/name: ollama
    app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
```

**Step 4: Commit**

```bash
git add .
git commit -m "feat: add Ollama Helm templates"
```

---

## Task 6.4: API and Indexer Templates

**Files:**
- Create: `helm/mirage/templates/configmap.yaml`
- Create: `helm/mirage/templates/secret.yaml`
- Create: `helm/mirage/templates/documents-pvc.yaml`
- Create: `helm/mirage/templates/api-deployment.yaml`
- Create: `helm/mirage/templates/api-service.yaml`
- Create: `helm/mirage/templates/indexer-deployment.yaml`

**Step 1: Create configmap.yaml**

`helm/mirage/templates/configmap.yaml`:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .Release.Name }}-config
  labels:
    app.kubernetes.io/instance: {{ .Release.Name }}
data:
  MIRAGE_OLLAMA_URL: "http://{{ .Release.Name }}-ollama:11434"
  MIRAGE_OLLAMA_MODEL: {{ .Values.ollama.model | quote }}
  MIRAGE_CHUNK_SIZE: {{ .Values.config.chunkSize | quote }}
  MIRAGE_CHUNK_OVERLAP: {{ .Values.config.chunkOverlap | quote }}
  MIRAGE_DOCUMENTS_PATH: "/data/documents"
```

**Step 2: Create secret.yaml**

`helm/mirage/templates/secret.yaml`:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: {{ .Release.Name }}-secrets
  labels:
    app.kubernetes.io/instance: {{ .Release.Name }}
type: Opaque
stringData:
  postgresql-password: {{ .Values.postgresql.auth.password | default (randAlphaNum 16) | quote }}
  api-key: {{ .Values.auth.apiKey | default (randAlphaNum 32) | quote }}
  database-url: "postgresql+asyncpg://{{ .Values.postgresql.auth.username }}:{{ .Values.postgresql.auth.password }}@{{ .Release.Name }}-postgresql:5432/{{ .Values.postgresql.auth.database }}"
```

**Step 3: Create documents-pvc.yaml**

`helm/mirage/templates/documents-pvc.yaml`:
```yaml
{{- if .Values.documents.persistence.enabled }}
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ .Release.Name }}-documents
  labels:
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  accessModes:
    - ReadWriteMany
  {{- if .Values.documents.persistence.storageClass }}
  storageClassName: {{ .Values.documents.persistence.storageClass }}
  {{- end }}
  resources:
    requests:
      storage: {{ .Values.documents.persistence.size }}
{{- end }}
```

**Step 4: Create api-deployment.yaml**

`helm/mirage/templates/api-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-api
  labels:
    app.kubernetes.io/name: api
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  replicas: {{ .Values.api.replicas }}
  selector:
    matchLabels:
      app.kubernetes.io/name: api
      app.kubernetes.io/instance: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: api
        app.kubernetes.io/instance: {{ .Release.Name }}
    spec:
      containers:
        - name: api
          image: "{{ .Values.api.image.repository }}:{{ .Values.api.image.tag }}"
          imagePullPolicy: {{ .Values.api.image.pullPolicy }}
          command: ["uvicorn", "mirage.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: {{ .Release.Name }}-config
          env:
            - name: MIRAGE_DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: {{ .Release.Name }}-secrets
                  key: database-url
            - name: MIRAGE_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ .Release.Name }}-secrets
                  key: api-key
          volumeMounts:
            - name: documents
              mountPath: /data/documents
          resources:
            {{- toYaml .Values.api.resources | nindent 12 }}
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
      volumes:
        - name: documents
          {{- if .Values.documents.persistence.enabled }}
          persistentVolumeClaim:
            claimName: {{ .Release.Name }}-documents
          {{- else }}
          emptyDir: {}
          {{- end }}
```

**Step 5: Create api-service.yaml**

`helm/mirage/templates/api-service.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ .Release.Name }}-api
  labels:
    app.kubernetes.io/name: api
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  type: ClusterIP
  ports:
    - port: 8000
      targetPort: 8000
  selector:
    app.kubernetes.io/name: api
    app.kubernetes.io/instance: {{ .Release.Name }}
```

**Step 6: Create indexer-deployment.yaml**

`helm/mirage/templates/indexer-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-indexer
  labels:
    app.kubernetes.io/name: indexer
    app.kubernetes.io/instance: {{ .Release.Name }}
spec:
  replicas: {{ .Values.indexer.replicas }}
  selector:
    matchLabels:
      app.kubernetes.io/name: indexer
      app.kubernetes.io/instance: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: indexer
        app.kubernetes.io/instance: {{ .Release.Name }}
    spec:
      containers:
        - name: indexer
          image: "{{ .Values.indexer.image.repository }}:{{ .Values.indexer.image.tag }}"
          imagePullPolicy: {{ .Values.indexer.image.pullPolicy }}
          command: ["python", "-m", "mirage.indexer"]
          envFrom:
            - configMapRef:
                name: {{ .Release.Name }}-config
          env:
            - name: MIRAGE_DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: {{ .Release.Name }}-secrets
                  key: database-url
            - name: MIRAGE_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ .Release.Name }}-secrets
                  key: api-key
          volumeMounts:
            - name: documents
              mountPath: /data/documents
          resources:
            {{- toYaml .Values.indexer.resources | nindent 12 }}
      volumes:
        - name: documents
          {{- if .Values.documents.persistence.enabled }}
          persistentVolumeClaim:
            claimName: {{ .Release.Name }}-documents
          {{- else }}
          emptyDir: {}
          {{- end }}
```

**Step 7: Commit**

```bash
git add .
git commit -m "feat: add API and Indexer Helm templates"
```

---

## Task 6.5: Ingress Template

**Files:**
- Create: `helm/mirage/templates/ingress.yaml`

**Step 1: Create ingress.yaml**

`helm/mirage/templates/ingress.yaml`:
```yaml
{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ .Release.Name }}-ingress
  labels:
    app.kubernetes.io/instance: {{ .Release.Name }}
  {{- if .Values.ingress.annotations }}
  annotations:
    {{- toYaml .Values.ingress.annotations | nindent 4 }}
  {{- end }}
spec:
  {{- if .Values.ingress.className }}
  ingressClassName: {{ .Values.ingress.className }}
  {{- end }}
  {{- if .Values.ingress.tls }}
  tls:
    {{- range .Values.ingress.tls }}
    - hosts:
        {{- range .hosts }}
        - {{ . | quote }}
        {{- end }}
      secretName: {{ .secretName }}
    {{- end }}
  {{- end }}
  rules:
    - host: {{ .Values.ingress.host | quote }}
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: {{ .Release.Name }}-api
                port:
                  number: 8000
{{- end }}
```

**Step 2: Commit**

```bash
git add .
git commit -m "feat: add Ingress Helm template"
```

---

## Verification

После завершения всех задач:

**1. Проверить синтаксис Helm:**
```bash
helm lint helm/mirage
```

**2. Проверить сборку Docker образа:**
```bash
docker build -t mirage:latest .
```

**3. Dry-run деплоя:**
```bash
helm install mirage ./helm/mirage --dry-run --debug \
  --set postgresql.auth.password=testpass \
  --set auth.apiKey=testapikey
```

Все команды должны завершиться успешно. Infrastructure готова к деплою.
