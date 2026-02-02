# k3s Deployment

> Design 01 — created 2026-02-02

## Problem

miRAGe currently runs только локально через docker-compose. Для production использования нужен деплой в k3s кластер с:
- Persistent storage для данных
- TLS/HTTPS доступ к API
- Возможность масштабирования
- Управление конфигурацией через Helm

## Goals

1. Создать Helm chart для деплоя всех компонентов miRAGe в k3s
2. Настроить persistent storage для PostgreSQL, Ollama и документов
3. Настроить Ingress с автоматическим TLS через cert-manager
4. Обеспечить правильную последовательность запуска и health checks
5. Документировать процесс деплоя и верификации

## Non-Goals

- Horizontal scaling (будет 1 реплика каждого сервиса из-за local-path storage)
- Автоматическая загрузка Ollama модели (будет ручная через kubectl exec)
- Monitoring/alerting (можно добавить позже)
- Backup/restore автоматизация (можно добавить позже)

## Design

### Архитектура компонентов

В кластере k3s развертываются 5 компонентов:

1. **PostgreSQL** (pgvector/pgvector:pg16)
   - База данных с pgvector расширением
   - 1 реплика (stateful)
   - PVC: local-path, 5Gi, ReadWriteOnce

2. **Ollama** (ollama/ollama:latest)
   - Сервис генерации embeddings
   - 1 реплика (stateful)
   - PVC: local-path, 10Gi, ReadWriteOnce для моделей

3. **API** (mirage:latest)
   - FastAPI REST сервер
   - 1 реплика (из-за shared storage)
   - Health probes на `/health`

4. **Indexer** (mirage:latest)
   - Воркер для индексации документов
   - 1 реплика (из-за shared storage)
   - Выполняет Alembic миграции при старте

5. **Ingress** (Traefik)
   - Встроенный k3s Ingress controller
   - TLS через cert-manager + Let's Encrypt

### Storage стратегия

**local-path storage class:**
- Встроен в k3s по умолчанию
- Поддерживает только ReadWriteOnce
- Данные хранятся локально на узлах

**Documents PVC проблема:**
API и Indexer должны иметь доступ к одному и тому же документному хранилищу, но local-path не поддерживает ReadWriteMany.

**Решение: Node Affinity**
- Documents PVC использует local-path (ReadWriteOnce)
- API и Indexer используют `podAffinity` для закрепления на одном узле
- Оба пода монтируют один и тот же PVC

```yaml
podAffinity:
  requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchLabels:
          app.kubernetes.io/name: api  # или indexer
      topologyKey: kubernetes.io/hostname
```

### Конфигурация и секреты

**ConfigMap** содержит публичные настройки:
- `MIRAGE_OLLAMA_URL=http://<release>-ollama:11434`
- `MIRAGE_OLLAMA_MODEL=mxbai-embed-large`
- `MIRAGE_CHUNK_SIZE=400`
- `MIRAGE_CHUNK_OVERLAP=100`
- `MIRAGE_DOCUMENTS_PATH=/data/documents`

**Secret** содержит чувствительные данные:
- `postgresql-password` (генерируется или передается через `--set`)
- `api-key` (генерируется или передается через `--set`)
- `database-url` (формируется автоматически из других значений)

Секреты монтируются через `secretKeyRef` в environment variables.

### Последовательность запуска

1. **PVCs** (postgresql, ollama, documents)
2. **Secret и ConfigMap**
3. **PostgreSQL** (с healthcheck `pg_isready`)
4. **Ollama** (параллельно с PostgreSQL)
5. **Indexer** (запускает `alembic upgrade head`)
6. **API** (ждет готовности PostgreSQL через readiness probe)
7. **Ingress**

### TLS конфигурация

Ingress использует cert-manager для автоматического получения сертификатов:

```yaml
annotations:
  cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - mirage.your-domain.ru
      secretName: mirage-tls
```

ClusterIssuer создается отдельно перед деплоем miRAGe.

### Health checks и resilience

**API probes:**
- Liveness: `GET /health` (проверяет что сервер жив)
- Readiness: `GET /health` (проверяет БД и Ollama)

**PostgreSQL healthcheck:**
```bash
pg_isready -U mirage
```

**Retry логика:**
- SQLAlchemy connection pooling с автоматическими retry
- Indexer воркеры используют exponential backoff при ошибках
- Chunks сохраняются со статусом `pending` для повторной обработки

**Resource limits:**
- API: requests 256Mi/100m, limits 512Mi/500m
- Indexer: requests 512Mi/200m, limits 1Gi/1000m
- PostgreSQL: requests 256Mi/100m, limits 512Mi/500m
- Ollama: requests 2Gi/500m, limits 4Gi/2000m

### Helm Chart структура

```
helm/mirage/
├── Chart.yaml
├── values.yaml
└── templates/
    ├── _helpers.tpl
    ├── configmap.yaml
    ├── secret.yaml
    ├── postgresql-pvc.yaml
    ├── postgresql-deployment.yaml
    ├── postgresql-service.yaml
    ├── ollama-pvc.yaml
    ├── ollama-deployment.yaml
    ├── ollama-service.yaml
    ├── documents-pvc.yaml
    ├── api-deployment.yaml
    ├── api-service.yaml
    ├── indexer-deployment.yaml
    └── ingress.yaml
```

### Deployment процесс

```bash
# 1. Собрать образ
docker build -t mirage:latest .
docker save mirage:latest | sudo k3s ctr images import -

# 2. Установить cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# 3. Создать ClusterIssuer
kubectl apply -f clusterissuer.yaml

# 4. Деплой mirage
helm install mirage ./helm/mirage \
  --namespace mirage \
  --create-namespace \
  --set postgresql.auth.password=<password> \
  --set auth.apiKey=<api-key> \
  --set ingress.enabled=true \
  --set ingress.host=mirage.your-domain.ru

# 5. Загрузить модель Ollama
kubectl exec -n mirage deploy/mirage-ollama -- ollama pull mxbai-embed-large
```

### Верификация

**До деплоя:**
```bash
helm lint helm/mirage
helm install mirage ./helm/mirage --dry-run --debug
docker build -t mirage:latest .
```

**После деплоя:**
```bash
kubectl get pods -n mirage                    # Все Running
kubectl port-forward -n mirage svc/mirage-api 8000:8000
curl http://localhost:8000/health             # {"status": "ok"}
mirage documents add --project test test.pdf  # E2E тест
```

### Логирование и мониторинг

**Логи:**
Все компоненты пишут в stdout, доступны через:
```bash
kubectl logs -n mirage -l app.kubernetes.io/name=api
kubectl logs -n mirage -l app.kubernetes.io/name=indexer
```

**Метрики:**
```bash
kubectl top pods -n mirage
kubectl get pvc -n mirage
```

**Health endpoint:**
API `/health` проверяет:
- Database connection
- Ollama availability
- Возвращает 200 только если все OK

## Alternatives Considered

### Alternative 1: NFS для shared storage

**Подход:** Настроить NFS provisioner для ReadWriteMany PVC для documents.

**Почему отклонено:**
- Требует дополнительной инфраструктуры (NFS сервер)
- Усложняет setup для простого use case
- local-path достаточно для single-node deployment
- Node affinity решает проблему проще

### Alternative 2: S3-совместимое хранилище для документов

**Подход:** Хранить документы в MinIO/S3 вместо PVC.

**Почему отклонено:**
- Требует изменений в коде (добавление S3 client)
- Усложняет архитектуру
- Не нужно для текущего масштаба
- Можно добавить позже если понадобится

### Alternative 3: Init container для автозагрузки Ollama модели

**Подход:** Добавить init container к Ollama pod, который проверяет и загружает модель.

**Почему отклонено:**
- Пользователь предпочел ручной подход
- Init container увеличивает время первого старта (>1GB download)
- Ручная загрузка проще контролировать
- Модель загружается только один раз, persistence работает

### Alternative 4: StatefulSet вместо Deployment для stateful компонентов

**Подход:** Использовать StatefulSet для PostgreSQL и Ollama.

**Почему отклонено:**
- StatefulSet избыточен для single replica
- Deployment проще и достаточен
- PVC привязывается к Deployment через volumeClaimTemplate не нужен
- Нет необходимости в stable network identity

## Open Questions

Нет открытых вопросов, дизайн полностью согласован.
