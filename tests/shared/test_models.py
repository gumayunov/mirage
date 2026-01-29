import uuid
from datetime import datetime
from mirage.shared.models import Project, Document, Chunk, IndexingTask, DocumentStatus, TaskStatus


def test_project_model():
    project = Project(
        id=uuid.uuid4(),
        name="test-project",
        created_at=datetime.utcnow(),
    )
    assert project.name == "test-project"


def test_document_model():
    project_id = uuid.uuid4()
    doc = Document(
        id=uuid.uuid4(),
        project_id=project_id,
        filename="test.pdf",
        original_path="/data/documents/test.pdf",
        file_type="pdf",
        status=DocumentStatus.PENDING,
        created_at=datetime.utcnow(),
    )
    assert doc.status == DocumentStatus.PENDING


def test_chunk_model():
    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Test content",
        embedding=[0.1] * 1024,
        position=0,
        structure={"chapter": "Test"},
    )
    assert chunk.position == 0


def test_indexing_task_model():
    task = IndexingTask(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        task_type="index",
        status=TaskStatus.PENDING,
        created_at=datetime.utcnow(),
    )
    assert task.status == TaskStatus.PENDING
