from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    api_key: str
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "nomic-embed-text"
    chunk_size: int = 3000
    chunk_overlap: int = 200
    child_chunk_size: int = 500
    child_chunk_overlap: int = 50
    documents_path: str = "/data/documents"
    log_level: str = "INFO"

    model_config = {"env_prefix": "MIRAGE_"}
