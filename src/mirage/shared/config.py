from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    api_key: str
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "mxbai-embed-large"
    chunk_size: int = 400
    chunk_overlap: int = 100
    documents_path: str = "/data/documents"

    model_config = {"env_prefix": "MIRAGE_"}
