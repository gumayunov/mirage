import os

from dotenv import load_dotenv

load_dotenv()


def get_api_url() -> str:
    return os.environ.get("MIRAGE_API_URL", "http://localhost:8000/api/v1")


def get_api_key() -> str:
    key = os.environ.get("MIRAGE_API_KEY", "")
    if not key:
        raise ValueError("MIRAGE_API_KEY environment variable not set")
    return key
