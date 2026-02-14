# tests/shared/test_models_registry.py
import pytest
from mirage.shared.models_registry import (
    SupportedModel,
    get_model,
    get_all_models,
    get_model_table_name,
)


def test_get_all_models_returns_three():
    models = get_all_models()
    assert len(models) == 3
    names = [m.name for m in models]
    assert "nomic-embed-text" in names
    assert "bge-m3" in names
    assert "mxbai-embed-large" in names


def test_get_model_by_name():
    model = get_model("bge-m3")
    assert model is not None
    assert model.name == "bge-m3"
    assert model.dimensions == 1024
    assert model.context_length == 8192


def test_get_model_unknown_returns_none():
    model = get_model("unknown-model")
    assert model is None


def test_get_model_table_name():
    assert get_model_table_name("nomic-embed-text") == "embeddings_nomic_768"
    assert get_model_table_name("bge-m3") == "embeddings_bge_m3_1024"
    assert get_model_table_name("mxbai-embed-large") == "embeddings_mxbai_1024"
