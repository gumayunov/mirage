from dataclasses import dataclass

SUPPORTED_MODELS: dict[str, "SupportedModel"] = {}


@dataclass(frozen=True)
class SupportedModel:
    name: str
    dimensions: int
    context_length: int
    ollama_name: str
    table_alias: str  # Short name for table naming


_nomic = SupportedModel(
    name="nomic-embed-text",
    dimensions=768,
    context_length=8192,
    ollama_name="nomic-embed-text",
    table_alias="nomic",
)
_bge_m3 = SupportedModel(
    name="bge-m3",
    dimensions=1024,
    context_length=8192,
    ollama_name="bge-m3",
    table_alias="bge_m3",
)
_mxbai = SupportedModel(
    name="mxbai-embed-large",
    dimensions=1024,
    context_length=512,
    ollama_name="mxbai-embed-large",
    table_alias="mxbai",
)

SUPPORTED_MODELS = {
    _nomic.name: _nomic,
    _bge_m3.name: _bge_m3,
    _mxbai.name: _mxbai,
}


def get_all_models() -> list[SupportedModel]:
    return list(SUPPORTED_MODELS.values())


def get_model(name: str) -> SupportedModel | None:
    return SUPPORTED_MODELS.get(name)


def get_model_table_name(model_name: str) -> str:
    model = get_model(model_name)
    if model is None:
        raise ValueError(f"Unknown model: {model_name}")
    return f"embeddings_{model.table_alias}_{model.dimensions}"
