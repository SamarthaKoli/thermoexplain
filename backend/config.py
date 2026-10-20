import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Settings that are safe to expose to the application, not to API responses."""

    cors_origins: tuple[str, ...]
    dataset_path: Path = Path("data/raw")
    manifest_path: Path = Path("data/manifest.json")
    audit_report_path: Path = Path("data/reports/data-audit.md")
    processed_path: Path = Path("data/processed")
    models_path: Path = Path("backend/saved_models")
    llm_provider: str = "gpt_oss"
    gpt_oss_api_key: str | None = None
    gpt_oss_model: str | None = "openai/gpt-oss-120b"
    gpt_oss_base_url: str | None = None
    knowledge_documents_path: Path = Path("data/knowledge")
    knowledge_index_path: Path = Path("data/processed/knowledge-index.json")
    bert_model_path: Path | None = None
    bert_test_data_path: Path | None = None

    @classmethod
    def from_environment(cls) -> "Settings":
        project_root = Path(__file__).resolve().parent.parent
        load_dotenv(dotenv_path=project_root / ".env")
        configured_origins = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        )
        cors_origins = tuple(
            origin.strip() for origin in configured_origins.split(",") if origin.strip()
        )
        if not cors_origins:
            raise ValueError("CORS_ORIGINS must contain at least one allowed origin.")
        configured_dataset_path = (
            os.getenv("THERMOEXPLAIN_DATASET_PATH")
            or str(project_root / "data" / "raw")
        )
        llm_provider = os.getenv("LLM_PROVIDER", "gpt_oss").strip().casefold()
        if llm_provider != "gpt_oss":
            raise ValueError("LLM_PROVIDER must be gpt_oss; no other LLM provider is supported.")
        gpt_oss_api_key = (
            os.getenv("GPT_OSS_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or None
        )
        gpt_oss_model = (
            os.getenv("GPT_OSS_MODEL")
            or os.getenv("GROQ_MODEL")
            or "openai/gpt-oss-120b"
        )
        gpt_oss_base_url = (
            os.getenv("GPT_OSS_BASE_URL")
            or os.getenv("GROQ_BASE_URL")
            or None
        )
        knowledge_documents_path = Path(
            os.getenv("THERMOEXPLAIN_KNOWLEDGE_PATH")
            or project_root / "data" / "knowledge"
        ).expanduser()
        knowledge_index_path = Path(
            os.getenv("THERMOEXPLAIN_KNOWLEDGE_INDEX_PATH")
            or project_root / "data" / "processed" / "knowledge-index.json"
        ).expanduser()
        bert_model_path = os.getenv("THERMOEXPLAIN_BERT_MODEL_PATH") or None
        bert_test_data_path = os.getenv("THERMOEXPLAIN_BERT_TEST_DATA_PATH") or None
        return cls(
            cors_origins=cors_origins,
            dataset_path=Path(configured_dataset_path).expanduser(),
            manifest_path=project_root / "data" / "manifest.json",
            audit_report_path=project_root / "data" / "reports" / "data-audit.md",
            processed_path=project_root / "data" / "processed",
            models_path=project_root / "backend" / "saved_models",
            llm_provider=llm_provider,
            gpt_oss_api_key=gpt_oss_api_key,
            gpt_oss_model=gpt_oss_model,
            gpt_oss_base_url=gpt_oss_base_url,
            knowledge_documents_path=knowledge_documents_path,
            knowledge_index_path=knowledge_index_path,
            bert_model_path=Path(bert_model_path).expanduser() if bert_model_path else None,
            bert_test_data_path=(
                Path(bert_test_data_path).expanduser() if bert_test_data_path else None
            ),
        )
