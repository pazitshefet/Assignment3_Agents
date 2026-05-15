from dataclasses import dataclass
import os
from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    dataset_path: str
    llm_provider: str
    openai_model: str | None
    azure_endpoint: str | None
    azure_api_version: str | None
    azure_deployment: str | None
    max_agent_iterations: int


def load_config() -> AppConfig:
    load_dotenv()

    return AppConfig(
        dataset_path=os.getenv("DATASET_PATH", "data/bitext_customer_service.csv"),
        llm_provider=os.getenv("LLM_PROVIDER", "openai").lower().strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        max_agent_iterations=int(os.getenv("MAX_AGENT_ITERATIONS", "12")),
    )