import os
from dotenv import load_dotenv
from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    """
    application's configuration
    """
    dataset_path: str
    nebius_api_key: str
    nebius_base_url: str
    nebius_agent_model: str
    nebius_router_model: str
    max_agent_iterations: int
    memory_db_path: str

def load_config() -> AppConfig:
    """
    load application's configuration from .env file
    """
    load_dotenv()

    return AppConfig(
        dataset_path=os.getenv("DATASET_PATH", "data/bitext_customer_service.csv"),
        nebius_api_key=os.getenv("NEBIUS_API_KEY", ""),
        nebius_base_url=os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"),
        nebius_agent_model=os.getenv("NEBIUS_AGENT_MODEL", "openai/gpt-oss-120b"),
        nebius_router_model=os.getenv("NEBIUS_ROUTER_MODEL", "openai/gpt-oss-120b"),
        max_agent_iterations=int(os.getenv("MAX_AGENT_ITERATIONS", "12")),
        memory_db_path=os.getenv("MEMORY_DB_PATH", "memory/agent_memory.sqlite")
    )