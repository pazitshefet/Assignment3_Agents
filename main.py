from langchain_openai import AzureChatOpenAI, ChatOpenAI

from src.cli import AgentCLI
from src.config import load_config
from src.dataset import BitextDataset
from src.graph import BitextAgentGraph
from src.tools import ToolFactory


def create_llm(config):
    if config.llm_provider == "azure":
        if not config.azure_endpoint or not config.azure_deployment:
            raise ValueError(
                "For Azure, set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT in .env"
            )

        return AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            azure_deployment=config.azure_deployment,
            api_version=config.azure_api_version,
            temperature=0,
        )

    if config.llm_provider == "openai":
        return ChatOpenAI(
            model=config.openai_model,
            temperature=0,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {config.llm_provider}. "
        f"Use 'openai' or 'azure'."
    )


def main():
    config = load_config()

    dataset = BitextDataset(config.dataset_path)
    tools = ToolFactory(dataset).create_tools()

    llm = create_llm(config)

    graph_builder = BitextAgentGraph(
        llm=llm,
        tools=tools,
        max_iterations=config.max_agent_iterations,
    )

    app = graph_builder.build()

    # LangGraph recursion limit counts graph execution steps.
    # The agent's own max_iterations is separate and stricter.
    cli = AgentCLI(
        app=app,
        recursion_limit=(config.max_agent_iterations * 3),
    )

    cli.run()


if __name__ == "__main__":
    main()