import argparse

from langchain_openai import ChatOpenAI

from src.cli import AgentCLI
from src.config import load_config
from src.dataset import BitextDataset
from src.graph import BitextAgentGraph
from src.tools import ToolFactory


DEMO_QUESTIONS = [
    "What categories exist in the dataset?",
    "How many refund requests did we get?",
    "Show me 5 examples of the SHIPPING category.",
    "Summarize how agents respond to complaint intents.",
    "Show me examples of people wanting their money back.",
    "What is the distribution of intents in the ACCOUNT category?",
    "What's the best CRM software for handling complaints?",
    "Who is the president of France?",
]


def create_llm(config):
    return ChatOpenAI(
        model=config.nebius_agent_model,
        api_key=config.nebius_api_key,
        base_url=config.nebius_base_url,
        temperature=0,
    )


def build_app():
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

    return app, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",
                        action="store_true",
                        help="Run the assignment demo questions automatically.")
    args = parser.parse_args()
    app, config = build_app()
    cli = AgentCLI(app=app,
                   recursion_limit=config.max_agent_iterations * 3)

    if args.demo:
        for question in DEMO_QUESTIONS:
            print("=" * 80)
            print(f"Demo question: {question}")
            print("=" * 80)
            cli._run_single_query(question)
    else:
        cli.run()


if __name__ == "__main__":
    main()