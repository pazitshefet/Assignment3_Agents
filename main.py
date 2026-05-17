import argparse
from src.cli import AgentCLI
from src.config import load_config
from src.dataset import BitextDataset
from src.graph import BitextAgentGraph
from src.tools import ToolFactory
from src.memory import MemoryFactory
from langchain_openai import ChatOpenAI

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
    return ChatOpenAI(model=config.nebius_agent_model,
                      api_key=config.nebius_api_key,
                      base_url=config.nebius_base_url,
                      temperature=0)

def build_app():
    """
    Build the full agent application.

    This function loads the configuration, loads the dataset, creates the dataset
    tools, initializes the LLM, builds the LangGraph agent, and returns the compiled
    graph app together with the config.
    """
    config = load_config()
    dataset = BitextDataset(config.dataset_path)
    tools = ToolFactory(dataset).create_tools()
    llm = create_llm(config)
    memory_factory = MemoryFactory(config.memory_db_path)
    checkpointer = memory_factory.create_checkpointer()
    graph_builder = BitextAgentGraph(llm=llm,
                                     tools=tools,
                                     max_iterations=config.max_agent_iterations,
                                     checkpointer=checkpointer)
    app = graph_builder.build()
    return app, config, memory_factory

def main():
    """
    MAIN - Program entry point.

    Parses command-line arguments, builds the agent app, and then runs either demo
    mode with predefined test questions or the regular interactive CLI mode.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",
                        action="store_true",
                        help="Run the assignment demo questions automatically.")
    parser.add_argument("--session",
                        default="default",
                        help="Persistent conversation session ID. Reuse the same ID to restore memory.")
    args = parser.parse_args()
    app, config, memory_factory = build_app()
    cli = AgentCLI(app=app,
                   session_id=args.session,
                   recursion_limit=config.max_agent_iterations * 3)
    print(f"Using session: {args.session}")

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