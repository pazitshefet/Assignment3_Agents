import json
from typing import Any
from fastmcp import FastMCP
from src.config import load_config
from src.dataset import BitextDataset
from src.tools import ToolFactory

config = load_config()
dataset = BitextDataset(config.dataset_path)
agent_tools = ToolFactory(dataset).create_tools()
tools_by_name = {tool.name: tool for tool in agent_tools}

mcp = FastMCP("Bitext Agent Tools")

def _parse_tool_result(result: Any) -> Any:
    """
    Convert LangChain tool output into a Python object when possible.
    """
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"result": result}

    return result

@mcp.tool
def list_categories() -> dict:
    """
    Call the agent's list_categories tool.
    Returns all unique customer service categories in the Bitext dataset.
    """
    result = tools_by_name["list_categories"].invoke({})
    return _parse_tool_result(result)

@mcp.tool
def count_rows(category: str | None = None, intent: str | None = None, text_search: str | None = None) -> dict:
    """
    Call the agent's count_rows tool.
    Counts rows in the Bitext dataset using optional category, intent, or keyword filters.
    """
    result = tools_by_name["count_rows"].invoke({
        "category": category,
        "intent": intent,
        "text_search": text_search,
    })
    return _parse_tool_result(result)

@mcp.tool
def sample_examples(category: str | None = None, intent: str | None = None, text_search: str | None = None, limit: int = 5, offset: int = 0) -> dict:
    """
    Call the agent's sample_examples tool.
    Returns example customer queries and agent responses from the Bitext dataset.
    """
    result = tools_by_name["sample_examples"].invoke({
        "category": category,
        "intent": intent,
        "text_search": text_search,
        "limit": limit,
        "offset": offset,
    })
    return _parse_tool_result(result)

if __name__ == "__main__":
    mcp.run()