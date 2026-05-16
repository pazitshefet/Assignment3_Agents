from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict
from src.dataset import BitextDataset
from src.schemas import (CountRowsInput,
                         ExamplesInput,
                         IntentDistributionInput,
                         ListIntentsInput,
                         RowsForSummaryInput)

class DatasetTool(BaseTool):
    """
    Base class for all dataset tools.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dataset: BitextDataset

class ListCategoriesTool(DatasetTool):
    """
    Tool for listing all unique dataset categories.

    The agent uses this when the user asks what categories exist or when it needs
    to inspect the dataset structure before filtering.
    """
    name: str = "list_categories"
    description: str = (
        "Use this tool to list all unique customer service categories in the Bitext dataset. "
        "Use it before filtering if you are not sure which categories exist."
    )

    def _run(self) -> str:
        return self.dataset.to_json({"categories": self.dataset.categories()})


class ListIntentsTool(DatasetTool):
    """
    Tool for listing all unique dataset intents.

    The agent can list all intents or only the intents inside a specific category,
    which helps it choose the correct intent before counting or filtering.
    """
    name: str = "list_intents"
    description: str = ("Use this tool to list all unique intents in the dataset. "
                        "Optionally provide an exact category name to list intents only inside that category. "
                        "Use this before counting if the user describes an intent in natural language, "
                        "for example 'refund requests' or 'cancellation requests'.")
    args_schema: Type[BaseModel] = ListIntentsInput

    def _run(self, category: str | None = None) -> str:
        return self.dataset.to_json({"category": category,
                                     "intents": self.dataset.intents(category=category)})

class CountRowsTool(DatasetTool):
    """
    Tool for counting dataset rows.

    The agent uses this for structured questions that ask how many records match a
    category, intent, or keyword search.
    """
    name: str = "count_rows"
    description: str = ("Use this tool to count dataset rows after filtering by exact category, exact intent, "
                        "or keyword search. For questions like 'How many refund requests did we get?', "
                        "first use list_intents if needed, then call this tool with the closest exact intent.")
    args_schema: Type[BaseModel] = CountRowsInput

    def _run(self, category: str | None = None,
             intent: str | None = None, text_search: str | None = None) -> str:
        count = self.dataset.count(category=category,
                                   intent=intent,
                                   text_search=text_search)
        return self.dataset.to_json({"category": category,
                                     "intent": intent,
                                     "text_search": text_search,
                                     "count": count})

class IntentDistributionTool(DatasetTool):
    """
    Tool for calculating intent distribution.

    The agent uses this when the user asks how intents are distributed in the whole
    dataset or inside a specific category.
    """
    name: str = "intent_distribution"
    description: str = ("Use this tool to calculate the distribution of intents. "
                        "For example: 'What is the distribution of intents in the ACCOUNT category?' "
                        "Provide category='ACCOUNT' if the user asks about a specific category.")
    args_schema: Type[BaseModel] = IntentDistributionInput

    def _run(self, category: str | None = None) -> str:
        return self.dataset.to_json({"category": category,
                                     "distribution": self.dataset.intent_distribution(category=category)})

class SampleExamplesTool(DatasetTool):
    """
    Tool for returning example dataset rows.

    The agent uses this when the user asks to see sample customer queries and agent
    responses, optionally filtered by category, intent, or keyword search.
    """
    name: str = "sample_examples"
    description: str = ("Use this tool to show concrete examples from the dataset. "
                        "It can filter by exact category, exact intent, or keyword search. "
                        "Use it for questions like 'Show me 3 examples from SHIPPING' or "
                        "'Show me examples of people wanting their money back'.")
    args_schema: Type[BaseModel] = ExamplesInput

    def _run(self, category: str | None = None,
        intent: str | None = None, text_search: str | None = None, limit: int = 5) -> str:
        examples = self.dataset.examples(category=category,
                                         intent=intent,
                                         text_search=text_search,
                                         limit=limit)
        return self.dataset.to_json({"category": category,
                                     "intent": intent,
                                     "text_search": text_search,
                                     "limit": limit,
                                     "examples": examples})

class RowsForSummaryTool(DatasetTool):
    """
    Tool for retrieving rows before summarization.

    The agent uses this for unstructured questions where it first needs relevant
    dataset examples and then summarizes patterns from those rows.
    """
    name: str = "get_rows_for_summary"
    description: str = ("Use this tool for unstructured dataset questions that require summarization. "
                        "It returns representative customer queries and agent responses from a category, intent, "
                        "or keyword search. After using this tool, summarize patterns from the returned rows only.")
    args_schema: Type[BaseModel] = RowsForSummaryInput

    def _run(self, category: str | None = None,
             intent: str | None = None, text_search: str | None = None, limit: int = 20) -> str:
        rows = self.dataset.rows_for_summary(category=category,
                                             intent=intent,
                                             text_search=text_search,
                                             limit=limit)
        return self.dataset.to_json({"category": category,
                                     "intent": intent,
                                     "text_search": text_search,
                                     "limit": limit,
                                     "rows": rows})

class ToolFactory:
    """
    Factory class for creating all tools.

    Returns the full list of tools used by the LangGraph agent.
    """
    def __init__(self, dataset: BitextDataset):
        self.dataset = dataset

    def create_tools(self) -> list[BaseTool]:
        return [ListCategoriesTool(dataset=self.dataset),
                ListIntentsTool(dataset=self.dataset),
                CountRowsTool(dataset=self.dataset),
                IntentDistributionTool(dataset=self.dataset),
                SampleExamplesTool(dataset=self.dataset),
                RowsForSummaryTool(dataset=self.dataset)]
