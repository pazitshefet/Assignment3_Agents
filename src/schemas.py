from typing import Literal
from pydantic import BaseModel, Field

class QueryRoute(BaseModel):
    """
    Structured output schema used by the router.

    The router returns this object to decide whether the user query should be handled
    as structured, unstructured, or out-of-scope.
    """
    route: Literal["structured", "unstructured", "out_of_scope"] = Field(
        description=("structured: concrete dataset questions requiring counts, lists, examples, or distributions. "
                     "unstructured: dataset questions requiring summarization or qualitative explanation. "
                     "out_of_scope: questions not answerable from the Bitext dataset.")
    )
    reason: str = Field(description="Short reason for the classification.")


class ListIntentsInput(BaseModel):
    """
    Input schema for the list_intents tool.

    The category field is optional. If provided, the tool returns only the intents
    that belong to that specific dataset category.
    """
    category: str | None = Field(default=None,
                                 description="Optional exact category name. Use this to list intents only inside one category.")


class CountRowsInput(BaseModel):
    """
    Input schema for the count_rows tool.

    This schema allows the agent to count dataset rows using optional filters such
    as category, intent, or keyword search.
    """
    category: str | None = Field(default=None,
                                 description="Optional exact category name to filter rows.")
    intent: str | None = Field(default=None,
                               description="Optional exact intent name to filter rows.")
    text_search: str | None = Field(default=None,
                                    description=("Optional keyword search over customer queries and agent responses. "
                                                 "Supports phrases like 'refund' or 'refund OR money back OR reimbursement'."))


class IntentDistributionInput(BaseModel):
    """
    Input schema for the intent_distribution tool.

    The category field is optional. If provided, the tool returns the distribution
    of intents only inside that category.
    """
    category: str | None = Field(default=None,
                                 description="Optional exact category name. Use this for questions like distribution of intents in ACCOUNT.")


class ExamplesInput(BaseModel):
    """
    Input schema for the sample_examples tool.

    This schema lets the agent request example rows from the dataset, optionally
    filtered by category, intent, or keyword search.
    """
    category: str | None = Field(default=None,
                                 description="Optional exact category name.")
    intent: str | None = Field(default=None,
                               description="Optional exact intent name.")
    text_search: str | None = Field(default=None,
                                    description=("Optional keyword search over customer queries and agent responses. "
                                                 "Useful when the user says something indirectly, for example 'people wanting their money back'."))
    limit: int = Field(default=5,
                       ge=1,
                       le=20,
                       description="Maximum number of examples to return.")
    offset: int = Field(default=0,
                        ge=0,
                        description="Number of matching rows to skip before returning examples. Use this for follow-up requests like 'show me 3 more'.")


class RowsForSummaryInput(BaseModel):
    """
    Input schema for the get_rows_for_summary tool.

    This schema is used when the agent needs representative dataset rows before
    creating an unstructured summary.
    """
    category: str | None = Field(default=None,
                                 description="Optional exact category name to collect rows for summarization.")
    intent: str | None = Field(default=None,
                               description="Optional exact intent name to collect rows for summarization.")
    text_search: str | None = Field(default=None,
                                    description="Optional keyword search over customer queries and agent responses.")
    limit: int = Field(default=20,
                       ge=5,
                       le=50,
                       description="Maximum number of rows to return for summarization.")
