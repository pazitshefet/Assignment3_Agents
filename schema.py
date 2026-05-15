from typing import Literal

from pydantic import BaseModel, Field


class QueryRoute(BaseModel):
    route: Literal["structured", "unstructured", "out_of_scope"] = Field(
        description=(
            "structured: concrete dataset questions requiring counts, lists, examples, or distributions. "
            "unstructured: dataset questions requiring summarization or qualitative explanation. "
            "out_of_scope: questions not answerable from the Bitext dataset."
        )
    )
    reason: str = Field(description="Short reason for the classification.")


class ListIntentsInput(BaseModel):
    category: str | None = Field(
        default=None,
        description="Optional exact category name. Use this to list intents only inside one category.",
    )


class CountRowsInput(BaseModel):
    category: str | None = Field(
        default=None,
        description="Optional exact category name to filter rows.",
    )
    intent: str | None = Field(
        default=None,
        description="Optional exact intent name to filter rows.",
    )
    text_search: str | None = Field(
        default=None,
        description=(
            "Optional keyword search over customer queries and agent responses. "
            "Supports phrases like 'refund' or 'refund OR money back OR reimbursement'."
        ),
    )


class IntentDistributionInput(BaseModel):
    category: str | None = Field(
        default=None,
        description="Optional exact category name. Use this for questions like distribution of intents in ACCOUNT.",
    )


class ExamplesInput(BaseModel):
    category: str | None = Field(
        default=None,
        description="Optional exact category name.",
    )
    intent: str | None = Field(
        default=None,
        description="Optional exact intent name.",
    )
    text_search: str | None = Field(
        default=None,
        description=(
            "Optional keyword search over customer queries and agent responses. "
            "Useful when the user says something indirectly, for example 'people wanting their money back'."
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of examples to return.",
    )


class RowsForSummaryInput(BaseModel):
    category: str | None = Field(
        default=None,
        description="Optional exact category name to collect rows for summarization.",
    )
    intent: str | None = Field(
        default=None,
        description="Optional exact intent name to collect rows for summarization.",
    )
    text_search: str | None = Field(
        default=None,
        description="Optional keyword search over customer queries and agent responses.",
    )
    limit: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Maximum number of rows to return for summarization.",
    )