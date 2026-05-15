from langchain_core.language_models.chat_models import BaseChatModel

from src.schemas import QueryRoute


class QueryRouter:
    """
    Dedicated router node.

    It decides whether a query is:
    1. structured
    2. unstructured
    3. out_of_scope
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm.with_structured_output(QueryRoute)

    def route(self, user_query: str) -> QueryRoute:
        prompt = f"""
You are a router for a dataset analysis agent.

The agent can ONLY answer questions about the Bitext customer service dataset.

Classify the user query into one of:

structured:
Concrete dataset questions requiring counts, lists, examples, filters, or distributions.
Examples:
- What categories exist in the dataset?
- How many refund requests did we get?
- Show me 3 examples from SHIPPING.
- What is the distribution of intents in ACCOUNT?

unstructured:
Questions about the dataset that require summarization or qualitative explanation.
Examples:
- Summarize the FEEDBACK category.
- How do agents respond to cancellation requests?
- Summarize complaint intents.

out_of_scope:
Questions not answerable from the dataset.
Examples:
- Who won the 2024 Champions League?
- Write me a poem.
- What is the best CRM software?
- Who is the president of France?

Important:
If the question asks for general knowledge, advice, recommendations, or creative writing,
classify it as out_of_scope, even if it mentions customer service.

User query:
{user_query}
"""

        return self.llm.invoke(prompt)
