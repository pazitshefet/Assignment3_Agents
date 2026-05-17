from src.schemas import QueryRoute
from langchain_core.language_models.chat_models import BaseChatModel

class QueryRouter:
    """
    Dedicated router node.

    It decides whether a query is: structured, unstructured or out_of_scope
    """
    def __init__(self, llm: BaseChatModel):
        self.llm = llm.with_structured_output(QueryRoute)

    def route(self, conversation_context: str) -> QueryRoute:
        """
        Classify the user's query before tool selection begins.

        The method asks the LLM to choose one route: structured, unstructured, or
        out_of_scope, and returns the decision with a short explanation.
        """
        prompt = f"""
    You are a router for a dataset analysis agent.

    The agent can ONLY answer questions about the Bitext customer service dataset.

    You will receive recent conversation history.
    Classify the latest user query using the full conversation context.

    Classify into one of:

    structured:
    Concrete dataset questions requiring counts, lists, examples, filters, or distributions.
    Follow-up questions like "show me 3 more", "another 3", "what about refunds?",
    or "what is the total count of the last two?" are structured if they refer to
    previous dataset questions.

    unstructured:
    Questions about the dataset that require summarization or qualitative explanation.
    Follow-up summary questions are also unstructured if they refer to previous
    dataset summaries.

    out_of_scope:
    Questions not answerable from the dataset.

    Important:
    If the latest message is a short follow-up and the previous conversation was
    about the dataset, do NOT mark it out_of_scope.

    Conversation:
    {conversation_context}
    """

        return self.llm.invoke(prompt)
