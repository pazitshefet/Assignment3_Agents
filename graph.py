from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.router import QueryRouter


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str | None
    route_reason: str | None
    iterations: int


class BitextAgentGraph:
    """
    Builds the LangGraph ReAct-style agent.

    Flow:
    START
      -> router
      -> if out_of_scope: END
      -> agent
      -> tools
      -> agent
      -> ...
      -> END

    The router is separate, as required by the assignment.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        max_iterations: int = 12,
    ):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.router = QueryRouter(llm)
        self.llm_with_tools = llm.bind_tools(tools)

    def build(self):
        graph = StateGraph(AgentState)

        graph.add_node("router", self._router_node)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_node("fallback", self._fallback_node)

        graph.add_edge(START, "router")

        graph.add_conditional_edges(
            "router",
            self._after_router,
            {
                "agent": "agent",
                "end": END,
            },
        )

        graph.add_conditional_edges(
            "agent",
            self._after_agent,
            {
                "tools": "tools",
                "fallback": "fallback",
                "end": END,
            },
        )

        graph.add_edge("tools", "agent")
        graph.add_edge("fallback", END)

        return graph.compile()

    def _router_node(self, state: AgentState) -> dict:
        user_query = self._last_user_message(state)
        route_result = self.router.route(user_query)

        if route_result.route == "out_of_scope":
            return {
                "route": route_result.route,
                "route_reason": route_result.reason,
                "messages": [
                    AIMessage(
                        content=(
                            "Sorry, I can only answer questions about the Bitext "
                            "customer service dataset. I cannot answer this from "
                            "general knowledge."
                        )
                    )
                ],
            }

        return {
            "route": route_result.route,
            "route_reason": route_result.reason,
        }

    def _agent_node(self, state: AgentState) -> dict:
        route = state.get("route", "structured")
        iterations = state.get("iterations", 0) + 1

        system_prompt = self._system_prompt(route=route)

        response = self.llm_with_tools.invoke(
            [SystemMessage(content=system_prompt)] + state["messages"]
        )

        return {
            "messages": [response],
            "iterations": iterations,
        }

    def _fallback_node(self, state: AgentState) -> dict:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I reached the maximum number of reasoning steps before producing "
                        "a reliable final answer. Please try asking the question more directly, "
                        "for example by naming a category, intent, or count you want."
                    )
                )
            ]
        }

    def _after_router(self, state: AgentState) -> Literal["agent", "end"]:
        if state.get("route") == "out_of_scope":
            return "end"
        return "agent"

    def _after_agent(self, state: AgentState) -> Literal["tools", "fallback", "end"]:
        messages = state["messages"]
        last_message = messages[-1]

        if state.get("iterations", 0) >= self.max_iterations:
            return "fallback"

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        return "end"

    def _last_user_message(self, state: AgentState) -> str:
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                return message.content
            if getattr(message, "type", None) == "human":
                return message.content

        raise ValueError("No user message found in state.")

    def _system_prompt(self, route: str) -> str:
        return f"""
You are a dataset analysis agent for the Bitext Customer Service Tagged Training Dataset.

The router classified the query as: {route}

Rules:
1. Answer ONLY from the dataset.
2. Do not use general knowledge.
3. If the query needs data, use tools.
4. If the user asks for counts, distributions, categories, intents, or examples, use structured tools.
5. If the user asks for a summary, first retrieve representative rows using get_rows_for_summary.
6. If the user uses natural language like "money back", "refund requests", or "cancellation requests",
   first inspect available intents/categories when needed, then choose the closest dataset filter.
7. For final answers, be concise and clear.
8. If the data is insufficient, say that clearly.
9. Do not invent categories, intents, counts, or examples.
"""