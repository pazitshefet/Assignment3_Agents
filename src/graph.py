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

    Flow: START -> router -> if out_of_scope: END
                          -> agent -> tools -> agent -> ... -> END
    """
    def __init__(self, llm: BaseChatModel, tools: list[BaseTool], max_iterations: int = 12, checkpointer=None,  profile_memory=None):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.router = QueryRouter(llm)
        self.llm_with_tools = llm.bind_tools(tools)
        self.checkpointer=checkpointer
        self.profile_memory=profile_memory

    def build(self):
        """
        Build and compile the LangGraph workflow.

        The graph connects the router, agent, tool node, and fallback node into one
        executable agent.
        """
        graph = StateGraph(AgentState)
        graph.add_node("router", self._router_node)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_node("fallback", self._fallback_node)
        graph.add_edge(START, "router")
        graph.add_conditional_edges("router",
                                    self._after_router,
                            {"agent": "agent",
                                      "end": END})
        graph.add_conditional_edges("agent",
                                    self._after_agent,
                            {"tools": "tools",
                                      "fallback": "fallback",
                                      "end": END})
        graph.add_edge("tools", "agent")
        graph.add_edge("fallback", END)
        return graph.compile(checkpointer=self.checkpointer)

    def _router_node(self, state: AgentState) -> dict:
        """
        Run the dedicated router node.

        This node classifies the user query as structured, unstructured, or out-of-scope
        before the agent is allowed to select tools.
        """
        conversation_context = self._conversation_context(state)
        route_result = self.router.route(conversation_context)

        if route_result.route == "out_of_scope":
            return {"route": route_result.route,
                    "route_reason": route_result.reason,
                    "messages": [AIMessage(content=("Sorry, I can only answer questions about the Bitext "
                                                    "customer service dataset. I cannot answer this from "
                                                    "general knowledge."))
                                ]}

        return {"route": route_result.route,
                "route_reason": route_result.reason}

    def _agent_node(self, state: AgentState) -> dict:
        """
        Run one reasoning step of the agent.

        The agent receives the system prompt and conversation messages, then either
        returns a final answer or requests one or more tool calls.
        """
        route = state.get("route", "structured")
        iterations = state.get("iterations", 0) + 1

        system_prompt = self._system_prompt(route=route)

        response = self.llm_with_tools.invoke([SystemMessage(content=system_prompt)] + state["messages"])
        return {"messages": [response],
                "iterations": iterations}

    def _fallback_node(self, state: AgentState) -> dict:
        """
        Return a graceful fallback answer.

        This is used when the agent reaches the maximum number of allowed iterations
        without producing a reliable final response.
        """
        return {"messages": [AIMessage(content=("I reached the maximum number of reasoning steps before producing "
                                                "a reliable final answer. Please try asking the question more directly, "
                                                "for example by naming a category, intent, or count you want."))
                            ]}

    def _after_router(self, state: AgentState) -> Literal["agent", "end"]:
        """
        Decide what happens after the router node.

        Out-of-scope queries stop immediately, while structured and unstructured queries
        continue to the agent node.
        """
        if state.get("route") == "out_of_scope":
            return "end"
        return "agent"

    def _after_agent(self, state: AgentState) -> Literal["tools", "fallback", "end"]:
        """
        Decide what happens after the agent node.

        If the agent requested tools, execution moves to the tool node. If the iteration
        limit was reached, execution moves to fallback. Otherwise, the graph ends.
        """
        messages = state["messages"]
        last_message = messages[-1]

        if state.get("iterations", 0) >= self.max_iterations:
            return "fallback"

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        return "end"

    def _last_user_message(self, state: AgentState) -> str:
        """
        Find the latest human message in the current graph state.

        This is used by the router to classify the actual user query.
        """
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                return message.content
            if getattr(message, "type", None) == "human":
                return message.content

        raise ValueError("No user message found in state.")

    def _conversation_context(self, state: AgentState, max_messages: int = 12) -> str:
        """
        Build a short readable conversation history for the router.

        This helps the router understand follow-up questions such as 'another 3'
        or 'what about refunds?'.
        """
        recent_messages = state["messages"][-max_messages:]
        lines = []

        for message in recent_messages:
            msg_type = getattr(message, "type", "")

            if msg_type == "human":
                role = "User"
            elif msg_type == "ai":
                role = "Assistant"
            elif msg_type == "tool":
                role = "Tool"
            else:
                role = msg_type or "Message"

            content = str(getattr(message, "content", ""))
            if len(content) > 700:
                content = content[:700] + "... [truncated]"
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _system_prompt(self, route: str) -> str:
        """
        Create the system prompt for the agent.
        """
        user_profile = (self.profile_memory.load() if self.profile_memory is not None
                       else "# User Profile\n\nNo stable user facts known yet.")

        return f"""
You are a dataset analysis agent for the Bitext Customer Service Tagged Training Dataset.

The router classified the query as: {route}

The known persistent user profile:
{user_profile}

Rules:
1. Answer ONLY from the dataset.
2. Do not use general knowledge.
3. If the query needs data, use tools.
4. If the user asks for counts, distributions, categories, intents, or examples, use structured tools.
5. If the user asks for a summary, first retrieve representative rows using get_rows_for_summary.
6. If the user uses natural language like "money back", "refund requests", or "cancellation requests",
   first inspect available intents/categories when needed, then choose the closest dataset filter.
7. User profile questions are allowed and are not out-of-scope
8. If the user asks what you remember about them, answer from the persistent user profile.
9. For final answers, be concise and clear.
10. If the data is insufficient, say that clearly.
11. Do not invent categories, intents, counts, or examples.
12. For follow-up requests that ask for more examples, reuse the same filters from the previous examples request and increase the offset by the number of examples already shown for that same request context. 
    Do not restart from offset 0 unless the user changes the category, intent, or search topic.
"""
