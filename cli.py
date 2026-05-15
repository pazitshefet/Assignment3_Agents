from __future__ import annotations

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class AgentCLI:
    """
    Interactive command-line interface.

    It prints:
    - router result
    - tool calls
    - tool observations
    - final answer
    """

    def __init__(self, app, recursion_limit: int = 30):
        self.app = app
        self.recursion_limit = recursion_limit

    def run(self) -> None:
        print("\nBitext Dataset Agent")
        print("Type 'exit' or 'quit' to stop.\n")

        while True:
            user_query = input("You: ").strip()

            if user_query.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break

            if not user_query:
                continue

            self._run_single_query(user_query)

    def _run_single_query(self, user_query: str) -> None:
        print("\n--- Reasoning trace ---")

        final_answer = None

        try:
            events = self.app.stream(
                {
                    "messages": [HumanMessage(content=user_query)],
                    "route": None,
                    "route_reason": None,
                    "iterations": 0,
                },
                config={"recursion_limit": self.recursion_limit},
                stream_mode="updates",
            )

            for event in events:
                for node_name, update in event.items():
                    self._print_node_update(node_name, update)

                    messages = update.get("messages", [])
                    for message in messages:
                        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                            final_answer = message.content

            print("\n--- Final answer ---")
            print(final_answer or "No final answer was produced.")

        except Exception as exc:
            print("\n--- Final answer ---")
            print(
                "The agent failed before completing the answer. "
                f"Error: {type(exc).__name__}: {exc}"
            )

        print()

    def _print_node_update(self, node_name: str, update: dict) -> None:
        if node_name == "router":
            route = update.get("route")
            reason = update.get("route_reason")

            if route:
                print(f"[router] route={route} | reason={reason}")

        messages = update.get("messages", [])

        for message in messages:
            if isinstance(message, AIMessage):
                tool_calls = getattr(message, "tool_calls", None)

                if tool_calls:
                    for tool_call in tool_calls:
                        name = tool_call.get("name")
                        args = tool_call.get("args")
                        print(f"[agent] tool_call: {name}({args})")
                else:
                    preview = self._truncate(message.content)
                    print(f"[agent] answer_draft: {preview}")

            elif isinstance(message, ToolMessage):
                preview = self._truncate(message.content)
                print(f"[tool] observation from {message.name}: {preview}")

    def _truncate(self, text: str, max_chars: int = 700) -> str:
        text = str(text)

        if len(text) <= max_chars:
            return text

        return text[:max_chars] + "... [truncated]"