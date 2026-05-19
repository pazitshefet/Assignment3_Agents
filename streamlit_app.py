import uuid
import streamlit as st
from main import build_app
from typing import Any
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

def init_state() -> None:
    """
    Initialize Streamlit session state.

    Streamlit reruns the script on every interaction, so we keep the compiled
    app and chat display history inside st.session_state.
    """
    if "user_id" not in st.session_state:
        st.session_state.user_id = "default_user"

    if "session_id" not in st.session_state:
        st.session_state.session_id = "default_session"

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = build_thread_id(st.session_state.user_id,
                                                     st.session_state.session_id)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "agent_bundle" not in st.session_state:
        st.session_state.agent_bundle = None


def build_thread_id(user_id: str, session_id: str) -> str:
    """
    Build the LangGraph thread ID.

    Combining user and session prevents different users from accidentally sharing
    the same conversation checkpoint.
    """
    return f"{user_id}:{session_id}"


def reset_agent_if_needed(user_id: str, session_id: str) -> None:
    """
    Rebuild the agent if the user ID or session ID changed.

    The user profile belongs to the user ID, while the conversation checkpoint
    is selected by the combined thread ID.
    """
    new_thread_id = build_thread_id(user_id, session_id)
    changed = (user_id != st.session_state.user_id or
               session_id != st.session_state.session_id or
               new_thread_id != st.session_state.thread_id)
    if changed:
        st.session_state.user_id = user_id
        st.session_state.session_id = session_id
        st.session_state.thread_id = new_thread_id
        st.session_state.chat_history = []
        st.session_state.agent_bundle = None


def get_agent_bundle() -> tuple[Any, Any, Any, Any, Any]:
    """
    Build or reuse the agent bundle.

    build_app returns the compiled LangGraph app, config, conversation memory
    factory, user profile memory, and LLM used for profile updates.
    """
    if st.session_state.agent_bundle is None:
        st.session_state.agent_bundle = build_app(st.session_state.user_id)

    return st.session_state.agent_bundle


def run_agent_query(user_query: str) -> tuple[str, list[str]]:
    """
    Run one user query through the LangGraph agent.

    Returns the final answer and a list of reasoning trace lines, including
    router decisions, tool calls, and tool observations.
    """
    app, config, _memory_factory, profile_memory, profile_llm = get_agent_bundle()
    input_state = {"messages": [HumanMessage(content=user_query)],
                   "route": None,
                   "route_reason": None,
                   "iterations": 0}
    run_config = {"configurable": {"thread_id": st.session_state.thread_id},
                  "recursion_limit": config.max_agent_iterations * 3}

    final_answer = ""
    reasoning_steps: list[str] = []
    events = app.stream(input=input_state,
                        config=run_config,
                        stream_mode="updates")

    for event in events:
        for node_name, update in event.items():
            reasoning_steps.extend(format_node_update(node_name, update))
            messages = update.get("messages", [])
            for message in messages:
                if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                    if str(message.content).strip():
                        final_answer = message.content

    if not final_answer:
        final_answer = "No final answer was produced."

    if profile_memory is not None and profile_llm is not None:
        profile_memory.update_after_turn(llm=profile_llm,
                                         user_message=user_query,
                                         assistant_answer=final_answer)

    return final_answer, reasoning_steps


def format_node_update(node_name: str, update: dict) -> list[str]:
    """
    Convert a LangGraph node update into readable reasoning trace lines.

    This mirrors the CLI behavior but returns strings so Streamlit can display
    them inside expanders.
    """
    lines: list[str] = []

    if node_name == "router":
        route = update.get("route")
        reason = update.get("route_reason")

        if route:
            lines.append(f"Router: route={route} | reason={reason}")

    messages = update.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None)

            if tool_calls:
                for tool_call in tool_calls:
                    name = tool_call.get("name")
                    args = tool_call.get("args")
                    lines.append(f"Agent tool call: {name}({args})")
            else:
                content = str(message.content).strip()
                if content:
                    lines.append(f"Agent answer draft: {truncate(content)}")

        elif isinstance(message, ToolMessage):
            lines.append(
                f"Tool observation from {message.name}: {truncate(message.content)}"
            )

    return lines


def truncate(text: str, max_chars: int = 1200) -> str:
    """
    Shorten long reasoning outputs for the Streamlit UI.
    """
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def render_sidebar() -> None:
    """
    Render the sidebar controls.

    The user can set user ID and session ID to resume or switch conversations.
    """
    st.sidebar.header("Session Settings")
    user_id = st.sidebar.text_input("User ID",
                                    value=st.session_state.user_id,
                                    help="Controls the persistent user profile.")
    session_id = st.sidebar.text_input("Session ID",
                                       value=st.session_state.session_id,
                                       help="Controls the persistent conversation memory.")
    reset_agent_if_needed(user_id=user_id, session_id=session_id)

    st.sidebar.caption(f"**Thread ID: `{st.session_state.thread_id}`**")
    if st.sidebar.button("Clear visible chat"):
        st.session_state.chat_history = []
        st.rerun()
    st.sidebar.markdown("""
        <p style="font-size: 15px; color: #262730; margin: 0; line-height: 1.5;">
            <b>Changing the user ID</b> switches the persistent user profile.<br>
            <b>Changing the session ID</b> resumes or starts a different LangGraph checkpoint.
        </p>
        """, unsafe_allow_html=True)


def render_chat_history() -> None:
    """
    Render all visible chat messages and reasoning traces.
    """
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

            if item["role"] == "assistant" and item.get("reasoning"):
                with st.expander("Reasoning steps"):
                    for step in item["reasoning"]:
                        st.markdown(f"- {step}")


def main() -> None:
    """
    Streamlit app entry point.

    Shows a chat interface, sends user questions to the LangGraph agent, displays
    final answers, and exposes reasoning steps in an expandable section.
    """
    st.set_page_config(page_title="Bitext Dataset Agent",
                       page_icon="🤖",
                       layout="wide")
    init_state()

    st.title("Bitext Customer Service Dataset Agent")
    st.caption("LangGraph ReAct agent with persistent session memory and user profiles.")

    render_sidebar()
    render_chat_history()

    user_query = st.chat_input("Ask a question about the Bitext dataset...")
    if user_query:
        st.session_state.chat_history.append({"id": str(uuid.uuid4()),
                                              "role": "user",
                                              "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    final_answer, reasoning_steps = run_agent_query(user_query)
                except Exception as exc:
                    final_answer = ("The agent failed before completing the answer. "
                                    f"Error: {type(exc).__name__}: {exc}")
                    reasoning_steps = []

            st.markdown(final_answer)
            if reasoning_steps:
                with st.expander("Reasoning steps"):
                    for step in reasoning_steps:
                        st.markdown(f"- {step}")

        st.session_state.chat_history.append({"id": str(uuid.uuid4()),
                                              "role": "assistant",
                                              "content": final_answer,
                                              "reasoning": reasoning_steps})

if __name__ == "__main__":
    main()