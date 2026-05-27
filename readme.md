# Bitext Customer Service Dataset Agent

This project implements a LangGraph-based ReAct agent that answers questions about the **Bitext Customer Service Tagged Training Dataset**.

The agent supports:

1. **Structured dataset questions**
   - What categories exist in the dataset?
   - How many refund requests did we get?
   - Show me 5 examples from the SHIPPING category.
   - What is the distribution of intents in the ACCOUNT category?

2. **Unstructured dataset questions**
   - Summarize the FEEDBACK category.
   - How do customer service representatives typically respond to cancellation requests?

3. **Out-of-scope questions**
   - Who won the 2024 Champions League?
   - What is the best CRM software?
   - Write me a poem about customer service.

Out-of-scope questions are declined politely.  
The agent does not answer them from general knowledge.

The project also includes:

- persistent conversation memory using LangGraph SQLite checkpoints
- persistent per-user profile memory
- a FastMCP server exposing agent tools
- a Streamlit chat app
- a demo mode for quick testing

---

# 1. Project Structure

```text
Assignment3_Agents/
│
├── main.py
├── mcp_server.py
├── test_mcp_client.py
├── streamlit_app.py
├── requirements.txt
├── .env_example
├── README.md
├── .gitignore
│
├── data/
│   └── bitext_customer_service.csv
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── schemas.py
    ├── tools.py
    ├── router.py
    ├── graph.py
    ├── cli.py
    ├── memory.py
    └── profile_memory.py
```

Runtime files are created under:

```text
memory/
```

This folder is ignored by Git because it contains local conversation checkpoints and user profiles.

---

# 2. Setup

## Step 1: Clone the repository

```bash
git clone https://github.com/pazitshefet/Assignment3_Agents
cd Assignment3_Agents
```

## Step 2: Create a virtual environment

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

## Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

On Windows, if needed:

```cmd
py -m pip install -r requirements.txt
```

## Step 4: Configure environment variables

Copy the .env_example file to be the real .env file:

macOS / Linux:

```bash
cp .env.example .env
```

Windows:

```cmd
copy .env.example .env
```

Then edit `.env` and add your Nebius Token Factory API key.

---

# 3. Run the CLI Agent

Start the interactive command-line agent:

```bash
python main.py
```

On Windows, if needed:

```cmd
py main.py
```

You should see:

```text
Bitext Dataset Agent
Type 'exit' or 'quit' to stop.

You:
```

Then type a question.

Example questions:

```text
What categories exist in the dataset?
```
```text
How many refund requests did we get?
```
```text
Show me 5 examples of the SHIPPING category.
```
```text
Summarize how agents respond to complaint intents.
```
```text
Show me examples of people wanting their money back.
```
```text
What is the distribution of intents in the ACCOUNT category?
```

Out-of-scope examples:

```text
Who is the president of France?
```
```text
What's the best CRM software for handling complaints?
```

The agent should politely refuse out-of-scope questions.

---

# 4. CLI Options

## Demo Mode

Demo mode runs the assignment test questions automatically.

```bash
python main.py --demo
```

This is useful for checking:

- router behavior
- tool calls
- reasoning trace
- out-of-scope handling
- final answers

## Session ID

Use `--session` to resume or switch conversation memory.

```bash
python main.py --session refund_test
```

The same session ID restores the same conversation even after restarting the app.

Example:

```bash
python main.py --session refund_test
```

Ask:

```text
Show me 3 examples from the REFUND category
```

Then:

```text
Show me 3 more
```

Exit and restart:

```bash
python main.py --session refund_test
```

Ask again:

```text
Show me 3 more
```

The agent should continue the same conversation.

## User ID

Use `--user` to select a persistent user profile.

```bash
python main.py --user shay --session refund_test
```

The project combines user and session into the actual LangGraph thread ID:

```text
user_id:session_id
```

This prevents different users from accidentally sharing the same conversation checkpoint.

Example:

```bash
python main.py --user shay --session session1
```

and:

```bash
python main.py --user dana --session session1
```

use different conversation memories.

---

# 5. Reasoning Output

The CLI prints the agent's reasoning trace, not only the final answer.

Example:

```text
--- Reasoning trace ---
[router] route=structured | reason=The user asks for available dataset categories.
[agent] tool_call: list_categories({})
[tool] observation from list_categories: ...
[agent] answer_draft: The dataset contains these categories: ...

--- Final answer ---
The dataset contains these categories: ...
```

The trace shows:

- router decision
- tool calls
- tool arguments
- tool observations
- final answer

---

# 6. Streamlit Chat App

The project also includes a Streamlit chat interface.

Run:

```bash
streamlit run streamlit_app.py
```

If Streamlit is not recognized:

```bash
python -m streamlit run streamlit_app.py
```

The app provides:

- chat input for user questions
- assistant responses in a chat interface
- expandable reasoning steps
- sidebar input for User ID
- sidebar input for Session ID

The User ID controls the persistent user profile.

The Session ID controls the persistent conversation memory.

Example test:

```text
User ID: shay
Session ID: refund_test
```

Ask:

```text
Show me 3 examples from the REFUND category
```

Then:

```text
Show me 3 more
```

The reasoning expander should show the tool calls and observations.

---

# 7. MCP Server

The project includes a FastMCP server that exposes selected tools used by the LangGraph agent.

The MCP server is implemented in:

```text
mcp_server.py
```

It exposes these agent tools through MCP:

```text
list_categories
count_rows
sample_examples
```

The MCP server does not run the full LangGraph reasoning loop.  
It exposes selected tools so an external MCP client can call them directly.

Start the MCP server with:

```bash
python mcp_server.py
```

However, in normal use you do not manually type into this server.  
The MCP client starts it and communicates with it over stdio.

MCP is not REST in this project.  
There is no browser URL and no Postman endpoint.

---

# 8. Test the MCP Client

The project includes a small MCP test client:

```text
test_mcp_client.py
```

Run:

```bash
python test_mcp_client.py
```

The client will:

1. start `mcp_server.py`
2. connect to it
3. list available tools
4. call example tools

Expected output should include tools such as:

```text
list_categories
count_rows
sample_examples
```

Example tool call performed by the client:

```json
{
  "tool": "count_rows",
  "arguments": {
    "category": "REFUND"
  }
}
```

Example response:

```json
{
  "category": "REFUND",
  "intent": null,
  "text_search": null,
  "count": 1234
}
```

The exact count depends on the dataset version.

---

# 9. Connect an External MCP Client

A local MCP-compatible client can connect to this server using stdio.

Example MCP client configuration:

```json
{
  "mcpServers": {
    "bitext-agent-tools": {
      "command": "python",
      "args": ["mcp_server.py"]
    }
  }
}
```

On Windows, if the client cannot find `python`, use `py`:

```json
{
  "mcpServers": {
    "bitext-agent-tools": {
      "command": "py",
      "args": ["mcp_server.py"]
    }
  }
}
```

After connecting, the client can call tools such as:

```text
list_categories
count_rows
sample_examples
```

Example:

```text
Tool: sample_examples
Arguments:
{
  "category": "REFUND",
  "limit": 3,
  "offset": 0
}
```

---

# 10. Architecture Overview

The project is built around a LangGraph ReAct-style agent.

Main flow:

```text
User query
   ↓
Router node
   ↓
Structured / Unstructured / Out-of-scope decision
   ↓
Agent node
   ↓
Tool calls
   ↓
Tool observations
   ↓
Final answer
```

## Router

The router classifies each query before tool selection.

It returns one of:

```text
structured
unstructured
out_of_scope
```

Structured questions are concrete dataset questions, such as counts, examples, lists, or distributions.

Unstructured questions require summarization or qualitative explanation from the dataset.

User profile questions, such as:

```text
What do you remember about me?
```

are also handled as in-scope and answered from the persistent user profile.

Out-of-scope questions are rejected politely.

## Agent

The agent is implemented as a LangGraph ReAct graph.

It can:

- read the user query
- decide which tool to call
- observe the tool result
- call another tool if needed
- produce a final answer

Example multi-step query:

```text
How many refund requests did we get?
```

Possible reasoning flow:

```text
list_intents()
count_rows(intent="get_refund")
final answer
```

## Conversation Memory

Conversation memory is implemented using LangGraph checkpoints with SQLite.

The memory database path is configured in `.env`:

```env
MEMORY_DB_PATH=memory/agent_memory.sqlite
```

The checkpoint memory stores conversation state by thread ID.

The thread ID is built from:

```text
user_id:session_id
```

This allows the agent to handle follow-up questions such as:

```text
Show me 3 examples from the REFUND category
Show me 3 more
```

and:

```text
How many complaints did we get?
What about refunds?
What is the total count of the last two?
```

## User Profile Memory

User profile memory is stored separately from conversation history.

Each user has a profile file under:

```text
memory/profiles/
```

The profile is a distilled summary of stable facts, preferences, tools, environment, and recurring topics.

It is not a transcript.

Example profile facts:

```text
- User prefers short practical explanations.
- User uses PyCharm.
- User is working on a LangGraph agent assignment.
```

After every question, an LLM-based profile updater decides whether the latest turn contains new stable information.  
If yes, it updates the user's profile file.

## MCP Server

The MCP server exposes selected agent tools externally through FastMCP.

It uses the same tool logic as the LangGraph agent, but it does not run the full agent loop.

---

# 11. Tools Defined

## `list_categories`

Lists all unique categories in the dataset.

Use this for:

```text
What categories exist in the dataset?
```

## `list_intents`

Lists all unique intents.

It can optionally filter by category.

Use this when the agent needs to inspect available intents before counting or filtering.

## `count_rows`

Counts rows in the dataset.

It supports optional filtering by:

- category
- intent
- keyword search

Use this for:

```text
How many refund requests did we get?
```

## `intent_distribution`

Returns the distribution of intents.

It can optionally filter by category.

Use this for:

```text
What is the distribution of intents in the ACCOUNT category?
```

## `sample_examples`

Returns example customer queries and agent responses.

It supports:

- category
- intent
- keyword search
- limit
- offset

The `offset` parameter helps avoid repeating examples in follow-up requests such as:

```text
Show me 3 more
```

## `get_rows_for_summary`

Returns representative rows for summarization.

Use this for:

```text
Summarize the FEEDBACK category.
```

or:

```text
How do agents respond to cancellation requests?
```

---

# 12. Model Choice

This project uses only **Nebius Token Factory** models for LLM calls, as required by the assignment.

## Main Agent Model

```text
openai/gpt-oss-120b
```

This model is used for the main ReAct agent.

I chose it because the agent needs:

- reliable instruction following
- tool selection
- multi-step reasoning
- structured output
- summarization
- final answer generation

The task is not only simple question answering.  
The agent must decide when to use tools, read observations, and sometimes chain multiple tool calls.

## Router Model

```text
openai/gpt-oss-120b
```

The same model is used for routing.

The router task is simpler than the main agent task, but using the same Nebius model makes the project easier to reproduce during grading and avoids model availability issues.

A smaller Nebius model could also be used for routing, but this implementation uses one model consistently.

## Profile Updater Model

```text
openai/gpt-oss-120b
```

The same Nebius model is also used to update the persistent user profile.

The profile updater uses structured output to decide whether the latest turn contains new stable user information.

---

# 13. Max Iterations and Fallback

The agent has a maximum iteration limit.

Default:

```env
MAX_AGENT_ITERATIONS=12
```

If the agent does not produce a final answer within the limit, it returns a graceful fallback message instead of looping forever.

Example fallback:

```text
I reached the maximum number of reasoning steps before producing a reliable final answer.
Please try asking the question more directly.
```
---

# 14. Quick Start Summary

```bash
git clone https://github.com/pazitshefet/Assignment3_Agents
cd Assignment3_Agents
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows:

```cmd
git clone https://github.com/pazitshefet/Assignment3_Agents
cd Assignment3_Agents
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`, add your Nebius API key, and place the dataset here:

```text
data/bitext_customer_service.csv
```

Run the CLI:

```bash
python main.py
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Test MCP:

```bash
python test_mcp_client.py
```