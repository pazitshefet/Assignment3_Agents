# Bitext Customer Service Dataset Agent

This project implements a LangGraph-based ReAct agent that answers questions about the **Bitext Customer Service Tagged Training Dataset**.

The agent can answer questions about the dataset only.

It supports three query types:

1. **Structured questions**
   - What categories exist in the dataset?
   - How many refund requests did we get?
   - Show me 5 examples from the SHIPPING category.
   - What is the distribution of intents in the ACCOUNT category?

2. **Unstructured questions**
   - Summarize the FEEDBACK category.
   - How do agents respond to cancellation requests?
   - Summarize how agents respond to complaint intents.

3. **Out-of-scope questions**
   - Who won the 2024 Champions League?
   - What is the best CRM software?
   - Write me a poem about customer service.

Out-of-scope questions are declined politely.  
The agent does not answer them from general knowledge.

---

# 1. Project Structure

```text
Assignment3_agents/
│
├── main.py
├── requirements.txt
├── .env.example
├── README.md
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
    └── mcp_server.py
```

---

# 2. Setup

## Step 1: Clone the repository

```bash
git clone https://github.com/pazitshefet/Assignment3_Agents
cd Assignment3_agents
```

## Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it.

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

## Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Update the `.env` file

Edit `.env` and set your Nebius Token Factory API key.
other properties should not be changed, but it is possible to update them
For example: change the LLM models or the maximum number of
---

# 4. How to Run the CLI

Start the interactive command-line agent:
```bash
python main.py
```

You should see:
```text
Bitext Dataset Agent
Type 'exit' or 'quit' to stop.

You:
```

And then you should insert your question
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
The agent should politely refuse these because they are not answerable from the dataset.

## Demo Mode

The project also includes an optional demo mode.
Demo mode runs the 8 test questions specified above automatically, one after another.
This is useful for quickly checking that the router, tools, reasoning trace, out-of-scope handling, and final answers all work correctly.

Run:
```bash
python main.py --demo
````
Manual mode is still the default

---

# 5. Reasoning Output

The CLI prints the agent's reasoning trace.

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
---

# 5. MCP Server

The project also exposes selected dataset tools through a FastMCP server.
Start the MCP server:
```bash
python -m src.mcp_server
```

The MCP server exposes these tools:
```text
list_categories
list_intents
count_rows
intent_distribution
sample_examples
get_rows_for_summary
```

These are the same core dataset tools used by the LangGraph agent.

---

# 6. Connecting an MCP Client

A local MCP client can connect to the server over stdio.

Example MCP client configuration:

```json
{
  "mcpServers": {
    "bitext-dataset-tools": {
      "command": "python",
      "args": ["-m", "src.mcp_server"]
    }
  }
}
```

After connecting, the client can call a tool such as:

```text
list_categories
```

Example result:

```json
{
  "categories": [
    "ACCOUNT",
    "CANCELLATION",
    "FEEDBACK",
    "ORDER",
    "PAYMENT",
    "REFUND",
    "SHIPPING"
  ]
}
```

The exact categories depend on the dataset version.

---

# 7. Architecture Overview

The project is built around a LangGraph ReAct-style agent.

The main flow is:

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

The router classifies each user query before the agent starts tool selection.

It returns one of:

```text
structured
unstructured
out_of_scope
```

Out-of-scope queries are rejected immediately.

This prevents the agent from answering general knowledge questions that are unrelated to the dataset.

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

## Persistent Memory

The graph uses persistent memory with a SQLite checkpointer.

This allows conversation state to be saved across turns.

The memory database path is configured in `.env`:

```env
MEMORY_DB_PATH=memory/agent_memory.sqlite
```

## MCP Server

The FastMCP server exposes the dataset tools externally.

This means an MCP-compatible client can use the same dataset analysis tools without running the full CLI agent.

---

# 8. Tools Defined

## `list_categories`

Lists all unique categories in the dataset.

Use this for questions like:

```text
What categories exist in the dataset?
```

## `list_intents`

Lists all unique intents.

It can optionally filter by category.

Use this when the agent needs to understand which intents exist before counting or filtering.

Example:

```text
What intents exist in the ACCOUNT category?
```

## `count_rows`

Counts rows in the dataset.

It supports filtering by:

- category
- intent
- keyword search

Use this for questions like:

```text
How many refund requests did we get?
```

## `intent_distribution`

Returns the count of each intent.

It can optionally filter by category.

Use this for questions like:

```text
What is the distribution of intents in the ACCOUNT category?
```

## `sample_examples`

Returns example customer queries and agent responses.

Use this for questions like:

```text
Show me 5 examples from the SHIPPING category.
```

or:

```text
Show me examples of people wanting their money back.
```

## `get_rows_for_summary`

Returns representative rows for summarization.

Use this for unstructured questions like:

```text
Summarize the FEEDBACK category.
```

or:

```text
How do agents respond to cancellation requests?
```

---

# 9. Model Choice

This project uses only **Nebius Token Factory** models for LLM calls, as required by the assignment.

## Main model

```text
openai/gpt-oss-120b
```

I use `gpt-oss-120b` for the main ReAct agent.

I chose this model because the agent needs:

- reliable instruction following
- tool selection
- multi-step reasoning
- structured output
- summarization
- final answer generation

The task is not only simple question answering.  
The agent must decide when to use tools, read observations, and sometimes chain multiple tool calls.

## Router model

```text
openai/gpt-oss-120b
```

I use the same model for the router.

The router task is simple, but using the same model for both routing and generation makes the project easier to reproduce for grading.

A smaller Nebius model could also be used for routing, but using one model avoids model availability issues.

## Why not GPT?

I did not use GPT models from OpenAI because the assignment allows only Nebius Token Factory models for LLM calls.

Even if GPT models are strong for tool calling and agent workflows, using them would violate the project requirements.

`gpt-oss-120b` is acceptable here because it is used through Nebius Token Factory.

---

# 10. Max Iterations and Fallback

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

This prevents infinite tool-calling loops.

---

# 11. Example Test Queries

Use these queries to verify the project.

## Structured query

```text
What categories exist in the dataset?
```

Purpose:

Tests structured routing and the `list_categories` tool.

## Structured count query

```text
How many refund requests did we get?
```

Purpose:

Tests multi-step reasoning.  
The agent may need to inspect intents first, then count rows.

## Example retrieval query

```text
Show me 5 examples of the SHIPPING category.
```

Purpose:

Tests example retrieval by category.

## Unstructured summary query

```text
Summarize how agents respond to complaint intents.
```

Purpose:

Tests unstructured routing and summarization from dataset rows.

## Natural-language search query

```text
Show me examples of people wanting their money back.
```

Purpose:

Tests keyword search and semantic-style interpretation.

The user does not say "refund" directly, so the agent should connect "money back" to refund-like examples.

## Distribution query

```text
What is the distribution of intents in the ACCOUNT category?
```

Purpose:

Tests grouped structured analysis.

## Out-of-scope query

```text
What's the best CRM software for handling complaints?
```

Purpose:

Tests out-of-scope detection.

The query is related to customer service, but it is not answerable from the dataset.

## General knowledge out-of-scope query

```text
Who is the president of France?
```

Purpose:

Tests that the agent does not answer from general knowledge.

---

# 12. Important Notes

The agent should always answer from the dataset only.

If the dataset does not contain enough information, the agent should say so clearly.

The agent should not invent:

- categories
- intents
- counts
- examples
- external facts

The CLI should print reasoning steps, not only the final answer.

The MCP server should expose the dataset tools so external MCP clients can call them.

---

# 13. Quick Start Summary

```bash
git clone <your-repo-url>
cd bitext_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`, add your Nebius API key, and place the dataset here:

```text
data/bitext_customer_service.csv
```

Run the CLI:

```bash
python main.py
```

Run the MCP server:

```bash
python -m src.mcp_server
```