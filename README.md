# LLM Tool-Calling Agent

A minimal data-analysis agent built around an LLM tool-calling loop. The model
is given a sales CSV and a set of pandas-backed tools, and it decides which
tools to call, and in what order, to answer a question in plain English.

## How the loop works

The model cannot touch the filesystem or run code. Each turn:

1. The conversation and the tool schemas are sent to the model.
2. The model replies with a final answer, or with one or more tool calls
   (e.g. `aggregate(group_by="region", metric="revenue")`).
3. The matching Python function runs locally and its result is appended to the
   conversation as a `tool` message.
4. Repeat until the model returns an answer, capped at `MAX_STEPS`.

The loop is in `agent.py` (~35 lines).

## Layout

| File | Responsibility |
|------|----------------|
| `agent.py` | The tool-calling loop |
| `tools.py` | Six tools (`load_csv`, `describe_columns`, `filter_rows`, `aggregate`, `compute_stat`, `make_chart`) and their JSON schemas |
| `llm_client.py` | OpenRouter client, plus an offline rule-based backend |
| `main.py` | CLI: one-shot question or interactive REPL |
| `data/sales.csv` | Sample dataset (20 rows) |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # add a free key from https://openrouter.ai/keys
```

Without a key the agent falls back to `MockLLM`, a rule-based backend that maps
question keywords to a single tool. It runs the same loop offline, which is
enough for tests and demos:

| Question mentions | Tool selected |
|-------------------|---------------|
| chart / plot / draw / pie / bar / line | `make_chart` |
| average / median / std / total + a column | `compute_stat` |
| by / per / each / compare / break down | `aggregate` |
| only / just / where + a region or category | `filter_rows` |
| anything else | `describe_columns` |

## Usage

```bash
python main.py "Which region has the highest total revenue?"
python main.py                      # interactive
```

Sample questions:

```
What is the average number of units sold?
Break down revenue by product.
Compare total units sold per category.
What is the median revenue, and the standard deviation?
Show only West-region rows and total their revenue.
Make a bar chart of total revenue by region.
Plot a line chart of revenue over time.
Draw a pie chart of each category's share of revenue.
```

Every tool call is printed as it happens:

```
--- step 1: model requested 1 tool call(s) ---
  -> load_csv({'path': 'data/sales.csv'})
  <- {"name": "data", "rows": 20, "columns": [...]}
--- step 2: model requested 1 tool call(s) ---
  -> aggregate({'group_by': 'region', 'metric': 'revenue', 'agg': 'sum'})
  <- {"result": {"North": 7782.3, "East": 7750.9, ...}}

A: North has the highest total revenue, about $7,782.
```

Charts are written to `charts/` as PNG files.

## Adding a tool

1. Write the function in `tools.py`.
2. Add its schema to `TOOL_SCHEMAS`.
3. Register it in `TOOL_FUNCTIONS`.
