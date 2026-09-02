"""Chat backends for the agent.

``OpenRouterLLM`` calls the OpenRouter chat-completions API (OpenAI-compatible).
``MockLLM`` is a rule-based stand-in that runs the same loop offline with no API
key, useful for tests and demos. ``get_llm`` picks one based on the environment.
"""

from __future__ import annotations

import json
import os

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterLLM:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        """Send the conversation + tool schemas, return the assistant message dict."""
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]


# Columns in data/sales.csv, plus everyday words that map to them.
_COLUMNS = ["date", "region", "product", "category", "units", "unit_price", "revenue"]
_SYNONYMS = {
    "price": "unit_price", "time": "date", "day": "date",
    "sales": "revenue", "income": "revenue", "earnings": "revenue",
    "quantity": "units", "count": "units", "amount": "units",
}


def _columns_in(text: str) -> list[str]:
    """Return dataset columns mentioned in `text`, in the order they appear."""
    found = []
    for word in text.replace("_", " ").split():
        w = word.strip(".,?!'\"").lower()
        if w.endswith("'s"):
            w = w[:-2]
        w = w.rstrip("s") if w.endswith("s") and w[:-1] in _COLUMNS else w
        col = _SYNONYMS.get(w, w if w in _COLUMNS else None)
        if col and col not in found:
            found.append(col)
    # "unit price" (two words) -> unit_price
    if "unit_price" not in found and "unit price" in text.lower():
        found.append("unit_price")
    return found


class MockLLM:
    """Rule-based backend for offline runs.

    Maps question keywords to a single analysis tool (chart -> make_chart,
    average/median -> compute_stat, "by"/"per" -> aggregate, region/category
    name -> filter_rows, otherwise describe_columns), then summarises the
    result. It only covers the phrasings below; the OpenRouter backend handles
    the rest.
    """

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        question = next((m["content"] for m in messages if m.get("role") == "user"), "")
        q = question.lower()
        tool_results = [m for m in messages if m.get("role") == "tool"]
        n = len(tool_results)

        if n == 0:  # load the data first
            return _tool_call("load_csv", {"path": "data/sales.csv", "name": "data"})
        if n == 1:  # then one analysis tool
            return self._plan_analysis(q)
        return {"role": "assistant", "content": self._summarise(q, tool_results[-1])}

    def _plan_analysis(self, q: str) -> dict:
        cols = _columns_in(q)
        stat_words = {"average": "mean", "avg": "mean", "mean": "mean",
                      "median": "median", "std": "std", "deviation": "std",
                      "total": "sum", "sum": "sum", "max": "max",
                      "maximum": "max", "min": "min", "minimum": "min"}
        agg = next((v for k, v in stat_words.items() if k in q), None)

        if any(w in q for w in ("chart", "plot", "graph", "draw", "pie", "bar chart", "line chart")):
            kind = ("pie" if "pie" in q else "line" if "line" in q or "over time" in q
                    else "scatter" if "scatter" in q else "bar")
            y = next((c for c in cols if c in ("revenue", "units", "unit_price")), "revenue")
            x = next((c for c in cols if c in ("region", "product", "category", "date")),
                     "date" if kind == "line" else "region")
            return _tool_call("make_chart", {"kind": kind, "x": x, "y": y,
                                             "agg": agg or "sum",
                                             "title": f"{y} by {x}"})

        # grouped aggregate: "by region", "per product", "compare categories"
        group_words = ("by ", " per ", "each ", "compare", "break down", "breakdown", "grouped")
        group_col = next((c for c in cols if c in ("region", "product", "category", "date")), None)
        if group_col and (any(w in q for w in group_words) or len(cols) >= 2):
            metric = next((c for c in cols if c in ("revenue", "units", "unit_price")), "revenue")
            return _tool_call("aggregate", {"group_by": group_col, "metric": metric,
                                            "agg": agg or "sum"})

        # filter before the plain statistic, so "total revenue for the West"
        # narrows the rows first
        regions = {"north": "North", "south": "South", "east": "East", "west": "West"}
        cats = {"widgets": "Widgets", "gadgets": "Gadgets"}
        if any(w in q for w in ("only", "just", "filter", "where", "for the", "in the")):
            for key, val in {**regions, **cats}.items():
                if key in q:
                    col = "region" if key in regions else "category"
                    return _tool_call("filter_rows", {"column": col, "op": "==",
                                                      "value": val, "save_as": "subset"})

        # single-column statistic
        num_col = next((c for c in cols if c in ("revenue", "units", "unit_price")), None)
        if agg and num_col:
            return _tool_call("compute_stat", {"column": num_col, "stat": agg})

        # bare filter: a region/category name with no statistic requested
        for key, val in {**regions, **cats}.items():
            if key in q:
                col = "region" if key in regions else "category"
                return _tool_call("filter_rows", {"column": col, "op": "==",
                                                  "value": val, "save_as": "subset"})

        return _tool_call("describe_columns", {"name": "data"})

    def _summarise(self, q: str, tool_msg: dict) -> str:
        try:
            data = json.loads(tool_msg["content"])
        except (json.JSONDecodeError, KeyError):
            return "[MockLLM] Could not read the tool result."
        name = tool_msg.get("name", "")

        if "error" in data:
            return f"[MockLLM] The tool reported an error: {data['error']}"
        if name == "make_chart":
            return (f"[MockLLM] Saved a {data['kind']} chart ({data['points']} points) "
                    f"to {data['saved_to']}.")
        if name == "aggregate":
            rows = data.get("result", {})
            top = next(iter(rows), None)
            lines = "\n".join(f"  {k}: {v}" for k, v in rows.items())
            return (f"[MockLLM] {data['agg']} of {data['metric']} by {data['group_by']}:\n"
                    f"{lines}\n  -> highest: {top}")
        if name == "compute_stat":
            return f"[MockLLM] {data['stat']} of {data['column']} = {data['value']}"
        if name == "filter_rows":
            return (f"[MockLLM] {data['matched_rows']} matching rows "
                    f"(saved as {data.get('saved_as')}).")
        if name == "describe_columns":
            return f"[MockLLM] Column summary:\n{json.dumps(data, indent=2)[:800]}"
        return f"[MockLLM] {json.dumps(data)[:800]}"


def _tool_call(name: str, args: dict) -> dict:
    """Build an assistant message that requests one tool call (OpenAI shape)."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": f"call_{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }


def get_llm():
    """Pick a backend based on the environment."""
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free").strip()
    if key:
        print(f"[llm] Using OpenRouter model: {model}")
        return OpenRouterLLM(key, model)
    print("[llm] No OPENROUTER_API_KEY found -- using offline MockLLM.")
    return MockLLM()
