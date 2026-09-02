"""Data-analysis tools exposed to the model.

Each tool is a plain function plus a JSON schema in ``TOOL_SCHEMAS``. The model
selects tools from those schemas; ``agent.py`` runs the matching function and
feeds the return value back into the conversation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend, write PNGs without a display
import matplotlib.pyplot as plt
import pandas as pd

CHART_DIR = Path("charts")

# Loaded datasets, keyed by handle. Tools share state through this dict so a
# filter result can be referenced by a later call.
_DATAFRAMES: dict[str, pd.DataFrame] = {}


def _get_df(name: str) -> pd.DataFrame:
    if name not in _DATAFRAMES:
        raise ValueError(
            f"No dataset named {name!r}. Load one first with load_csv. "
            f"Available: {list(_DATAFRAMES) or 'none'}"
        )
    return _DATAFRAMES[name]


# Tool implementations. Each returns a small JSON-serialisable dict, since the
# return value is sent back to the model as text.
def load_csv(path: str, name: str = "data") -> dict:
    """Load a CSV file into memory so other tools can analyse it."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(csv_path)
    _DATAFRAMES[name] = df
    return {
        "name": name,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "preview": df.head(3).to_dict(orient="records"),
    }


def describe_columns(name: str = "data") -> dict:
    """Return summary statistics (count, mean, min, max, ...) for a dataset."""
    df = _get_df(name)
    return json.loads(df.describe(include="all").to_json())


def filter_rows(column: str, op: str, value, name: str = "data",
                save_as: str | None = None) -> dict:
    """Filter rows where `column op value`. op is one of ==, !=, >, >=, <, <=."""
    df = _get_df(name)
    if column not in df.columns:
        raise ValueError(f"Unknown column {column!r}. Have: {list(df.columns)}")
    ops = {
        "==": df[column] == value,
        "!=": df[column] != value,
        ">": df[column] > value,
        ">=": df[column] >= value,
        "<": df[column] < value,
        "<=": df[column] <= value,
    }
    if op not in ops:
        raise ValueError(f"Unsupported op {op!r}. Use one of {list(ops)}.")
    result = df[ops[op]]
    if save_as:
        _DATAFRAMES[save_as] = result
    return {
        "matched_rows": len(result),
        "saved_as": save_as,
        "sample": result.head(5).to_dict(orient="records"),
    }


def aggregate(group_by: str, metric: str, agg: str = "sum",
              name: str = "data") -> dict:
    """Group rows by `group_by` and aggregate `metric` (sum/mean/min/max/count)."""
    df = _get_df(name)
    for col in (group_by, metric):
        if col not in df.columns:
            raise ValueError(f"Unknown column {col!r}. Have: {list(df.columns)}")
    if agg not in {"sum", "mean", "min", "max", "count"}:
        raise ValueError(f"Unsupported agg {agg!r}.")
    grouped = getattr(df.groupby(group_by)[metric], agg)()
    grouped = grouped.sort_values(ascending=False)
    return {"group_by": group_by, "metric": metric, "agg": agg,
            "result": grouped.round(2).to_dict()}


def compute_stat(column: str, stat: str, name: str = "data") -> dict:
    """Compute a single statistic (sum/mean/median/std/min/max) for one column."""
    df = _get_df(name)
    if column not in df.columns:
        raise ValueError(f"Unknown column {column!r}. Have: {list(df.columns)}")
    series = df[column]
    if not hasattr(series, stat):
        raise ValueError(f"Unsupported stat {stat!r}.")
    return {"column": column, "stat": stat, "value": round(float(getattr(series, stat)()), 4)}


def make_chart(kind: str, x: str, y: str, name: str = "data",
               agg: str | None = None, title: str | None = None,
               filename: str | None = None) -> dict:
    """Draw a bar/line/scatter/pie chart from a dataset and save it as a PNG.

    If `agg` (sum/mean/min/max/count) is given, rows are grouped by `x` and
    `y` is aggregated first -- otherwise the raw rows are plotted.
    """
    df = _get_df(name)
    for col in (x, y):
        if col not in df.columns:
            raise ValueError(f"Unknown column {col!r}. Have: {list(df.columns)}")
    if kind not in {"bar", "line", "scatter", "pie"}:
        raise ValueError(f"Unsupported kind {kind!r}.")

    if agg:
        if agg not in {"sum", "mean", "min", "max", "count"}:
            raise ValueError(f"Unsupported agg {agg!r}.")
        plot_df = getattr(df.groupby(x)[y], agg)().reset_index()
    else:
        plot_df = df[[x, y]].copy()

    CHART_DIR.mkdir(exist_ok=True)
    out = CHART_DIR / (filename or f"{kind}_{y}_by_{x}.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if kind == "bar":
        ax.bar(plot_df[x].astype(str), plot_df[y])
        ax.tick_params(axis="x", rotation=45)
    elif kind == "line":
        ax.plot(plot_df[x].astype(str), plot_df[y], marker="o")
        ax.tick_params(axis="x", rotation=45)
    elif kind == "scatter":
        ax.scatter(plot_df[x], plot_df[y])
    elif kind == "pie":
        ax.pie(plot_df[y], labels=plot_df[x].astype(str), autopct="%1.1f%%")
    ax.set_title(title or f"{agg + ' ' if agg else ''}{y} by {x}")
    if kind != "pie":
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)

    return {"saved_to": str(out), "kind": kind, "points": len(plot_df)}


# OpenAI/OpenRouter tool schema. The description fields are the only guidance
# the model gets on when to call each tool, so they need to be specific.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from disk into memory. Must be called "
                           "before any analysis tool. Returns column names, dtypes "
                           "and a small preview.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the CSV file."},
                    "name": {"type": "string",
                             "description": "Handle to store the dataset under. Default 'data'."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_columns",
            "description": "Summary statistics for every column of a loaded dataset.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Dataset handle. Default 'data'."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_rows",
            "description": "Keep only rows where a column compares to a value. "
                           "Optionally save the filtered result under a new handle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "op": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<="]},
                    "value": {"description": "Value to compare against (string or number)."},
                    "name": {"type": "string", "description": "Source dataset handle. Default 'data'."},
                    "save_as": {"type": "string", "description": "Handle to save filtered rows under."},
                },
                "required": ["column", "op", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate",
            "description": "Group by a column and aggregate a numeric metric "
                           "(sum, mean, min, max, count). Returns a sorted mapping.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {"type": "string"},
                    "metric": {"type": "string"},
                    "agg": {"type": "string", "enum": ["sum", "mean", "min", "max", "count"]},
                    "name": {"type": "string", "description": "Dataset handle. Default 'data'."},
                },
                "required": ["group_by", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_stat",
            "description": "Compute one statistic for one numeric column "
                           "(sum, mean, median, std, min, max).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "stat": {"type": "string", "enum": ["sum", "mean", "median", "std", "min", "max"]},
                    "name": {"type": "string", "description": "Dataset handle. Default 'data'."},
                },
                "required": ["column", "stat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_chart",
            "description": "Render a chart (bar, line, scatter, pie) from a loaded "
                           "dataset and save it as a PNG in the charts/ folder. "
                           "Pass agg to group by x and aggregate y first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["bar", "line", "scatter", "pie"]},
                    "x": {"type": "string", "description": "Column for the x-axis / labels."},
                    "y": {"type": "string", "description": "Numeric column for the y-axis / values."},
                    "agg": {"type": "string", "enum": ["sum", "mean", "min", "max", "count"],
                            "description": "Optional: aggregate y grouped by x."},
                    "title": {"type": "string"},
                    "filename": {"type": "string", "description": "Optional output file name."},
                    "name": {"type": "string", "description": "Dataset handle. Default 'data'."},
                },
                "required": ["kind", "x", "y"],
            },
        },
    },
]

# Tool name -> implementation.
TOOL_FUNCTIONS = {
    "load_csv": load_csv,
    "describe_columns": describe_columns,
    "filter_rows": filter_rows,
    "aggregate": aggregate,
    "compute_stat": compute_stat,
    "make_chart": make_chart,
}
