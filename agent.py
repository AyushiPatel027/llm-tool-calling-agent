"""Tool-calling loop.

The model is sent the conversation plus the available tool schemas. It responds
with either a final text answer or one or more tool calls. Tool calls are
executed locally and their results appended to the conversation as `tool`
messages, then the loop repeats until the model returns an answer.
"""

from __future__ import annotations

import json

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

SYSTEM_PROMPT = (
    "You are a data analyst assistant with tools to load and analyse a CSV "
    "dataset. Always call load_csv before other tools. Use tools for every "
    "number; do not guess. Give a short, clear answer once you have enough "
    "information."
)

MAX_STEPS = 8  # guard against a model that keeps requesting tools


def run_agent(llm, user_question: str, *, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    for step in range(1, MAX_STEPS + 1):
        assistant_msg = llm.chat(messages, TOOL_SCHEMAS)
        messages.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls")
        if not tool_calls:
            return assistant_msg.get("content") or "(no answer)"

        if verbose:
            print(f"\n--- step {step}: model requested {len(tool_calls)} tool call(s) ---")

        for call in tool_calls:
            name = call["function"]["name"]
            raw_args = call["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            if verbose:
                print(f"  -> {name}({args})")

            try:
                result = TOOL_FUNCTIONS[name](**args)
                payload = json.dumps(result, default=str)
            except Exception as exc:  # return the error so the model can retry
                payload = json.dumps({"error": str(exc)})

            if verbose:
                print(f"  <- {payload[:300]}")

            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", name),
                "name": name,
                "content": payload,
            })

    return "Stopped: reached the maximum number of tool-calling steps."
