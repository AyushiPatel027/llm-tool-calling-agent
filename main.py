"""CLI entry point.

    python main.py                 interactive REPL
    python main.py "your question"  one question, then exit
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from agent import run_agent
from llm_client import get_llm

load_dotenv()  # read .env if present


def main() -> None:
    llm = get_llm()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\nQ: {question}")
        print(f"\nA: {run_agent(llm, question)}")
        return

    print("\nData-analysis agent. Ask about data/sales.csv. Ctrl+C or 'exit' to quit.")
    while True:
        try:
            question = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            continue
        print(f"\nagent > {run_agent(llm, question)}")


if __name__ == "__main__":
    main()
