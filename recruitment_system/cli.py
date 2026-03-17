from __future__ import annotations

import argparse
import os
from typing import Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from recruitment_system.graph import AgentBackend, MockBackend, RecruitmentState, build_graph


class OpenAIBackend(AgentBackend):
    def __init__(self, model: str, temperature: float):
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def run(self, system_prompt: str, user_prompt: str) -> str:
        message = self.llm.invoke(
            [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
        )
        return message.content


def format_result(result: Dict[str, str]) -> str:
    blocks = [
        ("Perception Agent", result.get("perception_output", "")),
        ("Action Agent", result.get("action_output", "")),
        ("Affect Agent", result.get("affect_output", "")),
        ("Context Agent", result.get("context_output", "")),
        ("Recruitment Agent", result.get("recruitment_output", "")),
        ("Simulation Agent", result.get("simulation_output", "")),
    ]
    return "\n\n".join(f"## {title}\n{text}" for title, text in blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recruitment Learning Simulator (LangGraph multi-agent pipeline)."
    )
    parser.add_argument("--word", required=True, help="Новое слово для обучения")
    parser.add_argument("--context", required=True, help="Контекст освоения слова")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI/ChatGPT model")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Запустить без API (детерминированный mock backend)",
    )

    args = parser.parse_args()

    load_dotenv()

    if args.mock:
        backend: AgentBackend = MockBackend()
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY не найден. Добавьте его в окружение или используйте --mock."
            )
        backend = OpenAIBackend(model=args.model, temperature=args.temperature)

    app = build_graph(backend)
    initial_state: RecruitmentState = {"word": args.word, "context": args.context}
    result = app.invoke(initial_state)

    print(format_result(result))


if __name__ == "__main__":
    main()
