from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict

from recruitment_system.graph import AgentBackend, MockBackend, RecruitmentState, build_graph

DEFAULT_MODEL = "xiaomi/mimo-v2-flash"
DEFAULT_BASE_URL = "https://routerai.ru/api/v1"


@dataclass
class RecruitmentResult:
    word: str
    context: str
    perception_output: str
    action_output: str
    affect_output: str
    context_output: str
    recruitment_output: str
    simulation_output: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class OpenAIBackend(AgentBackend):
    def __init__(self, api_key: str, temperature: float = 0.2):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Не найден langchain-openai. Установите зависимости: pip install -e ."
            ) from exc

        self.llm = ChatOpenAI(
            model=DEFAULT_MODEL,
            temperature=temperature,
            openai_api_base=DEFAULT_BASE_URL,
            api_key=api_key,
        )

    def run(self, prompt: str) -> str:
        message = self.llm.invoke(prompt)
        return message.content


class RecruitmentLearningSystem:
    def __init__(
        self,
        api_key: str | None = None,
        temperature: float = 0.2,
        use_mock: bool = False,
    ):
        if use_mock:
            backend: AgentBackend = MockBackend()
        else:
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not resolved_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY не найден. Передайте api_key или используйте use_mock=True."
                )
            backend = OpenAIBackend(api_key=resolved_api_key, temperature=temperature)

        self._app = build_graph(backend)

    def learn_word(self, word: str, context: str) -> RecruitmentResult:
        initial_state: RecruitmentState = {"word": word, "context": context}
        result = self._app.invoke(initial_state)
        return RecruitmentResult(
            word=word,
            context=context,
            perception_output=result.get("perception_output", ""),
            action_output=result.get("action_output", ""),
            affect_output=result.get("affect_output", ""),
            context_output=result.get("context_output", ""),
            recruitment_output=result.get("recruitment_output", ""),
            simulation_output=result.get("simulation_output", ""),
        )


def analyze_word(
    word: str,
    context: str,
    api_key: str | None = None,
    temperature: float = 0.2,
    use_mock: bool = False,
) -> RecruitmentResult:
    """Convenient function-style API for Python integration."""
    system = RecruitmentLearningSystem(
        api_key=api_key,
        temperature=temperature,
        use_mock=use_mock,
    )
    return system.learn_word(word=word, context=context)
