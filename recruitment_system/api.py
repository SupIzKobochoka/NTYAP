from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict

from recruitment_system.graph import AgentBackend, MockBackend, RecruitmentState, build_graph


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

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def to_pretty_text(self, truncate_chars: int = 1000, use_color: bool = True) -> str:
        reset = "\033[0m" if use_color else ""
        palette = {
            "perception": "\033[96m",  # cyan
            "action": "\033[92m",  # green
            "affect": "\033[95m",  # magenta
            "context": "\033[94m",  # blue
            "recruitment": "\033[93m",  # yellow
            "simulation": "\033[91m",  # red
        }

        items = [
            ("[1] Perception Agent", self._truncate(self.perception_output, truncate_chars), palette["perception"]),
            ("[2] Action Agent", self._truncate(self.action_output, truncate_chars), palette["action"]),
            ("[3] Affect Agent", self._truncate(self.affect_output, truncate_chars), palette["affect"]),
            ("[4] Context Agent", self._truncate(self.context_output, truncate_chars), palette["context"]),
            ("[5] Recruitment Agent", self._truncate(self.recruitment_output, truncate_chars), palette["recruitment"]),
            # Финальный агент не обрезается по требованию.
            ("[6] Simulation Agent", self.simulation_output, palette["simulation"]),
        ]

        blocks = []
        for title, body, color in items:
            c = color if use_color else ""
            blocks.append(f"{c}{title}{reset}\n{body}")

        # По запросу выводим только блоки агентов, без промптов и служебной шапки.
        return "\n\n".join(blocks)


class OpenAIBackend(AgentBackend):
    def __init__(
        self,
        model: str = "xiaomi/mimo-v2-flash",
        temperature: float = 0.2,
        openai_api_base: str = "https://routerai.ru/api/v1",
        api_key: str | None = None,
    ):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Не найден langchain-openai. Установите зависимости: pip install -e ."
            ) from exc

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY не найден. Передайте api_key или задайте OPENAI_API_KEY."
            )

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_base=openai_api_base,
            api_key=api_key,
        )

    def run(self, system_prompt: str, user_prompt: str) -> str:
        message = self.llm.invoke(
            [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
        )
        return message.content


class RecruitmentLearningSystem:
    def __init__(
        self,
        model: str = "xiaomi/mimo-v2-flash",
        temperature: float = 0.2,
        openai_api_base: str = "https://routerai.ru/api/v1",
        api_key: str | None = None,
        use_mock: bool = False,
    ):
        if use_mock:
            backend: AgentBackend = MockBackend()
        else:
            backend = OpenAIBackend(
                model=model,
                temperature=temperature,
                openai_api_base=openai_api_base,
                api_key=api_key,
            )
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
