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

    def to_pretty_text(self) -> str:
        return (
            "=" * 72
            + f"\nRecruitment Learning Report\nСлово: {self.word}\nКонтекст: {self.context}\n"
            + "=" * 72
            + "\n\n"
            + "[1] Perception Agent\n"
            + f"{self.perception_output}\n\n"
            + "[2] Action Agent\n"
            + f"{self.action_output}\n\n"
            + "[3] Affect Agent\n"
            + f"{self.affect_output}\n\n"
            + "[4] Context Agent\n"
            + f"{self.context_output}\n\n"
            + "[5] Recruitment Agent\n"
            + f"{self.recruitment_output}\n\n"
            + "[6] Simulation Agent\n"
            + f"{self.simulation_output}"
        )


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
