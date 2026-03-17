from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class RecruitmentState(TypedDict, total=False):
    word: str
    context: str
    perception_output: str
    action_output: str
    affect_output: str
    context_output: str
    recruitment_output: str
    simulation_output: str


class AgentBackend:
    """Backend interface for agent calls without system-role prompts."""

    def run(self, prompt: str) -> str:
        raise NotImplementedError


class MockBackend(AgentBackend):
    """Deterministic fallback backend for local demos/tests without API."""

    def run(self, prompt: str) -> str:
        del prompt
        return "[MOCK MODE] Сформирован демонстрационный ответ без сетевого запроса."


PERCEPTION_PROMPT = """
Ты — Perception Agent в модели recruitment learning.

Твоя задача: для нового слова выделить сенсорные признаки объекта или явления.
Не давай словарное определение. Выделяй только такие признаки, которые можно воспринять:
- визуальные
- тактильные
- слуховые
- пространственные

Формат ответа:
1. Список сенсорных признаков
2. Краткое объяснение, почему они важны для узнавания объекта

Если слово абстрактное, выдели косвенные воспринимаемые признаки через типичные ситуации.
"""

ACTION_PROMPT = """
Ты — Action Agent в модели recruitment learning.

Твоя задача: выделить моторные схемы, связанные со словом.
Ищи:
- действия человека с объектом
- типичное движение самого объекта
- последовательности действий

Не объясняй слово через словарь.
Опиши его через действие и взаимодействие.

Формат ответа:
1. Список действий
2. 1-2 типичных сценария использования
"""

AFFECT_PROMPT = """
Ты — Affect Agent в модели recruitment learning.

Твоя задача: определить эмоциональные, оценочные и функциональные ассоциации слова.
Нужно ответить:
- какие эмоции вызывает слово
- какие социальные или ценностные оценки с ним связаны
- какую функцию оно обычно выполняет в опыте человека

Формат ответа:
1. Эмоции
2. Оценки
3. Функции
"""

CONTEXT_PROMPT = """
Ты — Context Agent в модели recruitment learning.

Твоя задача: выделить типичные ситуации, сцены и сценарии, в которых человек осваивает и использует слово.
Нужно показать:
- кто участвует в ситуации
- что происходит до, во время и после
- какой контекст делает слово понятным

Формат ответа:
1. 2-3 типичных сценария
2. Для каждого сценария: участники, действие, результат
"""

RECRUITMENT_PROMPT = """
Ты — Recruitment Agent.

Твоя задача: смоделировать recruitment learning.
Используй результаты других агентов и собери новое слово как набор уже существующих схем.

Тебе нужно:
1. Определить, какие признаки являются базовыми
2. Показать, какие старые схемы были "рекрутированы"
3. Сформировать единую концептуальную схему слова
4. Объяснить, что будет частично активироваться при восприятии этого слова в будущем

Не давай просто определение.
Покажи именно механизм связывания слова с уже существующими схемами.

Формат ответа:
- Рекрутированные схемы
- Центральное ядро концепта
- Периферийные признаки
- Итоговая схема активации
"""

SIMULATION_PROMPT = """
Ты — Simulation Agent.

Твоя задача: по готовой концептуальной схеме слова построить короткую ментальную симуляцию.
Нужно показать:
- какую сцену активирует слово
- какие действия и ожидания с ним связаны
- что система "предсказывает" о дальнейшем развитии ситуации

Формат ответа:
1. Активированная сцена
2. Ожидаемые действия
3. Ожидаемые последствия
"""


def _with_context(prompt: str, state: RecruitmentState) -> str:
    return f"{prompt.strip()}\n\nВходные данные:\nслово: {state['word']}\nконтекст: {state['context']}"


def build_graph(backend: AgentBackend):
    def perception_node(state: RecruitmentState):
        return {"perception_output": backend.run(_with_context(PERCEPTION_PROMPT, state))}

    def action_node(state: RecruitmentState):
        return {"action_output": backend.run(_with_context(ACTION_PROMPT, state))}

    def affect_node(state: RecruitmentState):
        return {"affect_output": backend.run(_with_context(AFFECT_PROMPT, state))}

    def context_node(state: RecruitmentState):
        return {"context_output": backend.run(_with_context(CONTEXT_PROMPT, state))}

    def recruitment_node(state: RecruitmentState):
        prompt = (
            f"{RECRUITMENT_PROMPT.strip()}\n\n"
            f"Входные данные:\n"
            f"слово: {state['word']}\n"
            f"сенсорные признаки: {state['perception_output']}\n"
            f"действия: {state['action_output']}\n"
            f"эмоции и функции: {state['affect_output']}\n"
            f"сценарии: {state['context_output']}"
        )
        return {"recruitment_output": backend.run(prompt)}

    def simulation_node(state: RecruitmentState):
        prompt = (
            f"{SIMULATION_PROMPT.strip()}\n\n"
            f"Входные данные:\n"
            f"слово: {state['word']}\n"
            f"концептуальная схема: {state['recruitment_output']}"
        )
        return {"simulation_output": backend.run(prompt)}

    graph = StateGraph(RecruitmentState)
    graph.add_node("perception", perception_node)
    graph.add_node("action", action_node)
    graph.add_node("affect", affect_node)
    graph.add_node("context", context_node)
    graph.add_node("recruitment", recruitment_node)
    graph.add_node("simulation", simulation_node)

    graph.add_edge(START, "perception")
    graph.add_edge(START, "action")
    graph.add_edge(START, "affect")
    graph.add_edge(START, "context")

    graph.add_edge("perception", "recruitment")
    graph.add_edge("action", "recruitment")
    graph.add_edge("affect", "recruitment")
    graph.add_edge("context", "recruitment")

    graph.add_edge("recruitment", "simulation")
    graph.add_edge("simulation", END)

    return graph.compile()
