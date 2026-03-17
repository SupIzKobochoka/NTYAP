# Recruitment Learning Simulator (NTYAP)

Мультиагентная система на **LangChain + LangGraph**, которая моделирует `recruitment learning`: новое слово не задаётся через словарную дефиницию, а связывается с уже существующими сенсорными, моторными, аффективными и контекстными схемами.

## Что реализовано

Система использует 6 агентов:

1. **Perception Agent** — извлекает сенсорные признаки.
2. **Action Agent** — извлекает моторные схемы.
3. **Affect Agent** — извлекает эмоции/оценки/функции.
4. **Context Agent** — извлекает сценарии употребления.
5. **Recruitment Agent** — собирает единую концептуальную схему.
6. **Simulation Agent** — строит короткую ментальную симуляцию.

В `LangGraph` первые 4 агента запускаются параллельно, после чего их ответы агрегируются и передаются в `Recruitment Agent`, затем в `Simulation Agent`.

---

## API и модель по умолчанию

Система настроена под OpenAI-compatible endpoint:

- `openai_api_base='https://routerai.ru/api/v1'`
- `model='xiaomi/mimo-v2-flash'`

API-ключ берётся из `OPENAI_API_KEY`.

---

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Настройка ключа

```bash
export OPENAI_API_KEY="your_api_key"
```

или через `.env`:

```env
OPENAI_API_KEY=your_api_key
```

---

## Использование через Python-функции (рекомендуется)

```python
from recruitment_system import RecruitmentLearningSystem

system = RecruitmentLearningSystem(
    model="xiaomi/mimo-v2-flash",                    # можно не указывать, это default
    openai_api_base="https://routerai.ru/api/v1",   # можно не указывать, это default
)

result = system.learn_word(
    word="мяч",
    context="ребенок увидел предмет, потрогал его и услышал слово",
)

print(result.to_pretty_text())  # структурированный человекочитаемый отчёт
# print(result.to_json())       # структурированный JSON для интеграций
```

### Mock-режим (без API)

```python
from recruitment_system import RecruitmentLearningSystem

system = RecruitmentLearningSystem(use_mock=True)
result = system.learn_word("справедливость", "школьник обсуждает честность оценки")
print(result.to_pretty_text())
```

---

## CLI

### 1) Реальный запуск через RouterAI/OpenAI-compatible API

```bash
python -m recruitment_system \
  --word "мяч" \
  --context "ребенок увидел предмет, потрогал его и услышал слово"
```

### 2) С указанием модели/endpoint

```bash
python -m recruitment_system \
  --word "справедливость" \
  --context "школьник обсуждает честность оценки на уроке" \
  --model "xiaomi/mimo-v2-flash" \
  --api-base "https://routerai.ru/api/v1"
```

### 3) JSON-вывод

```bash
python -m recruitment_system \
  --word "мяч" \
  --context "ребенок увидел предмет, потрогал его и услышал слово" \
  --json
```

### 4) Локальный `--mock`

```bash
python -m recruitment_system \
  --word "справедливость" \
  --context "школьник обсуждает честность оценки на уроке" \
  --mock
```

---

## Формат вывода

Теперь вывод структурирован:

- единая шапка отчёта (`слово`, `контекст`)
- 6 нумерованных блоков агентов
- доступен JSON-формат для программного использования

То есть это уже не «поток текста», а стабильный формат для демонстрации и интеграций.

---

## Архитектура файлов

- `recruitment_system/graph.py` — граф LangGraph и промпты агентов.
- `recruitment_system/api.py` — Python API (`RecruitmentLearningSystem`, `RecruitmentResult`, backends).
- `recruitment_system/cli.py` — CLI-обёртка над API.
- `recruitment_system/__main__.py` — запуск через `python -m recruitment_system`.
- `pyproject.toml` — зависимости проекта.
