# Recruitment Learning Simulator (NTYAP)

Мультиагентная система на **LangChain + LangGraph**, которая моделирует `recruitment learning`: новое слово не определяется через словарь, а связывается с уже существующими сенсорными, моторными, аффективными и контекстными схемами.

## Что реализовано

Система использует 6 агентов:

1. **Perception Agent** — извлекает сенсорные признаки.
2. **Action Agent** — извлекает моторные схемы.
3. **Affect Agent** — извлекает эмоции/оценки/функции.
4. **Context Agent** — извлекает сценарии употребления.
5. **Recruitment Agent** — объединяет всё в концептуальную схему (ядро + периферия + связи).
6. **Simulation Agent** — запускает «ментальную симуляцию» по схеме (сцена, действия, ожидания).

В `LangGraph` первые 4 агента запускаются параллельно, затем их выходы объединяются в `Recruitment Agent`, после чего вызывается `Simulation Agent`.

---

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Настройка API

Проект рассчитан на ChatGPT API через пакет `langchain-openai`.

```bash
export OPENAI_API_KEY="your_api_key"
```

Можно также использовать `.env` файл:

```env
OPENAI_API_KEY=your_api_key
```

---

## Запуск

### 1) Реальный запуск через ChatGPT API

```bash
python -m recruitment_system \
  --word "мяч" \
  --context "ребенок увидел предмет, потрогал его и услышал слово" \
  --model "gpt-4o-mini"
```

### 2) Локальный демонстрационный режим без API

```bash
python -m recruitment_system \
  --word "справедливость" \
  --context "школьник обсуждает честность оценки на уроке" \
  --mock
```

`--mock` полезен для проверки пайплайна, структуры графа и CLI без сетевых вызовов.

---

## Формат результата

На выходе печатаются блоки всех 6 агентов:

- Perception Agent
- Action Agent
- Affect Agent
- Context Agent
- Recruitment Agent
- Simulation Agent

Именно так можно показать, что система делает не «словарное определение», а **сборку значения через рекрутирование существующих схем**.

---

## Архитектура файлов

- `recruitment_system/graph.py` — описание состояния и графа LangGraph, промпты агентов.
- `recruitment_system/cli.py` — CLI, выбор backend (`OpenAIBackend` или `MockBackend`), запуск графа.
- `recruitment_system/__main__.py` — запуск через `python -m recruitment_system`.
- `pyproject.toml` — зависимости и параметры пакета.

---

## Идея для демонстрации на защите

Запустите два примера:

1. **Конкретное слово**: `мяч`.
2. **Абстрактное слово**: `справедливость`.

И сравните:

- для конкретного слова сильнее сенсомоторные признаки;
- для абстрактного — сценарии, социальные оценки и функции.

Это напрямую иллюстрирует recruitment learning из курса.
