# Recruitment Learning Simulator (NTYAP)

Мультиагентная система на **LangChain + LangGraph**, которая моделирует `recruitment learning`: новое слово не определяется через словарь, а связывается с уже существующими сенсорными, моторными, аффективными и контекстными схемами.

## Что изменено в текущей версии

- Вызовы LLM идут **без system-role сообщений** (единый текстовый prompt).
- CLI печатает результат в терминал **цветными блоками**.
- Поддерживается удобное использование как Python-модуля через функцию `analyze_word(...)`.
- Зафиксированы endpoint и модель:
  - `https://routerai.ru/api/v1`
  - `xiaomi/mimo-v2-flash`

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Настройка API

```bash
export OPENAI_API_KEY="your_api_key"
```

## CLI запуск

```bash
python -m recruitment_system \
  --word "мяч" \
  --context "ребенок увидел предмет, потрогал его и услышал слово"
```

Для офлайн-проверки структуры без API:

```bash
python -m recruitment_system \
  --word "справедливость" \
  --context "школьник обсуждает честность оценки на уроке" \
  --mock
```

## Использование как Python-модуль

```python
from recruitment_system import analyze_word

result = analyze_word(
    word="мяч",
    context="ребенок играет на площадке и учит слово",
    api_key="your_api_key",  # можно не передавать, если задан OPENAI_API_KEY
)

print(result.to_json())
print(result.recruitment_output)
```

## Архитектура файлов

- `recruitment_system/graph.py` — состояние и граф LangGraph + шаблоны prompt.
- `recruitment_system/api.py` — backend, объектная обертка и функция `analyze_word`.
- `recruitment_system/cli.py` — цветной CLI вывод.
- `recruitment_system/__main__.py` — запуск через `python -m recruitment_system`.
