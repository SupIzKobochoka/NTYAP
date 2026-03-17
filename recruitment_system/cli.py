from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from recruitment_system.api import RecruitmentLearningSystem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recruitment Learning Simulator (LangGraph multi-agent pipeline)."
    )
    parser.add_argument("--word", required=True, help="Новое слово для обучения")
    parser.add_argument("--context", required=True, help="Контекст освоения слова")
    parser.add_argument(
        "--model",
        default="xiaomi/mimo-v2-flash",
        help="Модель в RouterAI/OpenAI-compatible API",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument(
        "--api-base",
        default="https://routerai.ru/api/v1",
        help="Базовый URL OpenAI-compatible API",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Запустить без API (детерминированный mock backend)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Вывести результат в JSON для интеграций",
    )
    parser.add_argument(
        "--truncate-chars",
        type=int,
        default=1000,
        help="Максимальная длина вывода на агента (кроме финального)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Отключить ANSI-цвета в текстовом выводе",
    )

    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    system = RecruitmentLearningSystem(
        model=args.model,
        temperature=args.temperature,
        openai_api_base=args.api_base,
        api_key=api_key,
        use_mock=args.mock,
    )

    result = system.learn_word(args.word, args.context)
    if args.json:
        print(result.to_json(indent=2))
    else:
        print(
            result.to_pretty_text(
                truncate_chars=args.truncate_chars,
                use_color=not args.no_color,
            )
        )


if __name__ == "__main__":
    main()
