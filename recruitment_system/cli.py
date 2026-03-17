from __future__ import annotations

import argparse

from dotenv import load_dotenv

from recruitment_system.api import RecruitmentResult, analyze_word

RESET = "\033[0m"
TITLE = "\033[1;36m"
TEXT = "\033[0;37m"


def format_result_colored(result: RecruitmentResult) -> str:
    blocks = [
        ("Perception Agent", result.perception_output),
        ("Action Agent", result.action_output),
        ("Affect Agent", result.affect_output),
        ("Context Agent", result.context_output),
        ("Recruitment Agent", result.recruitment_output),
        ("Simulation Agent", result.simulation_output),
    ]
    header = f"{TITLE}Слово: {result.word} | Контекст: {result.context}{RESET}"
    body = "\n\n".join(f"{TITLE}## {title}{RESET}\n{TEXT}{text}{RESET}" for title, text in blocks)
    return f"{header}\n\n{body}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recruitment Learning Simulator (routerai + xiaomi/mimo-v2-flash)."
    )
    parser.add_argument("--word", required=True, help="Новое слово для обучения")
    parser.add_argument("--context", required=True, help="Контекст освоения слова")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Запустить без API (детерминированный mock backend)",
    )

    args = parser.parse_args()
    load_dotenv()

    result = analyze_word(
        word=args.word,
        context=args.context,
        temperature=args.temperature,
        use_mock=args.mock,
    )
    print(format_result_colored(result))


if __name__ == "__main__":
    main()
