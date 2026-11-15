"""Write dialogue essays to files, without the database, images or WeChat.

Each essay goes through the same steps as the admin console: a topic brief (unless one is
given), the draft, then the polish pass. Use it to compare text models or to try a change
to prompts/dialogue_essay.md before it reaches the scheduler.

    python scripts/write_dialogue_essays.py --env-file ../.env --provider minimax --count 3
    python scripts/write_dialogue_essays.py --provider deepseek \
        --topic "刻舟求剑｜分手一年，他又去了两个人常去的那家面馆" \
        --brief "写法：解惑体。靶子：还想回去就是还爱她。论点：想回去和还爱着是两回事。"
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

READERS = "恋爱中的男女，以女性读者为主，也覆盖暧昧、分手、复合与单身成长阶段"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", default="mock", help="mock, openai, deepseek, minimax, kimi, compatible …")
    parser.add_argument(
        "--models", default="",
        help="MiniMax only: models for topic,draft,polish, e.g. MiniMax-M2.7-highspeed,MiniMax-M3,MiniMax-M3.1-Flash."
        " A plan's rate limit may be counted per model, so spreading the steps can avoid waiting.",
    )
    parser.add_argument("--count", type=int, default=1, help="essays to write when no --topic is given")
    parser.add_argument("--topic", help="“典故名｜处境” or any subject; skips the topic step")
    parser.add_argument("--brief", default="", help="写法、靶子、论点、落点、人物 for --topic")
    parser.add_argument("--directions", help="what the topic step should choose from")
    parser.add_argument("--length", type=int, help="target length in characters")
    parser.add_argument("--no-polish", action="store_true", help="skip the polish pass")
    parser.add_argument("--pause", type=float, default=0, help="seconds to wait between essays")
    parser.add_argument(
        "--rate-limit-wait", type=float, default=300,
        help="seconds to wait before retrying a request the provider refused for rate limits (0: no retry)",
    )
    parser.add_argument("--rate-limit-max-wait", type=float, default=3600, help="give up a request after this long")
    parser.add_argument("--out", type=Path, default=Path("dialogue-essays"))
    parser.add_argument("--env-file", type=Path, help="KEY=VALUE file with provider keys")
    return parser.parse_args()


RATE_LIMIT_MARKS = ("2062", "rate_limit", "速率限制", "429")


class PatientProvider:
    """Retries a request the provider refused for rate limits instead of failing the essay.

    Subscription plans such as MiniMax Token Plan refuse the next long request for many
    minutes after a successful one; topic, draft and polish come one after another.
    """

    def __init__(self, provider, wait: float, max_wait: float):
        self._provider = provider
        self._wait = wait
        self._max_wait = max_wait

    def __getattr__(self, name):
        return getattr(self._provider, name)

    # The last answer, kept so a malformed one can be saved for diagnosis.
    last_answer = None

    def _patiently(self, call, *args, **kwargs):
        waited = 0.0
        while True:
            try:
                self.last_answer = None
                self.last_answer = call(*args, **kwargs)
                return self.last_answer
            except Exception as error:
                limited = any(mark in str(error) for mark in RATE_LIMIT_MARKS)
                if not limited or not self._wait or waited + self._wait > self._max_wait:
                    raise
                print(f"  {time.strftime('%H:%M:%S')} 被限流，{self._wait:.0f} 秒后重试", flush=True)
                time.sleep(self._wait)
                waited += self._wait

    def generate_json(self, *args, **kwargs):
        return self._patiently(self._provider.generate_json, *args, **kwargs)

    def generate_article(self, *args, **kwargs):
        return self._patiently(self._provider.generate_article, *args, **kwargs)


def load_environment(path: Path) -> None:
    # Settings read the environment when the app package is imported, so this runs first.
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    arguments = parse_arguments()
    if arguments.env_file:
        load_environment(arguments.env_file)

    from app.config import Settings
    from app.content_workflows import get_content_workflow
    from app.emotional_defaults import DIALOGUE_ESSAY_CATEGORY, DIALOGUE_ESSAY_DIRECTIONS
    from app.schemas import ArticleGenerateRequest
    from app.services import (
        dialogue_essay,
        generator,  # noqa: F401 - importing it registers the generation handlers
    )
    from app.services.llm import build_llm_provider
    from app.services.renderer import render_emotional_html

    workflow = get_content_workflow("emotional_article")
    def provider(model: str = ""):
        settings = Settings()
        if model:
            if arguments.provider != "minimax":
                raise SystemExit("--models only applies to --provider minimax")
            settings = replace(settings, minimax_model=model)
        return PatientProvider(
            build_llm_provider(settings, arguments.provider),
            arguments.rate_limit_wait,
            arguments.rate_limit_max_wait,
        )

    models = [name.strip() for name in arguments.models.split(",")] if arguments.models else []
    if models and len(models) != 3:
        raise SystemExit("--models needs three names: topic,draft,polish")
    topic_llm, llm, polish_llm = (provider(name) for name in models) if models else (provider(),) * 3
    audience = {
        "description": READERS,
        "article_length": arguments.length or dialogue_essay.DEFAULT_LENGTH,
        # Only the mock provider reads these two.
        "age_range": "18-40",
        "image_style": "无文字",
    }
    arguments.out.mkdir(parents=True, exist_ok=True)
    finished = sorted(arguments.out.glob("*.json"))
    # Earlier runs in the same folder count as recent, so batches keep moving through the library.
    recent = [json.loads(path.read_text(encoding="utf-8"))["topic"] for path in finished]

    for offset in range(1 if arguments.topic else arguments.count):
        if offset and arguments.pause:
            time.sleep(arguments.pause)
        tokens = 0
        if arguments.topic:
            topic, brief = arguments.topic, arguments.brief
        else:
            try:
                choice = topic_llm.generate_json(
                    workflow.build_topic_prompt(
                        directions=arguments.directions or DIALOGUE_ESSAY_DIRECTIONS,
                        category=DIALOGUE_ESSAY_CATEGORY,
                        audience=audience,
                        recent_topics=recent[-30:],
                        local_date=date.today().isoformat(),
                    ),
                    system_prompt=workflow.topic_system_prompt,
                )
            except Exception as error:  # noqa: BLE001 - one refused topic should not end the batch
                print(f"  选题失败，跳过这一篇：{error}", flush=True)
                continue
            topic, brief = str(choice["topic"]).strip(), str(choice.get("rationale", "")).strip()
            tokens += topic_llm.usage.total_tokens or 0
        request = ArticleGenerateRequest(
            content_type="emotional_article",
            category=DIALOGUE_ESSAY_CATEGORY,
            topic=topic[:300],
            source_prompt=f"选题的切入点：{brief}" if brief else "",
            audience_slug="relationship-readers",
            provider=arguments.provider,
            target_length=audience["article_length"],
            humanize=not arguments.no_polish,
        )
        print(f"  选题：{topic}", flush=True)
        try:
            article, runs, version = workflow.generate(
                session=None,
                llm=llm,
                request=request,
                effective_topic=request.topic,
                audience_context=audience,
                source_material="",
                polish_llm=polish_llm,
            )
        except Exception as error:  # noqa: BLE001 - keep the batch going, keep the evidence
            failed = arguments.out / f"{len(finished) + offset + 1:02d}-failed.json"
            failed.write_text(
                json.dumps(
                    {"topic": topic, "brief": brief, "error": str(error), "raw_answer": llm.last_answer},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"  这一篇失败：{error}；模型原始返回存在 {failed.name}", flush=True)
            continue
        recent.append(topic)
        tokens += sum(run.get("total_tokens") or 0 for run in runs)
        body = [
            article["opening"],
            *[paragraph for section in article["sections"] for paragraph in section["paragraphs"]],
            article["conclusion"],
        ]
        allusion = dialogue_essay.find_allusion(topic, brief)
        stem = arguments.out / f"{len(finished) + offset + 1:02d}-{allusion.name if allusion else 'essay'}"
        stem.with_suffix(".txt").write_text(
            article["selected_title"] + "\n\n" + "\n\n".join(body) + "\n", encoding="utf-8"
        )
        stem.with_suffix(".html").write_text(render_emotional_html(article, [], []), encoding="utf-8")
        stem.with_suffix(".json").write_text(
            json.dumps(
                {
                    "topic": topic,
                    "brief": brief,
                    "provider": llm.name,
                    "models": {"topic": topic_llm.model, "draft": llm.model, "polish": polish_llm.model},
                    "prompt_version": version,
                    "steps": [
                        {key: run[key] for key in ("step", "status", "error_message") if run.get(key)}
                        for run in runs
                    ],
                    "article": article,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        failed = [run["step"] for run in runs if run["status"] != "completed"]
        print(
            f"{stem.name}.txt  《{article['selected_title']}》  "
            f"{sum(len(paragraph) for paragraph in body)}字 {len(body)}段  {llm.model}  {tokens} tokens"
            + (f"  未完成：{'、'.join(failed)}" if failed else "")
        )


if __name__ == "__main__":
    main()
