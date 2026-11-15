# wechat-cg

WeChat article generator: FastAPI, SQLAlchemy and Jinja admin pages in `wechat-content-service/`.
`README.zh-CN.md` describes the architecture; this file holds what is easy to get wrong.

## Checks

Run from `wechat-content-service/` with the repository's virtualenv (`../.venv`):

```bash
python -m pytest
python -m ruff check app tests migrations
```

CI runs both, plus `alembic upgrade head && alembic check`. Scripts under `scripts/` are not linted in CI; lint them by hand when you touch one.

## Things to know

- Prompts are product code. The files in `prompts/` are read at import and assembled in `app/services/prompts.py`, `humanizer.py` and `dialogue_essay.py`; tests pin their key phrases. See `wechat-content-service/prompts/README.md`.
- For relationship articles the category is the subject, except the form categories in `EMOTIONAL_FORM_CATEGORIES` (清醒短句, 对话长文), which change how the article is written. Only the plan or the person may pick a form; a topic model may not.
- Anything a generated article cites must come from a checked library (`relationship_psychology_sources.md`, `dialogue_essay_allusions.md`). Check an entry against its source before adding it, and say in the entry what is traditional or disputed.
- The repository is public. Reference essays and other people's text stay out of it; `prompts/samples/` is git-ignored for local few-shot samples.
- Tests use the mock provider. A real provider needs its key in the environment; when a proxy injects the key, give the variable a placeholder such as `MINIMAX_API_KEY=proxy-injected`.
- `scripts/write_dialogue_essays.py` runs topic, draft and polish without the database, and `scripts/essay_shape.py` compares an essay with a reference. Use both when changing `prompts/dialogue_essay.md`.
