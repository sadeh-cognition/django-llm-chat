---
trigger: always_on
---

For testing use the pytest-django package documented here: <https://pytest-django.readthedocs.io/en/latest/>
When creating fixtures that involve django ORM models use the model-bakery package documented here: <https://github.com/model-bakers/model_bakery>
When testing the Django admin use `curl` command instead of the browser agent.
When testing involves using LLMs use the "groq" provider and model name "llama-3.1-8b-instant".
Do not use ollama in tests.
When running tests use `uv run pytest ...`.
