---
trigger: always_on
---

This is a terminal based application i.e. a TUI.
The user interface should use the python `rich` package.
The backend is a Django HTTP API implemented using `django-ninja`.

When interacting with the backend, always use the HTTP API.
Do not use the database directly.
Always use the HTTP API for fetching, updating, or deleting data.

When writing Django code, try your hardest not to use Django signals.