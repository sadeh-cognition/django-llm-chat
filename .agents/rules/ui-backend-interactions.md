---
trigger: always_on
---

All TUI interactions with data should be done via the backend HTTP API.
All business logic should be extracted into functions which can be used without the TUI.
When calling the backend API use the ninja schemas that define the endpoints incoming request type. Also, use the response schema to parse the response from the HTTP API.