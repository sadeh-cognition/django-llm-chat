---
trigger: always_on
---

Whenever you create a new endpoint or modify an existing one make sure the endpoint is tested functionally.
To test the endpoint use the TestClient of the django-ninja package.
For fixtures and test dependencies use pytest fixtures.
In tests that call the backend API use the ninja schemas that define the endpoints incoming request type. Also, use the response schema to parse the response from the HTTP API.
Do not use any mocks.
Do not monkeypatch anything.