---
trigger: always_on
---

Use ChromaDB as vector db. Docs are here: https://docs.trychroma.com/docs/overview/getting-started
Use Chromadb for vector and text search features.
Do not monkeypatch Chroma in tests.
To create embeddings use the local LMStudio server I have running in my environment.
The embedding model and provider are configured in the `EmbeddingModelConfig` table.