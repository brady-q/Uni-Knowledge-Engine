# Uni Knowledge Engine

A study assistant that answers questions from a course's own material, and
shows exactly where each answer came from.

Students upload lecture notes, slides, handbooks and past papers as PDFs to a
course. The engine extracts the text page by page, splits it into chunks, and
indexes them for semantic search. When someone asks a question, such as "When
is the second assignment due?" or "How is the learning rate chosen?", it
retrieves the most relevant passages and has Claude write an answer grounded
only in them.

Every claim in the answer carries a citation like `[S1]`. Clicking it opens the
original PDF at the cited page, so the answer can be checked against the
source. If the course material doesn't cover the question, the engine says so
instead of guessing. Every question and answer is logged, so gaps in the
material are easy to spot.

## How citations stay honest

The model never sees a filename or page number. Retrieved chunks go into the
prompt labelled `[S1]…[S6]`, and the model cites only those markers. The backend
resolves each marker to its chunk row and reads the filename and pages from the
database. Invented markers are dropped and logged, and an answer whose markers
are all invented is discarded.

## Stack

FastAPI, async SQLAlchemy, PostgreSQL 17 + pgvector (HNSW), PyMuPDF,
Sentence-Transformers (`all-MiniLM-L6-v2`, 384-d), Claude, Next.js +
TypeScript, Docker Compose.

```
PDF → extract (per page) → chunk → embed → vector search → Claude → resolve markers → answer
```

The API is layered `routers → services → repositories → models`.
`ANTHROPIC_API_KEY` is only given to the API container.

## Quickstart

Requires Docker.

```bash
make up        # start db, api, web (creates .env from .env.example)
make migrate   # apply migrations
```

- Web: http://localhost:3000
- API docs: http://localhost:8000/docs
- Health: `/health` (liveness), `/health/ready` (db, pgvector, embedder, Claude)

Set `ANTHROPIC_API_KEY` in `.env` to enable answering. Without it, everything
else works and asking a question returns `503`. On first start, the embedding
model takes about a minute to download.

## API

All routes are under `/api/v1/courses/{course_id}`:

| Route                    | Purpose                                  |
| ------------------------ | ---------------------------------------- |
| `POST /documents`        | Upload a PDF (ingested in the background) |
| `GET /documents/{id}`    | Ingestion status: `pending` → `ready` / `failed` |
| `GET /search?q=`         | Vector search over the course's chunks   |
| `POST /questions`        | Ask a question, get a cited answer       |
| `GET /questions`         | Query log                                |

## Development

```bash
make test        # unit + integration (no network, no API spend)
make lint        # ruff + mypy --strict
make check       # lint + test + frontend typecheck
make test-slow   # loads the real embedding model
make test-live   # calls the real Claude API (costs money)
```

Unit tests need no database. Integration tests run real migrations against
Postgres + pgvector. The default suite uses fake embeddings and a fake model.
