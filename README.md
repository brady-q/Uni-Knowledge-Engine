# Uni Knowledge Engine

Retrieval-augmented question answering over university course PDFs, where every
claim in an answer is cited back to a specific document and page.

> **Status: M5 (cited answers) complete.** The backend is end to end: PDF
> upload, PyMuPDF extraction with page metadata, chunking,
> Sentence-Transformers embeddings and HNSW vector search over `/search`, and
> now Claude answers over `/questions` whose citations are resolved from the
> database and logged. The frontend is still the M1 status page. See
> [Roadmap](#roadmap).

---

## The core design idea

The property this project is built to guarantee is that **citations cannot be
hallucinated**. The language model never writes a page number.

Retrieved chunks are injected into the prompt labelled `[S1]…[S6]`, as text and
nothing else — no filename, no page number, no id. The model returns structured
JSON citing only those markers. The backend then resolves each marker back to
the chunk row it came from and reads the filename and page range **out of the
database**.

So a page number the model could get wrong was never in its context to begin
with, and the guarantee is a property of the interface rather than of the
model's obedience. Markers it invents anyway are deleted from the answer,
counted in the query log, and logged as warnings — and an answer that claimed
to be answering but whose every marker was invented is discarded rather than
served with an empty citation list, because a confident uncited claim is the
one thing this system exists not to produce.

Everything else in the architecture — the typed boundaries, the query log, the
planned eval harness — exists to make that claim testable rather than asserted.

## Architecture

```
┌────────────────┐        ┌──────────────────────────────┐        ┌──────────────┐
│  Next.js web   │  HTTP  │        FastAPI api           │  SQL   │  Postgres 17 │
│  (App Router)  │───────▶│  routers → services → repos  │───────▶│  + pgvector  │
└────────────────┘        └──────────────────────────────┘        └──────────────┘
                                        │
                                        │ Claude API (server-side only)
                                        ▼
```

Dependencies inside the API point one way only: **routers → services →
repositories → models**. Routers handle HTTP concerns (validation, status
codes, headers); services hold pipeline logic, own the transaction boundary and
are testable as plain Python; repositories own all SQL. Nothing below the
router layer imports FastAPI, which is why a service can be driven from a test,
a script or a background task. Embedding and LLM access sit behind narrow
`Protocol` interfaces so tests substitute deterministic fakes and never touch
the network.

The `ANTHROPIC_API_KEY` is only ever given to the `api` service. The `web`
service does not receive it, so it cannot reach the browser bundle.

## Quickstart

Requires Docker and Docker Compose. Nothing else.

```bash
cp .env.example .env     # or just run `make up`, which does this for you
make up                  # build and start db, api, web
make migrate             # apply migrations 0001 and 0002
```

| Service | URL                                  |
| ------- | ------------------------------------ |
| Web     | http://localhost:3000                |
| API     | http://localhost:8000                |
| Docs    | http://localhost:8000/docs           |

Verify the stack:

```bash
curl -s localhost:8000/health          | jq
curl -s localhost:8000/health/ready    | jq
make test                              # 279 tests, in-container, no network
make lint                              # ruff + mypy --strict
```

Answering additionally needs `ANTHROPIC_API_KEY` in `.env`. Without it
everything else still works: the API starts, search is served, `/health/ready`
says why answering is unavailable, and asking a question is a `503`.

Expected readiness output once the migrations have been applied and the
embedding model has finished loading:

```json
{
  "status": "ok",
  "checks": {
    "db": { "status": "ok", "detail": null },
    "pgvector": { "status": "ok", "detail": "installed 0.8.6" },
    "embedder": { "status": "ok", "detail": "loaded sentence-transformers/all-MiniLM-L6-v2" },
    "claude": { "status": "ok", "detail": "configured for claude-opus-5" }
  }
}
```

Without a key, `claude` instead reports
`"answering is not configured on this server: ANTHROPIC_API_KEY is not set"`
and readiness is **503**.

Before `make migrate`, `pgvector` instead reports
`"available 0.8.6, not yet installed"` — still `ok`, because the extension only
has to be *installable* until the first migration creates it.

On the very first `make up` the `embedder` check reports
`"... is still loading"` and readiness is **503** for a minute or so while the
model is downloaded into the `hf_cache` volume. `/health` is `200` throughout,
which is the whole reason the two endpoints are separate.

Create a course:

```bash
curl -sX POST localhost:8000/api/v1/courses \
  -H 'content-type: application/json' \
  -d '{"code":"CS229","title":"Machine Learning"}' | jq
```

Run `make help` for every available target.

### Host-side development (optional)

The container is the source of truth, but editors want a local interpreter:

```bash
make venv          # backend/.venv from python3.12
make migrate-local # apply migrations from the host against localhost Postgres
make test-local    # runs the suite from the host against localhost Postgres
```

`make revision` runs in the container as the host user, so the migration it
generates is owned by you and not by the image's non-root uid.

## Health endpoints

Two endpoints, because they answer different questions:

| Endpoint        | Question                          | Touches the database |
| --------------- | --------------------------------- | -------------------- |
| `/health`       | Is this process alive?            | **No**               |
| `/health/ready` | Should traffic be routed here?    | Yes                  |

`/health` deliberately checks nothing external. If liveness depended on
Postgres, a transient database outage would cause an orchestrator to kill a
perfectly healthy API container and make the incident worse.

`/health/ready` runs one probe per dependency and returns **503** if any fails,
with a per-component reason:

```json
{
  "status": "error",
  "checks": {
    "db": { "status": "error", "detail": "OperationalError: connection refused" },
    "pgvector": { "status": "ok", "detail": "installed 0.8.6" },
    "embedder": { "status": "ok", "detail": "loaded sentence-transformers/all-MiniLM-L6-v2" },
    "claude": { "status": "ok", "detail": "configured for claude-opus-5" }
  }
}
```

The probes convert failures into values instead of raising, so an unhealthy
dependency produces an informative 503 rather than an opaque 500. The
`pgvector` probe exists to catch running against a plain `postgres` image at
startup instead of when the first migration fails. The `embedder` probe
*inspects* the model rather than using it — a probe that triggered the load
would hang the first readiness check for as long as the weights take to read,
and one run on an interval would keep a model alive that nothing else wanted.
The `claude` probe checks configuration only, for a blunter version of the same
reason: a probe that asked the model anything would bill somebody every few
seconds for the privilege of being monitored. It therefore cannot tell a valid
key from an invalid one — only a present one from a missing one, which is the
failure people actually hit. A key that is present but rejected is reported by
the first question asked.

Both endpoints are unversioned: probes should not have to track the API
version. Feature routers mount under `/api/v1`.

## Data model

Migration `0001` creates the whole content schema. Three tables, related by
foreign key rather than by nesting:

```
courses                  documents                      chunks
─────────                ──────────                     ──────
id          uuid pk      id          uuid pk            id          uuid pk
code        unique  ◀────course_id   fk cascade    ◀────document_id fk cascade
title                    filename                       ordinal
description              content_hash                   page_start
created_at               byte_size                      page_end
updated_at               page_count  null               text
                         status      checked            embedding   vector(384) null
                         error       null
                         created_at
                         updated_at
```

The choices worth knowing about:

- **`courses.code` is UNIQUE, but the primary key is a UUID.** Course codes get
  renamed and reused; identity must never change. Codes are normalised to upper
  case at the API boundary, so `cs229` and `CS229` collide and no
  case-insensitive column type is needed.
- **Both foreign keys are `ON DELETE CASCADE`, declared in the database.** A
  chunk whose document is gone is a citation that can never be resolved.
  Deleting a course is one `DELETE`, not an application-level cleanup someone
  has to remember. The ORM relationships use `passive_deletes=True` so they
  trust the database rather than loading every child to delete it individually.
- **Foreign keys are not automatically indexed by Postgres**, so each one is
  covered deliberately: `uq_documents(course_id, content_hash)` and
  `uq_chunks(document_id, ordinal)` both lead with the foreign key column, which
  serves lookups by parent as well as enforcing uniqueness.
- **`chunks.embedding` is nullable**, even though ingestion writes the vector
  with the row. Embedding is a separate concern from the text: a chunk without
  a vector — written before the embedding stage existed, or waiting on a
  re-embed after a model change — is a valid state rather than corruption, and
  `NOT NULL` would make swapping models a data migration with no intermediate
  step. Retrieval excludes these rows explicitly.
- **`documents.status` is a checked `VARCHAR`, not a native `ENUM`.** Adding a
  pipeline stage later is then a CHECK constraint swap inside a normal
  transaction, instead of an `ALTER TYPE` that cannot always run in one.
- **Page numbers are constrained in the database** (`page_start >= 1`,
  `page_end >= page_start`). These are the values the citation guarantee rests
  on, so they are enforced rather than assumed from the chunker.
- **The HNSW index is built while the table is empty**, which is free; building
  it after ingestion would mean a long lock on a large table. Its opclass is
  `vector_cosine_ops` to match the `<=>` operator retrieval will use — an index
  built for the wrong distance function is silently never used.

Migration `0002` adds the query log. Two more tables, and the second one is
the citation guarantee written down:

```
questions                        citations
─────────                        ─────────
id              uuid pk          id          uuid pk
course_id       fk cascade  ◀────question_id fk cascade
question        text             chunk_id    fk cascade ──▶ chunks.id
answer          text             marker      int (1 = [S1])
answered        bool
retrieved       int              uq(question_id, marker)
dropped_markers int
model           null
input_tokens    null
output_tokens   null
latency_ms      int
created_at
```

- **`citations` stores a foreign key and a marker, and no copy of the filename
  or page range.** Those are read back through `chunk.document` every time a
  citation is rendered, so a citation cannot drift from the document it claims
  to come from — and an answer read out of the log a week later describes that
  document as it is now, not as it was.
- **The consequence is intended: deleting or re-ingesting a document removes
  the citations that pointed into it**, via `ON DELETE CASCADE`. A marker
  resolving to a chunk that no longer exists is exactly the unresolvable
  citation this schema exists to make impossible, so the row goes rather than
  dangles. The answer itself survives, listing nothing.
- **`citations.chunk_id` is indexed**, because that is the side the cascade is
  driven from: re-ingesting a document deletes every one of its chunks, and
  each delete has to find the citations referencing it. `uq(question_id,
  marker)` covers the other direction, and also enforces that a marker the
  model used twice is one citation.
- **`questions` has no `updated_at`.** It is append-only; a timestamp that can
  only ever lie about what happened is worse than no timestamp.
- **`questions.model` is nullable, and NULL is information.** It means no model
  was called — which happens when retrieval comes back empty, where there is
  nothing to cite and no prompt worth paying for. That is a different fact from
  a model that was called and answered nothing.
- **`dropped_markers` is a column, not just a log line.** "How often does the
  model invent a citation?" is the central question about this system, and by
  the time an answer is read the evidence has already been removed from it. A
  number in a column can be queried; a warning in yesterday's logs cannot.

## Course API

| Method   | Path                    | Success | Failures                    |
| -------- | ----------------------- | ------- | --------------------------- |
| `POST`   | `/api/v1/courses`       | `201` + `Location` | `409` duplicate code, `422` |
| `GET`    | `/api/v1/courses`       | `200`   | `422` bad paging            |
| `GET`    | `/api/v1/courses/{id}`  | `200`   | `404`, `422` bad id         |
| `PATCH`  | `/api/v1/courses/{id}`  | `200`   | `404`, `409`, `422`         |
| `DELETE` | `/api/v1/courses/{id}`  | `204`   | `404`                       |

`PATCH` is a true partial update: an absent key leaves the column alone, while
an explicit `null` clears a nullable one. `null` for `code` or `title` is a
`422`, because those columns are `NOT NULL` — the DTO refuses it rather than
letting the database do it.

List responses are paginated and carry the full count, so a client can render
"51–100 of 240" without a second request:

```json
{ "items": [ ... ], "total": 240, "limit": 50, "offset": 50 }
```

Results are ordered by `code`. An unordered `LIMIT`/`OFFSET` has no defined row
order, so paging through it can repeat or skip rows.

### Uniqueness is enforced by the database, not checked first

There is deliberately no "does this code already exist?" query before the
insert. Two concurrent requests can both pass that check before either commits.
The `UNIQUE` index is evaluated atomically, so the only correct order is to
attempt the write and translate the rejection:

```
IntegrityError → constraint name is uq_courses_code → ConflictError → 409
```

The constraint name is read from the driver error rather than matched against
the message text, so a table with two unique constraints can never report one
violation as the other. `tests/integration/test_courses_crud.py` fires five
identical creates concurrently and asserts on exactly one `201` and four `409`s.

Domain errors carry no HTTP knowledge: services raise `NotFoundError` or
`ConflictError` from `app/errors.py`, and handlers registered in `create_app`
map them to statuses. That is why no route handler contains a `try`/`except`.

## Ingestion pipeline

`POST /api/v1/courses/{course_id}/documents` takes a multipart PDF, stores it,
and returns **201** immediately with status `pending`. Extraction and chunking
run afterwards; the client polls the document to watch it progress.

```
upload  →  pending  →  extracting  →  chunking  →  embedding  →  ready
                              │            │            │
                              └────────────┴────────────┴──▶  failed
                                                              + error on the row
```

`ready` is the only terminal success state and it means extracted, chunked
**and** embedded — that is, searchable. A client that sees `ready` can search;
one that sees anything else knows why not.

| Method   | Path                                            | Success | Failures                           |
| -------- | ----------------------------------------------- | ------- | ---------------------------------- |
| `POST`   | `/api/v1/courses/{cid}/documents`               | `201` + `Location` | `404`, `409`, `413`, `415`, `422` |
| `GET`    | `/api/v1/courses/{cid}/documents`               | `200`   | `404`, `422`                       |
| `GET`    | `/api/v1/courses/{cid}/documents/{did}`         | `200`   | `404`, `422`                       |
| `GET`    | `/api/v1/courses/{cid}/documents/{did}/chunks`  | `200`   | `404`, `422`                       |
| `GET`    | `/api/v1/courses/{cid}/documents/{did}/file`    | `200` PDF | `404`                            |
| `DELETE` | `/api/v1/courses/{cid}/documents/{did}`         | `204`   | `404`                              |

```bash
CID=$(curl -sX POST localhost:8000/api/v1/courses \
  -H 'content-type: application/json' \
  -d '{"code":"CS229","title":"Machine Learning"}' | jq -r .id)

DID=$(curl -sX POST localhost:8000/api/v1/courses/$CID/documents \
  -F 'file=@lecture.pdf' | jq -r .id)

curl -s localhost:8000/api/v1/courses/$CID/documents/$DID | jq
# { "status": "ready", "page_count": 42, "error": null, ... }

curl -s "localhost:8000/api/v1/courses/$CID/documents/$DID/chunks?limit=2" | jq
```

Every route is nested under its course, so ownership is structural: there is no
`/documents/{id}` that could return a document belonging to a course the caller
did not name. A document requested through the wrong course is a `404`, for the
same reason a missing id is — answering differently would confirm that it
exists somewhere else.

### Extraction keeps the page number attached

Page numbers are the reason this stage exists at all. Text is never
concatenated into one blob and re-split later: the page each fragment came from
is attached the moment it is read and travels with it into the chunk row. The
answer path then *reads* a page number rather than computing one, which is what
makes the citation guarantee mechanical rather than aspirational.

Extraction uses PyMuPDF's **block** mode, not `get_text("text")`. Plain text
mode emits one newline per visual line and no blank lines at all, so a page
arrives as a single undifferentiated run and the chunker has no paragraph
boundaries to prefer. Blocks are MuPDF's own paragraph segmentation, and
`sort=True` puts them in reading order rather than in the order the content
stream happened to draw them.

Failures are named for the person who uploaded the file, because that message
is what ends up on `documents.error` and on their screen:

| Cause                       | `error`                                                       |
| --------------------------- | ------------------------------------------------------------- |
| Scanned / image-only PDF    | `no extractable text found; this looks like a scanned PDF...`  |
| Password protected          | `the PDF is password protected`                                |
| Corrupt or not really a PDF | `the file could not be read as a PDF`                          |
| Row exists, bytes do not    | `the stored file for this document is missing`                 |

An encrypted PDF is detected explicitly rather than left to present as "no
text", which would send someone looking for an OCR problem they do not have.

### Chunking is a pure function

`chunk_pages` takes a list of pages and returns a list of chunks. No database,
no filesystem, no configuration lookup — which is why every case in
`tests/unit/test_chunking.py` is a literal input and a literal expectation. The
component most likely to be retuned is the one whose behaviour can be pinned
down exactly.

- **Sizes are in characters, not tokens.** `CHUNK_SIZE = 1000` lands just under
  the embedding model's 256 word-piece window at roughly four characters per
  token. Measuring real tokens would mean loading a tokenizer here, coupling the
  chunker to a specific model and making these tests slow, for no gain in a
  bound that is approximate either way.
- **Text is broken down to sentences before anything is assembled.** Overlap is
  carried as whole units, so if a unit were as large as a chunk, chunks would be
  single-unit and there would be nothing to carry — the overlap would silently
  never happen. This was a real bug, caught by
  `test_consecutive_chunks_overlap`.
- **`CHUNK_OVERLAP = 150` characters** are repeated from the end of each chunk
  into the start of the next, so a passage straddling a boundary survives intact
  in one of the two. The carry never takes the whole chunk — the first segment
  is always left behind, which is what guarantees progress.
- **Chunks may span pages.** A paragraph continuing across a page break belongs
  in one passage; splitting it there would be an artefact of the PDF's layout.
  The chunk records the full range, so the citation names both pages.
- **A runt final chunk is folded into its predecessor**, but only if the merge
  fits in `CHUNK_SIZE`. Merging rather than dropping keeps the invariant that no
  extracted text is missing from the index; refusing to merge past the limit
  keeps a chunk from being silently truncated at embedding time. Both outcomes
  lose text — the rule picks whichever loses none.

Oversized input degrades through three fallbacks, each worse than the last and
each reached only when the one before cannot make the piece fit: sentence
boundaries → word boundaries → a hard cut. The last is for a base64 blob or a
table dumped without spaces, where there is no boundary left to respect.

### Two stores, no transaction across them

A row in Postgres and a file on disk cannot be committed together, so the order
is chosen to make the failure modes harmless:

- **The row is committed before the file is moved into place.** A duplicate
  upload is rejected by `uq_documents(course_id, content_hash)` without any
  bytes having been published, and the staged file is discarded.
- **The file is deleted after the row.** The worst outcome is an orphaned file,
  which wastes disk. The reverse order risks a live document whose bytes are
  gone — a citation that cannot be resolved.
- **Deleting a course reads its document ids before the cascade fires.**
  `ON DELETE CASCADE` does not reach the filesystem, and after the delete there
  is no record of which files to remove, so every PDF in that course would leak.

Uploads are streamed to a staging directory and moved with `os.replace`, which
is atomic within a filesystem. A crash mid-upload leaves a `.part` file, never
a truncated PDF that extraction would read as real.

Storage is deliberately **not** content-addressed. Hashing the bytes into the
path would let one file serve the same PDF in two courses, but then deleting a
document has to prove no other row still points at it — and getting that
reference count wrong either leaks files forever or deletes one that is still
cited. One file per document is a little more disk and no bookkeeping.

### Uploads are validated by their bytes

The declared `Content-Type` is not consulted. Browsers routinely send
`application/octet-stream` for a file picked off disk, so trusting the header
would reject valid uploads; and a header can be set to anything, so it would
also accept invalid ones. The `%PDF-` magic number is the only part of the
claim a client cannot get wrong for free.

The size limit is enforced **twice**: once against `Content-Length`, so an
oversized body is refused before Starlette spools it to disk, and again while
streaming, because that header is a client's claim and not a fact. A client
that lies fills at most `MAX_UPLOAD_BYTES` plus one read chunk before being cut
off.

### Ingestion has no caller to raise to

It runs after the response has been sent, so a failure has nowhere to
propagate. The pipeline catches everything and writes the reason onto the row;
a document that silently stopped halfway would be indistinguishable from one
still working. Unexpected exceptions are logged in full and reported to the
uploader as a generic message — a stack trace on a web page is noise to them
and detail we should not publish.

Status is committed as each stage *begins*, not once at the end, so progress is
visible while a long PDF is still processing and a crash leaves the document at
the stage that killed it rather than at `pending`.

Scheduling is a FastAPI dependency (`get_ingestion_scheduler`) rather than an
inline `background.add_task`, for the same reason `get_readiness_report` is
one: it is the seam a test overrides. Without it, "was ingestion scheduled?"
would not be observable, and a unit test of the upload contract would run the
real pipeline against the unreachable database the test app points at.

## Retrieval

Once a document is `ready`, every chunk has a 384-dimension unit vector and the
course is searchable.

| Method | Path                            | Success | Failures                        |
| ------ | ------------------------------- | ------- | ------------------------------- |
| `GET`  | `/api/v1/courses/{cid}/search`  | `200`   | `404`, `422`, `503`             |

```bash
curl -s "localhost:8000/api/v1/courses/$CID/search?q=how+big+a+step+does+the+optimiser+take&limit=3" | jq
```

```json
{
  "query": "how big a step does the optimiser take",
  "limit": 3,
  "items": [
    {
      "chunk_id": "6f1e…",
      "document_id": "b3c2…",
      "filename": "week2-optimisation.pdf",
      "ordinal": 14,
      "page_start": 7,
      "page_end": 8,
      "text": "The learning rate controls the size of each step…",
      "score": 0.71
    }
  ]
}
```

The query shares no words with the passage it retrieved. That is the entire
reason for the model: a keyword index scores those two strings at zero.

- **`score` is cosine similarity, not distance.** `1.0` is identical and `0.0`
  unrelated. Reporting the raw `<=>` distance would hand every consumer a
  number that gets *better* as it gets smaller, which is a trap.
- **Every field is read from the stored rows** — the chunk's recorded page
  range, the document's own filename. Nothing here is computed at query time,
  which is the same mechanism the answer path relies on.
- **The default `limit` is 6**, because that is how many passages the answer
  prompt injects as `[S1]…[S6]`. Search and generation retrieve identically, so
  the eval harness can measure one and predict the other.
- **There is no `total` and no `offset`.** Paging through a ranking is not a
  meaningful operation: page two of a nearest-neighbour search is precisely the
  part the ranking exists to exclude.
- **An empty `items` is a valid 200.** It means the course has no documents,
  its documents are still embedding, or nothing in them is relevant. A missing
  course is a `404` instead — which is why the service looks the course up
  before embedding anything, rather than letting both cases return `[]`.

### Embeddings

`all-MiniLM-L6-v2`, 384 dimensions, run locally on CPU. Chosen over an
embedding API because the corpus is embedded once at ingestion and the query
side has to be fast and free; a network round trip and a per-token bill on the
hot path of a portfolio demo buys nothing. 384 dimensions is also a third of
the index size of a 1024-dimension model, on a quality difference that does not
show up on lecture notes.

The model is a process-wide singleton behind `Protocol Embedder`:

- **Loaded lazily, in a worker thread, exactly once.** Startup fires the load
  and does not wait for it, so `/health` answers immediately while the weights
  are still being read. Concurrent first requests are serialised on a lock —
  without it, five simultaneous requests read five copies of the same 90 MB for
  a result they all share.
- **A failed load is recorded, not raised.** Raising in the lifespan would take
  down a boot that is otherwise fine. The reason lands on `/health/ready`, and
  the next call that needs the model tries again, because a torn download is
  usually transient.
- **Encoding always runs off the event loop.** It is synchronous, CPU-bound
  torch; inline it would stall every other request for the length of the batch.
- **The model's width is checked against `EMBEDDING_DIMENSIONS` at load.** A
  mismatch caught here is one clear error; caught by Postgres it is an
  `INSERT` rejected once per chunk per document, reported as an unexpected
  ingestion failure with the cause nowhere in sight.
- **Vectors are normalised at creation**, which makes cosine distance and the
  dot product the same number and is what `vector_cosine_ops` assumes.

`EmbeddingError` maps to **503**, not 500: still loading, or a download that
failed, is a state that fixes itself, and the client should be told to retry
rather than that something is broken. On the ingestion path there is no status
code to return, so the pipeline converts it into a reason on the document row.

### The vector query

```sql
SET LOCAL hnsw.ef_search = 100;

SELECT chunks.*, documents.*, chunks.embedding <=> :query AS distance
FROM chunks JOIN documents ON documents.id = chunks.document_id
WHERE documents.course_id = :course_id AND chunks.embedding IS NOT NULL
ORDER BY chunks.embedding <=> :query
LIMIT :limit;
```

- **`ORDER BY` repeats the expression** rather than referring to the selected
  label. Only the indexable form makes the HNSW scan a candidate at all.
- **`ef_search` is raised from its default of 40.** That default assumes an
  unfiltered top-k; this query filters by course, and pgvector applies the
  filter *after* the index has picked its candidates — so a course whose chunks
  are not in the global top 40 would come back short, or empty. Over-fetching
  is the cheap fix at this scale; pgvector 0.8's `hnsw.iterative_scan` is what
  to reach for when it stops being enough. `SET LOCAL` scopes it to the
  transaction, so it cannot leak into another request on the same connection.
- **`IS NOT NULL` is explicit.** pgvector would sort a NULL last anyway, but a
  row that has not been embedded is not "distant" — including it in a result
  set that claims to be ranked by relevance is a lie, and with a small `limit`
  it pushes out something that is actually relevant.
- **The course scope is a join, not an optional filter.** A chunk from an
  unrelated unit in an answer's citations is a correctness failure, not a
  ranking one, so there is no cross-course search endpoint to forget to scope.

## Answers

The generation half. Retrieve, generate, resolve, log.

| Method | Path                                     | Success | Failures                                |
| ------ | ---------------------------------------- | ------- | --------------------------------------- |
| `POST` | `/api/v1/courses/{cid}/questions`        | `201` + `Location` | `404`, `409`, `422`, `502`, `503` |
| `GET`  | `/api/v1/courses/{cid}/questions`        | `200`   | `404`, `422`                            |
| `GET`  | `/api/v1/courses/{cid}/questions/{qid}`  | `200`   | `404`, `422`                            |

```bash
curl -sX POST "localhost:8000/api/v1/courses/$CID/questions" \
  -H 'content-type: application/json' \
  -d '{"question":"When is the second assignment due?"}' | jq
```

```json
{
  "id": "9a4c…",
  "course_id": "b3c2…",
  "question": "When is the second assignment due?",
  "answer": "The second assignment is due at the end of week six, and is submitted through the course portal rather than by email [S1]. Work handed in late loses ten percent a day, to a maximum of three days [S3].",
  "answered": true,
  "citations": [
    {
      "marker": "S1",
      "chunk_id": "6f1e…",
      "document_id": "b3c2…",
      "filename": "unit-handbook.pdf",
      "page_start": 4,
      "page_end": 4,
      "text": "The second assignment is due at the end of week six and is submitted…"
    },
    {
      "marker": "S3",
      "chunk_id": "c07a…",
      "document_id": "b3c2…",
      "filename": "unit-handbook.pdf",
      "page_start": 5,
      "page_end": 5,
      "text": "Work handed in after the week six deadline loses ten percent per day…"
    }
  ],
  "retrieved": 6,
  "dropped_markers": 0,
  "model": "claude-opus-5",
  "input_tokens": 1284,
  "output_tokens": 96,
  "latency_ms": 3120
}
```

A client renders footnotes by finding `[S1]` in the prose and looking up the
citation carrying that `marker`. `document_id` and `page_start` are what `M6`'s
PDF pane will jump to.

- **`POST`, not `GET`, unlike `/search`.** Answering writes a log row and
  spends money at a third party, and neither is something a browser should be
  free to repeat when somebody presses back.
- **Markers are ordered by retrieval rank**, because passage *n* is always the
  *n*-th ranked chunk. So the citation list is ordered by relevance without
  anything sorting it, and it does not depend on where in the prose the model
  happened to mention things first.
- **`answered: false` is a `201`, not an error.** A course that does not cover
  something is a real answer, and the log row saying "somebody asked this and
  we had nothing" is the most useful line in the whole table. `answered: true`
  always implies at least one citation.
- **A citation carries no `score`.** How close a passage was to the query is a
  retrieval diagnostic, which `/search` reports; a citation read back out of
  the log has none, and a field present on fresh answers and absent on stored
  ones is worse than no field at all.

### What the model is allowed to see

`app/llm.py` exposes one method, and its signature is the design:

```python
async def generate(question: str, passages: Sequence[str]) -> AnswerDraft
```

`passages` is a sequence of **strings**. Not chunks, not rows, not anything
carrying a filename or a page number. The prompt lays them out under opaque
markers and states which markers exist:

```
<passages>
[S1]
The second assignment is due at the end of week six…

[S2]
…
</passages>

The markers that exist are: [S1], [S2], …. There are no others.

Question: When is the second assignment due?
```

The model replies with JSON constrained by `output_config.format` — `{"answer":
"… [S1] …", "answered": true}` — so there is no preamble to strip and no "sure,
here is your answer" to parse around. What a schema cannot do is make the
*markers* trustworthy: it constrains shape, not content, and a model shown six
passages can still write `[S9]`. Dropping those is the resolver's job, and it
is not optional.

### Resolution, and what it refuses to serve

`resolve_markers` scans the prose, and for each `[S#]`:

| Marker | Outcome |
| ------ | ------- |
| Names a retrieved passage | Becomes a citation built from the stored chunk and document rows |
| Names the same passage twice | One citation; both mentions stay in the prose, where the claims are |
| Names nothing (`[S9]`, `[S0]`) | Deleted from the answer, counted in `dropped_markers`, logged as a warning |

Deleting a marker takes the space before it, so `"as shown [S9]."` becomes
`"as shown."` rather than `"as shown ."`.

Then the strict part. If the model set `answered: true` but **no** marker
survived, the prose is thrown away and replaced with a plain statement that the
retrieved material does not support an answer. A confident paragraph with all
of its support stripped out is precisely what this project promises not to
produce, and serving it because the wording sounded authoritative would be the
whole guarantee undone by a fallback.

### The model call

- **`claude-opus-5`, at `effort: "low"`.** The task is small and
  well-specified — read six passages, answer from them, mark what came from
  where — and it is bounded by what the passages say rather than by how long
  the model thinks about them. The budget belongs in better retrieval. Raise
  `ANSWER_EFFORT` if answers come back shallow.
- **Adaptive thinking is left on** rather than disabled. Turning it off is
  documented to make the model occasionally leak internal tags into the visible
  text, and that text is what the marker resolver parses. Lowering effort is
  the cheaper dial and does not have that failure mode.
- **The database connection is released before the call.** Generation takes
  seconds; holding a pooled connection across it would let a handful of
  concurrent questions exhaust the pool while doing no database work at all.
  The read transaction is committed rather than rolled back, because
  `expire_on_commit=False` leaves the loaded rows usable while a rollback would
  expire them and turn the next attribute read into blocking IO inside an async
  request.
- **One retry, 60-second timeout.** The SDK defaults to two retries and ten
  minutes, which is right for a batch job and wrong for a request a browser is
  waiting on.

### Failure modes

| Cause                                            | Status | What the caller is told                        |
| ------------------------------------------------ | ------ | ---------------------------------------------- |
| No such course                                   | `404`  | `no course with id …`                          |
| Documents changed mid-answer (a cited chunk went) | `409`  | `… ask again`                                  |
| Model refused, or the answer hit the token ceiling | `502` | `the model declined …` / `try a narrower question` |
| `ANTHROPIC_API_KEY` unset or rejected            | `503`  | `answering is not configured …`                |
| Rate limited, overloaded, unreachable, timed out | `503`  | `the answer service is busy; try again shortly` |
| Embedding model still loading                    | `503`  | `the embedding model is not available yet …`   |

`502` rather than `503` for a refusal or a truncation is the distinction worth
having: the upstream was reachable and did reply, so "unavailable, try later"
would be a lie, and the advice that helps is to ask something different. A
failed call writes **no** log row — the query log is a record of answers given,
and a third-party outage is an incident, not an answer.

## Project layout

```
├── docker-compose.yml       # db (pgvector/pg17), api, web
├── Makefile                 # every workflow: up, test, lint, migrate, ...
├── .env.example             # the only committed env file
├── backend/
│   ├── Dockerfile           # python:3.12-slim, non-root
│   ├── requirements.txt     # runtime deps, exactly pinned
│   ├── requirements-dev.txt # test/lint deps
│   ├── pyproject.toml       # ruff + mypy + pytest config only
│   ├── alembic.ini
│   ├── migrations/
│   │   ├── env.py           # async; target_metadata + pgvector rendering
│   │   └── versions/        # 0001 content schema, 0002 question log
│   ├── app/
│   │   ├── main.py          # create_app factory, lifespan, error handlers
│   │   ├── config.py        # pydantic-settings
│   │   ├── db.py            # async engine, session dependency
│   │   ├── models.py        # ORM models = the source of truth for all DDL
│   │   ├── schemas.py       # Pydantic DTOs (separate from the models)
│   │   ├── errors.py        # domain errors; no HTTP, no FastAPI
│   │   ├── storage.py       # on-disk PDF store; staging + atomic rename
│   │   ├── embedding.py     # the model, behind a Protocol; one per process
│   │   ├── llm.py           # the prompt, the schema, the Claude client
│   │   ├── routers/         # health, courses, documents, search, answers
│   │   ├── services/        # health, courses, documents, extraction,
│   │   │                    #   chunking, ingestion, retrieval, answers
│   │   └── repositories/    # every SQL statement in the application
│   └── tests/
│       ├── pdfs.py          # PDFs synthesised in-process, not binary fixtures
│       ├── embeddings.py    # deterministic stand-in for the embedder
│       ├── answers.py       # deterministic stand-in for Claude
│       ├── unit/            # no database, no network
│       ├── integration/     # real Postgres; auto-skips when unreachable
│       └── slow/            # the real models; excluded from the default run
└── frontend/
    ├── Dockerfile           # next dev against the bind mount
    └── src/
        ├── app/             # App Router: layout, status page
        └── lib/             # typed API client + DTO mirrors
```

## Testing

```bash
make test        # in-container: unit + integration, 279 tests, ~14s
make test-slow   # the six tests that load the real embedding model
make test-live   # the three that call the real Claude API (spends money)
make test-local  # host venv; integration tests skip if no local Postgres
make check       # lint + test + frontend typecheck (what CI would run)
```

207 unit tests, 72 integration, plus 6 behind the `slow` marker and 3 behind
`live`. Deliberate choices:

- **Unit tests point at `127.0.0.1:1`**, a port nothing can be listening on.
  This makes it impossible for a unit test to quietly depend on a live
  database, and it means the readiness *failure* path is covered end-to-end
  with no infrastructure at all.
- **`httpx.ASGITransport` with a manual `lifespan_context`** drives the app
  in-process — no live server to start, but startup logic (engine creation,
  storage directory) is still exercised.
- **Integration tests run against a dedicated `uke_test` database**, created
  and migrated on demand next to the development one. They truncate tables
  between cases, and doing that to the database someone is developing against
  would be hostile. Migrations are applied by shelling out to the real
  `alembic upgrade head`, so the suite proves the *migration* works and not
  merely that the models agree with themselves.
- **`alembic check` is a test.** `test_schema.py` fails if the models, the
  migration and the live schema disagree, which is what keeps `make revision`
  trustworthy for the next change instead of a source of surprise diffs. The
  other schema tests query the Postgres catalogue or attempt an illegal write,
  rather than trusting the ORM's own picture of the world.
- **PDFs are synthesised in-process** by `tests/pdfs.py`, not committed as
  binary fixtures. A checked-in file is opaque — nobody can tell from the
  repository what text a test expects to get back out — whereas these are
  readable, parameterisable, and cannot drift from the assertions using them.
  Building them costs no measurable time and no network, so they stay in the
  default suite.
- **Ingestion is observable in tests without polling.** Starlette runs
  background tasks before the ASGI call completes, so by the time
  `await client.post(...)` returns the pipeline has already finished. That is a
  property of the in-process transport, not of production, where a client polls.
- **Markers `integration`, `slow`, `live`** separate tests by what they cost,
  and the filter is in `addopts` rather than in a contributor's memory: a suite
  that downloads models and spends money on API calls is a suite people stop
  running. `slow` is the real embedding model; `live` is the real Claude API.
- **The default suite cannot reach the Claude API even by mistake.** Answers
  come from `tests/answers.py`, whose whole value is that a test chooses the
  markers "the model" writes — including markers that name nothing, which is
  the case the design exists for and which a real model would not produce on
  request. Behind that, an autouse fixture *removes* `ANTHROPIC_API_KEY` from
  the environment for every test not marked `live`, so a forgotten dependency
  override raises before any HTTP happens rather than quietly billing
  somebody. The embedding guard protects a cold cache; this one protects a
  credit card.
- **The default suite never loads the real model.** Vectors come from
  `tests/embeddings.py` — hashed bags of words, normalised to unit length — so
  the integration tests exercise pgvector, the HNSW opclass and the retrieval
  SQL for real, with a ranking a reader of the test can work out. An autouse
  fixture forces `EMBEDDING_PRELOAD=false` for the whole session, so a future
  test that builds its own `Settings` cannot start a download by accident.
- **What the fake cannot prove is what `tests/slow` is for.** A bag of words
  cannot tell us whether "step size" lands near "learning rate", and that
  inequality is the entire premise of retrieving by vector rather than by
  keyword. If it broke, every fake-backed test would stay green and the product
  would not work — so it is asserted directly, along with the model's width
  against the `vector(384)` column it has to fit.
- **`tests/slow/test_real_claude.py` is the same argument for the prompt.** A
  fake cannot tell us whether a real model, shown six labelled passages, cites
  the one it used, invents no marker, and declines rather than answering from
  what it already knows. Its passages describe a unit that does not exist, so
  an answer containing those facts can only have read them. Two calls, a few
  cents, and it skips without a key.

## Decisions and tradeoffs

**Python 3.12, not the host's 3.14.** `torch`/`sentence-transformers` cp314
wheels are unreliable and fall back to multi-hour source builds. Committing to
3.12 now avoids a painful re-pin when embeddings land.

**`requirements*.txt` with `==` pins, not `pyproject` dependencies.** Neither
`uv` nor `poetry` is available here, and pip has no lockfile; exact pins are the
closest substitute. Splitting runtime from dev deps also gives real Docker
layer caching. `pyproject.toml` holds tool configuration only, because this is
an application, not a distributable package.

**No module-level `app` object.** The API is served with
`uvicorn app.main:create_app --factory`. A module-level `app = create_app()`
would mean that merely *importing* `app.main` reads the environment, so tests
and Alembic would need a configured `DATABASE_URL` just to import it.

**Startup does not require the database.** The lifespan creates the engine but
never connects. An API that boots and honestly reports its own unreadiness is
more useful than one that crash-loops while Postgres is still starting.

**Alembic owns all DDL.** `Base.metadata.create_all` is never used, in tests or
anywhere else — the test database is built by running the real migration. The
URL comes from application settings rather than `alembic.ini`, so the app and
its migrations cannot disagree about which database they mean, and no
credentials sit in a committed file.

**Autogenerate is a draft, not an answer.** It reliably detects added tables
and columns, and reliably misses `CREATE EXTENSION`, data migrations and
renames — it sees a rename as a drop plus an add, which silently destroys data.
Migration `0001` was generated and then hand-edited to put
`CREATE EXTENSION IF NOT EXISTS vector` before the table that uses the type.
`make revision` prints this reminder.

**Constraints are named by convention, not by Postgres.** Left to itself
Postgres invents names like `courses_code_key`, and those names are what a
later `ALTER` or a `downgrade` has to reference. A `MetaData`
naming convention means every constraint name is derivable from `models.py` —
which is also what lets the 409 mapping match on `uq_courses_code` exactly.

**UUID primary keys, not `bigserial`.** These ids appear in frontend routes and
in citation payloads, where a sequential integer leaks how many rows exist and
invites enumeration. The cost is 16 bytes and unordered index inserts, which is
irrelevant at this scale. They are generated in Python *and* defaulted
server-side, so data migrations and hand-written SQL do not have to invent one.

**Invariants live in the schema; services handle the violation.** A `UNIQUE`
index is atomic and unconditional; a check-then-insert in Python is a race two
concurrent requests can both win. See
[Course API](#uniqueness-is-enforced-by-the-database-not-checked-first).

**`eager_defaults=True` on the mapper.** Server-generated timestamps are
fetched back with `RETURNING` rather than left expired, so a router can
serialise a just-committed row without a second `SELECT` — which, paired with
`expire_on_commit=False`, is what keeps a write to one round trip.

**Readiness reports are a FastAPI dependency.** `get_readiness_report` is
overridable, which is what lets the endpoint's contract be tested without a
database and lets an `embedder` check be added in exactly one place later.

**No UI framework yet.** The status page is one CSS file. Choosing a design
system before there are screens is a bet with no payoff; the real UI arrives
with the course and Q&A views.

**Two frontend base URLs.** Server Components resolve the API by its compose
service name (`http://api:8000`); the browser can only reach the published port
(`http://localhost:8000`). Collapsing these into one variable breaks under
Docker.

**`ready` means searchable, not merely stored.** Embedding is the last stage
before it, and chunks are inserted with their vectors already attached rather
than written empty and updated row by row — one `INSERT` per document instead
of one plus N, and no window in which a document is visible as a set of chunks
that retrieval would skip. Parking a document in a non-terminal state instead
would be more literal and less useful: a UI cannot render it and a client
cannot stop polling on it.

**Embeddings are computed locally, not bought from an API.** The corpus is
embedded once at ingestion and the query side has to be fast and free, so a
network round trip and a per-token bill on the hot path buys nothing here. The
cost is a ~200 MB CPU-only torch in the image and a 90 MB download on first
boot, both cached in volumes. `torch` is pinned to the `+cpu` build from
PyTorch's own index; the default PyPI wheel drags in ~2.5 GB of CUDA libraries
that no part of this can use.

**The embedding model is a `Protocol`, and the default test suite never loads
it.** Not for purity: a 90 MB download inside `make test` means a suite that
fails on a train and takes minutes when it does not. The fake produces real
normalised 384-dimension vectors, so everything downstream of it — pgvector,
the HNSW opclass, the SQL — is still exercised for real.

**A search that arrives before the model is 503, not 500.** The weights load in
the background, so "not ready yet" is a state that fixes itself within seconds.
`EmbeddingError` is the one domain error mapped to a 5xx status, because it is
the only one that is neither the caller's fault nor a bug.

**Structured output, not a tool call and not free text.** The model replies
with JSON constrained by `output_config.format`, so there is no preamble to
strip and no format instruction to be disobeyed. A tool call would express the
same thing with more moving parts, and the prompt does not need the model to
*do* anything — only to say one thing in one shape. What the schema explicitly
does not buy is trustworthy markers: it constrains shape, not content.

**The markers are the only citation channel.** An early sketch had the model
return its answer *and* a list of sources it used. Two channels can disagree,
and then something has to decide which one is the citation — so there is one:
the markers written inline where the claims are. The only other field is
`answered`, which exists so that an honest abstention is distinguishable from a
confident answer that forgot to cite, and those two need different handling.

**Answering is synchronous, and the connection is released across the call.**
Generation takes seconds, which is slow for an HTTP request and fine for this
scale; a job queue and a polling client would be more machinery than a
portfolio prototype earns, and streaming would arrive before resolution had
decided which markers survive. What the request does *not* do is hold a pooled
database connection while it waits.

**PyMuPDF is treated as untyped.** It ships `py.typed` but leaves its own
functions unannotated, so under `mypy --strict` every call into it is a "call to
untyped function" and iterating a `Document` reads as yielding `Never`. A
per-module override makes it `Any` at the boundary, which is the honest
description — `app/services/extraction.py` converts everything into typed
dataclasses before it returns, and those types are ours, not theirs.

**Known limitation, deferred deliberately:** ingestion runs on FastAPI
`BackgroundTasks` rather than a task queue, so in-flight work does not survive
an API restart and a document can be left in `extracting` forever. Swapping in
an `arq` worker is a later milestone, and a reaper for stuck rows belongs with
it. A smaller working system beats a half-finished Celery setup.

**The other known limitation:** there is no rate limiting and no
authentication, so anything that can reach `/questions` can spend money. That
is acceptable for something running on a laptop behind Docker and would be the
first thing to fix before this was exposed to anyone. `ANSWER_MAX_TOKENS` and
the retrieval `limit` bound the cost of a single question, not the number of
them.

## Roadmap

| Milestone | Scope                                                          |
| --------- | -------------------------------------------------------------- |
| **M1**    | Scaffolding, Docker, health checks — **done**                   |
| **M2**    | Content schema in migration `0001`, models, repositories, course CRUD — **done** |
| **M3**    | PDF upload, PyMuPDF extraction with page metadata, chunking — **done** |
| **M4**    | Sentence-Transformers embeddings, HNSW retrieval, `/search` — **done** |
| **M5**    | Cited answers via Claude, marker resolution, query log in `0002` — **done** |
| M6        | Course and Q&A screens, PDF pane that jumps to the cited page   |
| M7        | Golden-set eval harness (recall@k, citation precision), polish  |
