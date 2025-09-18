# Backend (Node + Express)

RAG backend for the news chatbot.

## Endpoints

- `POST /api/chat`
  - Body: `{ "message": string }`
  - Headers: Cookie-based `sessionId` auto-set if missing.
  - Returns: `{ reply, sources, sessionId }`

- `GET /api/session/:id/history`
  - Returns chat history for a session.

- `DELETE /api/session/:id`
  - Clears session history and resets any Redis keys.

## Ingestion

```bash
cp .env.example .env
npm install
npm run ingest
```
This script:
1. Fetches Reuters sitemap, collects ~50 article URLs.
2. Extracts article text (Unfluff).
3. Embeds text chunks via **Jina Embeddings**.
4. Upserts into **Qdrant** collection (`QDRANT_COLLECTION`).

## Config
- Qdrant local via Docker (`docker compose up -d`) or set `QDRANT_URL` / `QDRANT_API_KEY` to Qdrant Cloud.
- Redis for session history with TTL (`SESSION_TTL_SECONDS`).

## Caching & Performance
- **Redis** stores per-session message arrays under key `session:{sessionId}` with TTL.
- To enable cache warming, schedule a small script to ingest top headlines hourly:
  - Pre-embed the latest headlines and upsert to Qdrant.
  - Increase recall by increasing `TOP_K` or chunk overlap as needed.

## Design Decisions
- **Jina embeddings**: robust free tier, simple HTTP API.
- **Qdrant**: simple vector DB with powerful filtering and local/cloud flexibility.
- **Gemini**: cost-effective and fast (1.5-flash default).

## Potential Improvements
- Add webhooks to auto-refresh the news index periodically.
- Implement streaming via Server-Sent Events.
- Add reranking (e.g., `bge-reranker`) before final Gemini call.
- Persist transcripts to Postgres/MySQL (see `src/db/` placeholder).

## Run
```bash
npm run dev
```
