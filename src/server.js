import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import cookieParser from 'cookie-parser';
import morgan from 'morgan';
import { v4 as uuidv4 } from 'uuid';
import Redis from 'ioredis';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { QdrantClient } from '@qdrant/js-client-rest';
import axios from 'axios';

const app = express();
app.use(express.json({ limit: '2mb' }));
app.use(cors({ origin: true, credentials: true }));
app.use(cookieParser());
app.use(morgan('dev'));

// ---- Env
const PORT = process.env.PORT || 8080;
const SESSION_TTL_SECONDS = parseInt(process.env.SESSION_TTL_SECONDS || '86400', 10);

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-1.5-flash';
const GEMINI_FALLBACK_MODEL = process.env.GEMINI_FALLBACK_MODEL || 'gemini-1.5-flash-8b';

const JINA_API_KEY = process.env.JINA_API_KEY;
const JINA_MODEL = process.env.JINA_EMBEDDING_MODEL || 'jina-embeddings-v3';

const QDRANT_URL = process.env.QDRANT_URL || 'http://localhost:6333';
const QDRANT_API_KEY = process.env.QDRANT_API_KEY || '';
const QDRANT_COLLECTION = process.env.QDRANT_COLLECTION || 'news_articles_v1';

// ---- Redis client
const redis = new Redis(process.env.REDIS_URL || 'redis://localhost:6379', {
  lazyConnect: true,
});

// ---- Qdrant (REST)
const { hostname, port, protocol } = new URL(QDRANT_URL);
const qdrant = new QdrantClient({
  host: hostname,
  port: port ? Number(port) : (protocol === 'https:' ? 443 : 6333),
  apiKey: QDRANT_API_KEY || undefined,
  https: protocol === 'https:',
  checkCompatibility: false, // quiet local version mismatch
});

// ---- Google Gemini
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

// ---- Generation config + retry/fallback helpers
const GEN_CONFIG = {
  temperature: 0.2,
  maxOutputTokens: 512,
};

async function generateWithRetries(genAI, primaryModel, prompt, maxAttempts = 4) {
  const delays = [400, 800, 1600, 3200]; // ms
  let lastErr;

  // try primary, then fallback model if needed
  const modelsToTry = [primaryModel, GEMINI_FALLBACK_MODEL];

  for (const modelName of modelsToTry) {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const model = genAI.getGenerativeModel({ model: modelName });
        const result = await model.generateContent({
          contents: [{ role: 'user', parts: [{ text: prompt }] }],
          generationConfig: GEN_CONFIG,
        });
        return result.response.text();
      } catch (err) {
        lastErr = err;
        const status = err?.status || err?.response?.status;
        const retriable = status === 429 || status === 500 || status === 502 || status === 503 || status === 504;
        if (!retriable) break; // don't loop on permanent errors
        const wait = delays[Math.min(attempt, delays.length - 1)];
        await new Promise(r => setTimeout(r, wait));
      }
    }
    // tried all attempts for this model—move to next (fallback)
  }
  throw lastErr;
}

// ---- Helpers
async function ensureSession(req, res, next) {
  let { sessionId } = req.cookies;
  if (!sessionId) {
    sessionId = uuidv4();
    res.cookie('sessionId', sessionId, { httpOnly: false, sameSite: 'Lax' });
  }
  req.sessionId = sessionId;
  next();
}

async function getHistory(sessionId) {
  const key = `session:${sessionId}`;
  const history = await redis.get(key);
  return history ? JSON.parse(history) : [];
}

async function pushHistory(sessionId, role, content) {
  const key = `session:${sessionId}`;
  const history = await getHistory(sessionId);
  history.push({ role, content, ts: Date.now() });
  await redis.set(key, JSON.stringify(history), 'EX', SESSION_TTL_SECONDS);
}

async function embedTexts(texts) {
  // Jina Embeddings REST
  const url = `https://api.jina.ai/v1/embeddings`;
  const payload = {
    input: texts,
    model: JINA_MODEL,
  };
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${JINA_API_KEY}`,
  };
  const resp = await axios.post(url, payload, { headers });
  return resp.data.data.map(d => d.embedding);
}

async function retrieve(query, topK = 5) {
  const [qvec] = await embedTexts([query]);
  const search = await qdrant.search(QDRANT_COLLECTION, {
    vector: qvec,
    limit: topK,
    with_payload: true,
    score_threshold: 0.0,
  });
  return search.map(hit => ({
    score: hit.score,
    text: hit.payload?.text || '',
    url: hit.payload?.url || '',
    title: hit.payload?.title || '',
  }));
}

function buildRagPrompt(query, contexts) {
  const contextText = contexts.map((c, i) => `Source ${i+1} (${c.score.toFixed(3)}): ${c.title}
${c.text}
URL: ${c.url}`).join('\n\n---\n\n');
  return `You are a helpful assistant that answers questions using ONLY the provided news context.
Cite sources inline with [S1], [S2], etc., where the number corresponds to the source order. If you are unsure, say so briefly.

Question: ${query}

Context:
${contextText}

Answer:`;
}

// ---- Routes
app.post('/api/chat', ensureSession, async (req, res) => {
  try {
    const { message } = req.body || {};
    if (!message || !message.trim()) {
      return res.status(400).json({ error: 'message is required' });
    }
    await pushHistory(req.sessionId, 'user', message);

    const docs = await retrieve(message, 5);

    // Graceful fallback if no context was found
    if (!docs || docs.length === 0) {
      const info = 'No indexed news found for retrieval. Please run `npm run ingest` (and ensure NEWS_RSS_LIST and JINA_API_KEY are set).';
      await pushHistory(req.sessionId, 'assistant', info);
      return res.json({ reply: info, sources: [], sessionId: req.sessionId });
    }

    const prompt = buildRagPrompt(message, docs);

    // Gemini with retry + fallback model (e.g., flash -> flash-8b)
    const reply = await generateWithRetries(genAI, GEMINI_MODEL, prompt);

    await pushHistory(req.sessionId, 'assistant', reply);

    return res.json({
      reply,
      sources: docs.map((d, idx) => ({ id: `S${idx+1}`, title: d.title, url: d.url, score: d.score })),
      sessionId: req.sessionId,
    });
  } catch (err) {
    console.error(err);
    // Friendlier message instead of hard 500 for transient errors
    const friendly = 'The news model is busy right now. Please try again in a few seconds.';
    return res.status(200).json({ reply: friendly, sources: [], sessionId: req.sessionId });
  }
});

app.get('/api/session/:id/history', async (req, res) => {
  try {
    const history = await getHistory(req.params.id);
    return res.json({ sessionId: req.params.id, history });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: err.message });
  }
});

app.delete('/api/session/:id', async (req, res) => {
  try {
    const key = `session:${req.params.id}`;
    await redis.del(key);
    return res.json({ ok: true });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: err.message });
  }
});

app.get('/health', (req, res) => res.json({ ok: true }));

async function start() {
  await redis.connect().catch(() => {});
  app.listen(PORT, () => console.log(`Backend listening on http://localhost:${PORT}`));
}
start();
