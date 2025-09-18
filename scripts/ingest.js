import 'dotenv/config';
import axios from 'axios';
import { parseStringPromise } from 'xml2js';
import extractor from 'unfluff'; // kept for optional HTML parsing paths if you ever need it
import { QdrantClient } from '@qdrant/js-client-rest';
import crypto from 'crypto';
import * as cheerio from 'cheerio';

const {
  NEWS_SITEMAP,
  ARTICLE_LIMIT = '50',
  QDRANT_URL = 'http://localhost:6333',
  QDRANT_API_KEY = '',
  QDRANT_COLLECTION = 'news_articles_v1',
  JINA_API_KEY,
  JINA_EMBEDDING_MODEL = 'jina-embeddings-v3',
  // NEW: comma-separated list of RSS feeds (preferred over Reuters sitemap to avoid 401/451)
  NEWS_RSS_LIST = '',
} = process.env;

// Qdrant client (REST). We parse URL into host/port and disable strict version check for local dev.
const { hostname, port, protocol } = new URL(QDRANT_URL);
const qdrant = new QdrantClient({
  host: hostname,
  port: port ? Number(port) : (protocol === 'https:' ? 443 : 6333),
  apiKey: QDRANT_API_KEY || undefined,
  https: protocol === 'https:',
  checkCompatibility: false,
});

async function ensureCollection() {
  try {
    const exists = await qdrant.getCollection(QDRANT_COLLECTION).catch(() => null);
    if (!exists) throw new Error('no');
    console.log('Collection exists.');
  } catch {
    console.log('Creating collection', QDRANT_COLLECTION);
    await qdrant.createCollection(QDRANT_COLLECTION, {
      vectors: { size: 1024, distance: 'Cosine' }, // Jina v3 = 1024 dims
    });
  }
}

/** ---------- Reuters sitemap (kept for fallback if you don’t set RSS) ---------- */
async function getLatestNewsUrls(limit = 50) {
  console.log('Fetching sitemap index:', NEWS_SITEMAP);
  const res = await axios.get(NEWS_SITEMAP);
  const xml = await parseStringPromise(res.data);
  const sitemaps = xml?.sitemapindex?.sitemap || [];
  if (!sitemaps.length) throw new Error('No sitemaps found');
  const newsSitemapUrl = sitemaps[0].loc[0];
  console.log('Using sitemap:', newsSitemapUrl);
  const sm = await axios.get(newsSitemapUrl);
  const xml2 = await parseStringPromise(sm.data);
  const urls = (xml2?.urlset?.url || []).map(u => u.loc[0]);
  return urls.slice(0, limit);
}

/** Always use Jina Reader to avoid 401/451 issues when fetching full pages directly */
async function fetchAndExtract(url) {
  const readerUrl = 'https://r.jina.ai/' + url; // Example: https://r.jina.ai/https://www.reuters.com/...
  try {
    const r = await axios.get(readerUrl, {
      timeout: 25000,
      headers: {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'text/plain',
      },
      // Some hosts need this to avoid 451/geo headers; harmless if ignored
      validateStatus: s => s >= 200 && s < 400,
    });
    const md = (r.data || '').toString(); // Jina returns clean markdown-ish text
    if (!md || md.length < 200) {
      console.warn('Too short after reader:', url);
      return null;
    }

    // Try to pull a title from first heading in the markdown
    const m = md.match(/^\s*#\s+(.+)$/m);
    const title = (m ? m[1].trim() : url.split('/').slice(-2).join(' / ')) || 'Untitled';
    const text = md.trim();

    return { url, title, text };
  } catch (e) {
    console.warn('Reader fetch failed:', url, e.message);
    return null;
  }
}

/** ---------- RSS ingestion (preferred) ---------- */
const RSS_LIST = (NEWS_RSS_LIST || '')
  .split(',')
  .map(s => s.trim())
  .filter(Boolean);

async function getItemsFromOneRss(url) {
  try {
    const res = await axios.get(url, { timeout: 20000, headers: { 'User-Agent': 'Mozilla/5.0' } });
    const feed = await parseStringPromise(res.data);

    // Try RSS 2.0 path
    let items = feed?.rss?.channel?.[0]?.item || [];

    // Try Atom path if needed
    if (!items.length && Array.isArray(feed?.feed?.entry)) {
      items = feed.feed.entry.map(e => ({
        title: [e.title?.[0]?._ || e.title?.[0] || 'Untitled'],
        link: [e.link?.[0]?.$.href || ''],
        description: [e.summary?.[0]?._ || e.summary?.[0] || ''],
        'content:encoded': [e.content?.[0]?._ || e.content?.[0] || ''],
      }));
    }

    return items.map(it => {
      const link = (it.link && it.link[0]) || (it.guid && (it.guid[0]?._ || it.guid[0])) || '';
      const title = (it.title && (it.title[0]?._ || it.title[0])) || 'Untitled';
      const encoded = it['content:encoded'] ? (it['content:encoded'][0]?._ || it['content:encoded'][0]) : '';
      const desc = (it.description && (it.description[0]?._ || it.description[0])) || '';
      // Prefer content:encoded (often full text), else description
      const raw = (encoded || desc || '').toString();
      // strip HTML tags crudely
      const text = raw.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
      return { url: link, title, text };
    }).filter(x => x.url && x.text && x.text.length >= 200);
  } catch (e) {
    console.warn('RSS fetch failed:', url, e.message);
    return [];
  }
}

async function getLatestNewsFromRss(limit = 50) {
  const all = [];
  for (const url of RSS_LIST) {
    const items = await getItemsFromOneRss(url);
    for (const it of items) {
      all.push(it);
      if (all.length >= limit) break;
    }
    if (all.length >= limit) break;
  }
  return all;
}

/** ---- NEW: safety cap to avoid huge memory usage per article ---- */
function truncateText(str, maxChars = 60000) {
  if (!str) return '';
  return str.length > maxChars ? str.slice(0, maxChars) : str;
}

/** ---------- Chunking, embedding, upsert ---------- */
function chunkText(text, chunkSize = 800, overlap = 80, maxChunks = 80) {
  const chunks = [];
  let i = 0;
  const maxLen = text.length;
  while (i < maxLen && chunks.length < maxChunks) {
    const end = Math.min(i + chunkSize, maxLen);
    chunks.push(text.slice(i, end));
    i = end - overlap;
    if (i < 0) i = 0;
  }
  return chunks;
}

async function embedBatch(texts) {
  const resp = await axios.post('https://api.jina.ai/v1/embeddings', {
    input: texts,
    model: JINA_EMBEDDING_MODEL,
  }, {
    headers: {
      'Authorization': `Bearer ${JINA_API_KEY}`,
      'Content-Type': 'application/json',
    }
  });
  return resp.data.data.map(d => d.embedding);
}

/** ---- NEW: batch upserts to keep memory stable ---- */
async function upsert(chunks, meta) {
  const BATCH = 32; // adjust to 16 if your RAM is tight
  for (let i = 0; i < chunks.length; i += BATCH) {
    const slice = chunks.slice(i, i + BATCH);

    // embed this small batch
    const vectors = await embedBatch(slice);

    // build points for this batch only
    const points = vectors.map((vec, j) => ({
      id: crypto.createHash('md5').update(meta.url + '#' + (i + j)).digest('hex'),
      vector: vec,
      payload: {
        url: meta.url,
        title: meta.title,
        text: slice[j],
      },
    }));

    // upsert this batch only
    await qdrant.upsert(QDRANT_COLLECTION, { points });
  }
}

/** ---------- Main ---------- */
async function main() {
  if (!JINA_API_KEY) throw new Error('Missing JINA_API_KEY');
  await ensureCollection();

  const limit = parseInt(ARTICLE_LIMIT, 10);
  let docs = [];

  if (RSS_LIST.length) {
    console.log('Using RSS list:', RSS_LIST.join(', '));
    docs = await getLatestNewsFromRss(limit);
  } else {
    // Fallback to Reuters sitemap (may 401/451 depending on region); kept for completeness
    const urls = await getLatestNewsUrls(limit);
    const tmp = [];
    for (const url of urls) {
      const data = await fetchAndExtract(url);
      if (data && data.text && data.text.length >= 200) tmp.push(data);
    }
    docs = tmp;
  }

  console.log('Indexing', docs.length, 'articles');

  for (const data of docs) {
    try {
      // cap very long items to keep memory predictable
      data.text = truncateText(data.text, 60000);

      const chunks = chunkText(data.text, 800, 80, 80);
      console.log(`Upserting ${chunks.length} chunks for`, data.title);
      await upsert(chunks, { url: data.url, title: data.title });
    } catch (e) {
      console.warn('Upsert failed for', data.url, e.message);
    }
  }

  console.log('Done.');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
