# Cloudflare Worker — Echo Scribe RunPod Proxy

Sits between the React frontend and RunPod Serverless. Translates the frontend's multipart form upload into RunPod's JSON format, polls for completion, and returns the transcript. **No frontend code changes needed.**

```
[React Frontend] → [This Worker] → [RunPod Serverless GPU]
     same API          converts         transcribes
     as before         form→base64→JSON  and returns
```

## Setup

```bash
cd cloudflare-worker
npm install
```

## Configure Secrets

```bash
# Your RunPod API key
npx wrangler secret put RUNPOD_API_KEY

# Your RunPod serverless endpoint ID (from the RunPod dashboard)
npx wrangler secret put RUNPOD_ENDPOINT_ID

# Comma-separated API keys that the frontend can use (same keys as your api-keys.json)
npx wrangler secret put API_KEYS
```

## Local Development

```bash
npm run dev
```

This starts a local dev server (default `http://localhost:8787`). You'll be prompted for secret values on first run.

To test with the frontend, set the `BASE` URL in `frontend/src/api.ts`:

```ts
const BASE = import.meta.env.DEV ? "http://localhost:8787" : "";
```

Then run the frontend dev server (`cd frontend && npm run dev`).

## Deploy

```bash
npm run deploy
```

This deploys to `echo-scribe-proxy.<your-account>.workers.dev`.

### Custom Domain (Optional)

To use a custom domain (e.g., `api-scribe.mvp-scale.com`), uncomment the routes section in `wrangler.toml`:

```toml
[routes]
routes = [
  { pattern = "api-scribe.mvp-scale.com/*", zone_name = "mvp-scale.com" }
]
```

Then update the frontend's `BASE` to point at the custom domain.

## How It Works

### Request Flow

1. Frontend sends `POST /v1/audio/transcriptions` with multipart form data (file + options)
2. Worker validates the `Authorization: Bearer {key}` header against `API_KEYS`
3. Worker reads the audio file, converts to base64
4. Worker builds a JSON payload matching the RunPod handler's input format
5. Worker submits to RunPod via `/run` (async)
6. Worker polls RunPod `/status/{job_id}` every 2 seconds until complete (up to 10 min)
7. Worker returns the transcript JSON to the frontend

### Endpoints Proxied

| Endpoint | Behavior |
|---|---|
| `POST /v1/audio/transcriptions` | Proxied to RunPod (form → base64 → JSON) |
| `GET /health` | Synthetic OK response |
| `GET /v1/audio/usage` | Returns `null` (no usage tracking in serverless) |
| `POST /v1/audio/entities` | 501 — use `detect_entities` flag during transcription |
| `POST /v1/audio/sentiment` | 501 — use `detect_sentiment` flag during transcription |

## Requirements

- **Cloudflare Workers paid plan** ($5/month) — the free plan's 10ms CPU limit is too low
- The paid plan gives 30s CPU time and unlimited wall clock time for polling

## Connecting the Frontend

There are two approaches:

### Option A: Subdomain (Recommended)

Deploy the Worker to `api-scribe.mvp-scale.com` and set `BASE` in the frontend:

```ts
const BASE = "https://api-scribe.mvp-scale.com";
```

### Option B: Same Domain with Route

If the frontend is on Cloudflare Pages at `scribe.mvp-scale.com`, use Workers Routes to intercept `/v1/*` and `/health` on the same domain. The frontend's `BASE` stays empty (same origin).

## Troubleshooting

### "Missing API key" error
The `API_KEYS` secret must be set. It's a comma-separated list, e.g., `sk-key1,sk-key2`.

### Transcription times out (504)
The Worker polls for up to 10 minutes. If your audio is extremely long, the RunPod execution timeout (set in the RunPod dashboard) may need to be increased.

### Large file uploads fail
Cloudflare Workers support up to 100MB request bodies on the paid plan. For larger files, use `audio_url` instead (pass a URL the RunPod worker can download from).
