/**
 * Cloudflare Worker — proxy between the React frontend and RunPod Serverless.
 *
 * Translates the frontend's multipart form upload into RunPod's JSON format,
 * polls for completion, and returns the transcript. The frontend doesn't need
 * any code changes.
 *
 * Audio files are stored temporarily in R2 and RunPod downloads them via URL.
 * R2 lifecycle rule auto-deletes files after 1 day.
 *
 * Secrets (set via `wrangler secret put`):
 *   RUNPOD_API_KEY  — your RunPod API key
 *   RUNPOD_ENDPOINT_ID — your RunPod serverless endpoint ID
 *   API_KEYS — comma-separated list of valid Bearer tokens for frontend auth
 */

export interface Env {
	RUNPOD_API_KEY: string;
	RUNPOD_ENDPOINT_ID: string;
	API_KEYS: string; // comma-separated
	PAGES_URL: string; // e.g. https://echo-scribe.pages.dev
	UPLOADS: R2Bucket; // temp audio file storage
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		// CORS preflight
		if (request.method === "OPTIONS") {
			return corsResponse(204);
		}

		const url = new URL(request.url);
		const path = url.pathname;

		try {
			// Transcription — main endpoint
			if (path === "/v1/audio/transcriptions" && request.method === "POST") {
				return await handleTranscription(request, env);
			}

			// Temp file download — RunPod fetches audio from here
			if (path.startsWith("/tmp/") && request.method === "GET") {
				return await serveTempFile(path, env);
			}

			// Health check — synthetic response
			if (path === "/health" && request.method === "GET") {
				return jsonResponse({
					status: "ok",
					backend: "runpod-serverless",
					engine: "nemo",
					model_loaded: true,
					model_id: "nvidia/parakeet-tdt-0.6b-v2",
					model_name: "parakeet-tdt-0.6b-v2",
					cuda_available: true,
					gpu_name: "RunPod Serverless GPU",
					diarization_available: true,
				});
			}

			// Usage — not tracked in serverless mode
			if (path === "/v1/audio/usage" && request.method === "GET") {
				return jsonResponse(null);
			}

			// Entity/sentiment annotation — not available standalone in serverless.
			// Use detect_entities/detect_sentiment flags during transcription instead.
			if (path === "/v1/audio/entities" || path === "/v1/audio/sentiment") {
				return jsonResponse(
					{ detail: "Not available in serverless mode. Enable detect_entities/detect_sentiment during transcription." },
					501,
				);
			}

			// Everything else — proxy to Cloudflare Pages (serves the React frontend)
			return fetchFromPages(request, env);
		} catch (err: any) {
			console.error("Worker error:", err);
			return jsonResponse({ detail: err.message || "Internal error" }, 500);
		}
	},
} satisfies ExportedHandler<Env>;

// ---------------------------------------------------------------------------
// Transcription handler
// ---------------------------------------------------------------------------

async function handleTranscription(request: Request, env: Env): Promise<Response> {
	// 1. Validate auth
	const authErr = validateAuth(request, env);
	if (authErr) return authErr;

	// 2. Parse multipart form
	const form = await request.formData();
	const file = form.get("file");
	if (!file || !(file instanceof File)) {
		return jsonResponse({ detail: "Missing 'file' field" }, 400);
	}

	// 3. Store audio in R2, pass URL to RunPod
	const fileKey = crypto.randomUUID();
	const arrayBuffer = await file.arrayBuffer();
	await env.UPLOADS.put(fileKey, arrayBuffer, {
		httpMetadata: { contentType: file.type || "application/octet-stream" },
	});
	const workerUrl = new URL(request.url).origin;
	const audioUrl = `${workerUrl}/tmp/${fileKey}`;
	console.log(`Stored ${file.name} (${(arrayBuffer.byteLength / 1024 / 1024).toFixed(1)}MB) in R2: ${fileKey}`);

	// 4. Build RunPod input (map form fields to handler params)
	const input: Record<string, any> = {
		audio_url: audioUrl,
		filename: file.name,
		response_format: formStr(form, "response_format", "verbose_json"),
		diarize: formBool(form, "diarize", true),
		include_diarization_in_text: formBool(form, "include_diarization_in_text", true),
		detect_paragraphs: formBool(form, "detect_paragraphs", false),
		paragraph_silence_threshold: formFloat(form, "paragraph_silence_threshold", 0.8),
		remove_fillers: formBool(form, "remove_fillers", false),
		min_confidence: formFloat(form, "min_confidence", 0.0),
		detect_entities: formBool(form, "detect_entities", false),
		detect_topics: formBool(form, "detect_topics", false),
		detect_sentiment: formBool(form, "detect_sentiment", false),
	};

	// Optional fields (only include if present)
	const optStr = (key: string) => { const v = form.get(key); if (v) input[key] = String(v); };
	const optInt = (key: string) => { const v = form.get(key); if (v) input[key] = parseInt(String(v), 10); };
	optStr("language");
	optInt("num_speakers");
	optInt("min_speakers");
	optInt("max_speakers");
	optStr("find_replace");
	optStr("text_rules");
	optStr("speaker_labels");

	if (formBool(form, "word_timestamps", false)) input.word_timestamps = true;
	if (formBool(form, "timestamps", false)) input.timestamps = true;

	// 5. Submit to RunPod (async)
	const runpodBase = `https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}`;
	const headers = {
		"Authorization": `Bearer ${env.RUNPOD_API_KEY}`,
		"Content-Type": "application/json",
	};

	const submitRes = await fetch(`${runpodBase}/run`, {
		method: "POST",
		headers,
		body: JSON.stringify({ input }),
	});

	if (!submitRes.ok) {
		const errBody = await submitRes.text();
		console.error("RunPod submit failed:", submitRes.status, errBody);
		return jsonResponse({ detail: `RunPod error ${submitRes.status}: ${errBody}` }, 502);
	}

	const submitData: any = await submitRes.json();
	const jobId = submitData.id;

	if (!jobId) {
		return jsonResponse({ detail: "RunPod did not return a job ID" }, 502);
	}

	// 6. Poll for completion
	const POLL_INTERVAL_MS = 2000;
	const MAX_POLL_TIME_MS = 10 * 60 * 1000; // 10 minutes
	const startTime = Date.now();

	while (Date.now() - startTime < MAX_POLL_TIME_MS) {
		await sleep(POLL_INTERVAL_MS);

		const statusRes = await fetch(`${runpodBase}/status/${jobId}`, { headers });
		if (!statusRes.ok) {
			console.error("RunPod status check failed:", statusRes.status);
			continue;
		}

		const statusData: any = await statusRes.json();
		const status = statusData.status;

		if (status === "COMPLETED") {
			// Clean up R2 file immediately (don't wait for lifecycle)
			await env.UPLOADS.delete(fileKey);
			const output = statusData.output;
			if (output && output.error) {
				return jsonResponse({ detail: output.error }, 500);
			}
			return jsonResponse(output);
		}

		if (status === "FAILED") {
			await env.UPLOADS.delete(fileKey);
			const errMsg = statusData.error || "RunPod job failed";
			console.error("RunPod job failed:", errMsg);
			return jsonResponse({ detail: errMsg }, 500);
		}

		// IN_QUEUE or IN_PROGRESS — keep polling
	}

	await env.UPLOADS.delete(fileKey);
	return jsonResponse({ detail: "Transcription timed out" }, 504);
}

// ---------------------------------------------------------------------------
// Temp file serving — RunPod downloads audio from here
// ---------------------------------------------------------------------------

async function serveTempFile(path: string, env: Env): Promise<Response> {
	const key = path.replace("/tmp/", "");
	const object = await env.UPLOADS.get(key);

	if (!object) {
		return new Response("Not found or expired", { status: 404 });
	}

	return new Response(object.body, {
		headers: {
			"Content-Type": object.httpMetadata?.contentType || "application/octet-stream",
			"Content-Length": String(object.size),
		},
	});
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

function validateAuth(request: Request, env: Env): Response | null {
	const authHeader = request.headers.get("Authorization");
	if (!authHeader || !authHeader.startsWith("Bearer ")) {
		return jsonResponse({ detail: "Missing API key" }, 401);
	}

	const token = authHeader.slice(7);
	const validKeys = env.API_KEYS.split(",").map((k) => k.trim());

	if (!validKeys.includes(token)) {
		return jsonResponse({ detail: "Invalid API key" }, 401);
	}

	return null; // auth OK
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

function formStr(form: FormData, key: string, fallback: string): string {
	const v = form.get(key);
	return v ? String(v) : fallback;
}

function formBool(form: FormData, key: string, fallback: boolean): boolean {
	const v = form.get(key);
	if (v === null) return fallback;
	return String(v).toLowerCase() === "true";
}

function formFloat(form: FormData, key: string, fallback: number): number {
	const v = form.get(key);
	if (v === null) return fallback;
	const parsed = parseFloat(String(v));
	return isNaN(parsed) ? fallback : parsed;
}

function jsonResponse(data: any, status = 200): Response {
	return new Response(JSON.stringify(data), {
		status,
		headers: {
			"Content-Type": "application/json",
			"Access-Control-Allow-Origin": "*",
			"Access-Control-Allow-Methods": "GET, POST, OPTIONS",
			"Access-Control-Allow-Headers": "Content-Type, Authorization",
		},
	});
}

function corsResponse(status = 204): Response {
	return new Response(null, {
		status,
		headers: {
			"Access-Control-Allow-Origin": "*",
			"Access-Control-Allow-Methods": "GET, POST, OPTIONS",
			"Access-Control-Allow-Headers": "Content-Type, Authorization",
			"Access-Control-Max-Age": "86400",
		},
	});
}

// ---------------------------------------------------------------------------
// Pages proxy — serves the React frontend for non-API routes
// ---------------------------------------------------------------------------

async function fetchFromPages(request: Request, env: Env): Promise<Response> {
	const url = new URL(request.url);
	const pagesUrl = new URL(url.pathname + url.search, env.PAGES_URL);
	const res = await fetch(pagesUrl.toString(), {
		method: request.method,
		headers: request.headers,
	});

	// SPA fallback: if Pages returns 404, serve index.html
	if (res.status === 404) {
		const fallback = await fetch(new URL("/index.html", env.PAGES_URL).toString());
		return new Response(fallback.body, {
			status: 200,
			headers: fallback.headers,
		});
	}

	return res;
}
